// MPSGraph training engine for mongoose — full forward+backward in one dispatch.
//
// Architecture: one MPSGraph encodes the entire transformer training step.
// Apple's graph compiler fuses operations, picks optimal tile sizes per GPU,
// and computes gradients automatically via autograd.
//
// The graph is compiled once (first step), then reused. Each step feeds
// token IDs and weight buffers, gets back loss value. Gradients are
// computed inside the graph and written directly to gradient buffers.
//
// This is the same architecture PyTorch MPS uses internally.

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <math.h>

// Forward declarations from metal_impl.m
extern id<MTLDevice> g_device;
extern id<MTLCommandQueue> g_queue;

// --- Training Graph ---

@interface MongooseTrainGraph : NSObject

@property MPSGraph* graph;
@property MPSGraphTensor* inputTokens;   // [n] int32
@property MPSGraphTensor* targetTokens;  // [n-1] int32
@property MPSGraphTensor* loss;          // scalar

// All weight placeholders in order (embed, finalNorm, then 12 per layer)
@property NSMutableArray<MPSGraphTensor*>* weightPlaceholders;

// Gradient tensors (same order as weightPlaceholders)
@property NSMutableArray<MPSGraphTensor*>* gradientTensors;

// Learning rate placeholder (fed each step)
@property MPSGraphTensor* lrPlaceholder;

// Assign operations (trigger weight updates — graph Adam path)
@property NSMutableArray<MPSGraphOperation*>* updateOps;

// Differentiable variables (for gradient readback — split path)
@property NSMutableArray<MPSGraphTensor*>* diffableVars;

// Mode 2 (graph-accumulate): gradient accumulator variables + ops
@property NSMutableArray<MPSGraphTensor*>* accumVars;     // grad accumulators (same shape as weights)
@property NSMutableArray<MPSGraphOperation*>* accumOps;   // accum += grad (run each micro-batch)
@property NSMutableArray<MPSGraphOperation*>* adamAccumOps; // adam(accum/N) + zero accum (run once per window)
@property MPSGraphTensor* accumScalePlaceholder;          // 1/N scaling factor

@property int dim, kvDim, headDim, nHeads, nKVHeads, ffnDim, vocabSize, nLayers, seqLen;

@end

@implementation MongooseTrainGraph
@end

static MongooseTrainGraph* g_train_graph = nil;

// --- Subgraph helpers ---

// RMSNorm: out = x / sqrt(mean(x^2) + eps) * weight
static MPSGraphTensor* buildRMSNorm(MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* weight, int dim) {
    // x is [n, dim]. RMSNorm: out = x * weight / sqrt(mean(x^2) + eps)
    // Decomposed into ops that all have gradient support.
    MPSGraphTensor* xSq = [g multiplicationWithPrimaryTensor:x secondaryTensor:x name:nil];
    MPSGraphTensor* mean = [g meanOfTensor:xSq axes:@[@(-1)] name:nil]; // [n, 1]
    MPSGraphTensor* eps = [g constantWithScalar:1e-6 dataType:MPSDataTypeFloat32];
    MPSGraphTensor* variance = [g additionWithPrimaryTensor:mean secondaryTensor:eps name:nil];
    MPSGraphTensor* rms = [g squareRootWithTensor:variance name:nil];
    // x * weight / rms — use division (has gradient)
    MPSGraphTensor* xw = [g multiplicationWithPrimaryTensor:x secondaryTensor:weight name:nil];
    return [g divisionWithPrimaryTensor:xw secondaryTensor:rms name:nil];
}

// RoPE: rotary position embeddings applied to x[n, nHeads*headDim]
static MPSGraphTensor* buildRoPE(MPSGraph* g, MPSGraphTensor* x,
                                  int n, int nHeads, int headDim, float theta) {
    int halfDim = headDim / 2;

    // Precompute cos/sin as constants: [n, 1, halfDim]
    float* cosData = (float*)malloc(n * halfDim * sizeof(float));
    float* sinData = (float*)malloc(n * halfDim * sizeof(float));
    for (int pos = 0; pos < n; pos++) {
        for (int i = 0; i < halfDim; i++) {
            float freq = 1.0f / powf(theta, (float)(2*i) / (float)headDim);
            float v = (float)pos * freq;
            cosData[pos * halfDim + i] = cosf(v);
            sinData[pos * halfDim + i] = sinf(v);
        }
    }
    NSData* cosNS = [NSData dataWithBytes:cosData length:n*halfDim*sizeof(float)];
    NSData* sinNS = [NSData dataWithBytes:sinData length:n*halfDim*sizeof(float)];
    free(cosData); free(sinData);

    MPSGraphTensor* cosT = [g constantWithData:cosNS shape:@[@(n), @1, @(halfDim)] dataType:MPSDataTypeFloat32];
    MPSGraphTensor* sinT = [g constantWithData:sinNS shape:@[@(n), @1, @(halfDim)] dataType:MPSDataTypeFloat32];

    // Reshape to [n, nHeads, headDim]
    int totalDim = nHeads * headDim;
    MPSGraphTensor* xr = [g reshapeTensor:x withShape:@[@(n), @(nHeads), @(headDim)] name:nil];

    // Split into pairs: reshape [n, nHeads, headDim] → [n, nHeads, halfDim, 2] then slice
    // Split pairs: [n, nHeads, headDim] → [n, nHeads, halfDim, 2] → slice → reshape (no squeeze!)
    MPSGraphTensor* xPairs = [g reshapeTensor:xr withShape:@[@(n), @(nHeads), @(halfDim), @2] name:nil];
    MPSGraphTensor* xEven = [g sliceTensor:xPairs dimension:3 start:0 length:1 name:nil];
    xEven = [g reshapeTensor:xEven withShape:@[@(n), @(nHeads), @(halfDim)] name:nil];
    MPSGraphTensor* xOdd = [g sliceTensor:xPairs dimension:3 start:1 length:1 name:nil];
    xOdd = [g reshapeTensor:xOdd withShape:@[@(n), @(nHeads), @(halfDim)] name:nil];

    // Rotate: even' = even*cos - odd*sin, odd' = even*sin + odd*cos
    MPSGraphTensor* outEven = [g subtractionWithPrimaryTensor:[g multiplicationWithPrimaryTensor:xEven secondaryTensor:cosT name:nil]
                                              secondaryTensor:[g multiplicationWithPrimaryTensor:xOdd secondaryTensor:sinT name:nil] name:nil];
    MPSGraphTensor* outOdd = [g additionWithPrimaryTensor:[g multiplicationWithPrimaryTensor:xEven secondaryTensor:sinT name:nil]
                                          secondaryTensor:[g multiplicationWithPrimaryTensor:xOdd secondaryTensor:cosT name:nil] name:nil];

    // Interleave: reshape to add trailing dim, concat, reshape back
    MPSGraphTensor* ee = [g reshapeTensor:outEven withShape:@[@(n), @(nHeads), @(halfDim), @1] name:nil];
    MPSGraphTensor* oe = [g reshapeTensor:outOdd withShape:@[@(n), @(nHeads), @(halfDim), @1] name:nil];
    MPSGraphTensor* stacked = [g concatTensors:@[ee, oe] dimension:3 name:nil]; // [n, nHeads, halfDim, 2]
    MPSGraphTensor* result = [g reshapeTensor:stacked withShape:@[@(n), @(nHeads), @(headDim)] name:nil];
    return [g reshapeTensor:result withShape:@[@(n), @(totalDim)] name:nil];
}

// --- Build full graph ---

// mode: 0 = split (placeholders, gradient output), 1 = graph-Adam (variables, internal optimizer)
int mtl_graph_build_full(int dim, int kvDim, int headDim,
                         int nHeads, int nKVHeads, int ffnDim,
                         int vocabSize, int nLayers, int seqLen,
                         float ropeTheta, int mode) {
    if (g_train_graph) return 0;

    int n = seqLen + 1;
    MongooseTrainGraph* tg = [[MongooseTrainGraph alloc] init];
    tg.dim = dim; tg.kvDim = kvDim; tg.headDim = headDim;
    tg.nHeads = nHeads; tg.nKVHeads = nKVHeads; tg.ffnDim = ffnDim;
    tg.vocabSize = vocabSize; tg.nLayers = nLayers; tg.seqLen = seqLen;

    MPSGraph* g = [[MPSGraph alloc] init];
    tg.graph = g;
    tg.weightPlaceholders = [[NSMutableArray alloc] init];

    // --- Inputs ---
    tg.inputTokens = [g placeholderWithShape:@[@(n)] dataType:MPSDataTypeInt32 name:@"tokens"];
    tg.targetTokens = [g placeholderWithShape:@[@(n-1)] dataType:MPSDataTypeInt32 name:@"targets"];

    // --- Weight creation (mode-dependent) ---
    NSArray* layerShapes = @[
        @[@1, @(dim)], @[@(dim),@(dim)], @[@(kvDim),@(dim)], @[@(kvDim),@(dim)],
        @[@(dim),@(dim)], @[@1, @(dim)], @[@1, @(kvDim)], @[@1, @(kvDim)],
        @[@1, @(dim)], @[@(ffnDim),@(dim)], @[@(ffnDim),@(dim)], @[@(dim),@(ffnDim)]
    ];

    // Helper to create either a placeholder (split mode) or variable (graph-Adam mode)
    MPSGraphTensor* (^makeWeight)(NSArray<NSNumber*>*, NSString*) = ^MPSGraphTensor*(NSArray<NSNumber*>* shape, NSString* name) {
        if (mode == 0) {
            // Split: placeholder fed from shared buffer each step
            MPSGraphTensor* ph = [g placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:name];
            [tg.weightPlaceholders addObject:ph];
            return ph;
        } else {
            // Graph-Adam: variable persists inside graph, initialized to zero
            int sz = 1;
            for (NSNumber* s in shape) sz *= [s intValue];
            NSMutableData* zeros = [NSMutableData dataWithLength:sz * sizeof(float)];
            MPSGraphTensor* v = [g variableWithData:zeros shape:shape dataType:MPSDataTypeFloat32 name:name];
            [tg.weightPlaceholders addObject:v];
            return v;
        }
    };

    MPSGraphTensor* cEmbedW = makeWeight(@[@(vocabSize), @(dim)], @"embed");
    MPSGraphTensor* cFinalNormW = makeWeight(@[@1, @(dim)], @"fnorm");

    for (int l = 0; l < nLayers; l++) {
        for (int w = 0; w < 12; w++) {
            NSString* name = [NSString stringWithFormat:@"L%d_w%d", l, w];
            makeWeight(layerShapes[w], name);
        }
    }

    // --- Forward pass ---

    MPSGraphTensor* tokOneHot = [g oneHotWithIndicesTensor:tg.inputTokens depth:vocabSize axis:-1
                                                  dataType:MPSDataTypeFloat32 onValue:1.0f offValue:0.0f name:@"tok_oh"];
    MPSGraphTensor* hidden = [g matrixMultiplicationWithPrimaryTensor:tokOneHot
                                                     secondaryTensor:cEmbedW name:@"embed"];

    // Causal mask
    float* maskData = (float*)malloc(n*n*sizeof(float));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            maskData[i*n+j] = (j > i) ? -1e9f : 0.0f;
    NSData* maskNS = [NSData dataWithBytes:maskData length:n*n*sizeof(float)];
    MPSGraphTensor* causalMask = [g constantWithData:maskNS shape:@[@1,@(n),@(n)] dataType:MPSDataTypeFloat32];
    free(maskData);

    MPSGraphTensor* attnScale = [g constantWithScalar:1.0f/sqrtf((float)headDim) dataType:MPSDataTypeFloat32];

    for (int l = 0; l < nLayers; l++) {
        int base = 2 + l * 12;
        MPSGraphTensor* norm1W = tg.weightPlaceholders[base+0];
        MPSGraphTensor* wq = tg.weightPlaceholders[base+1];
        MPSGraphTensor* wk = tg.weightPlaceholders[base+2];
        MPSGraphTensor* wv = tg.weightPlaceholders[base+3];
        MPSGraphTensor* wo = tg.weightPlaceholders[base+4];
        MPSGraphTensor* bq = tg.weightPlaceholders[base+5];
        MPSGraphTensor* bk = tg.weightPlaceholders[base+6];
        MPSGraphTensor* bv = tg.weightPlaceholders[base+7];
        MPSGraphTensor* norm2W = tg.weightPlaceholders[base+8];
        MPSGraphTensor* wg = tg.weightPlaceholders[base+9];
        MPSGraphTensor* wu = tg.weightPlaceholders[base+10];
        MPSGraphTensor* wd = tg.weightPlaceholders[base+11];

        // RMSNorm
        MPSGraphTensor* normed = buildRMSNorm(g, hidden, norm1W, dim);

        // QKV + bias (biases are [1, dim] / [1, kvDim] — broadcast against [n, dim])
        MPSGraphTensor* Q = [g additionWithPrimaryTensor:
                             [g matrixMultiplicationWithPrimaryTensor:normed
                                secondaryTensor:[g transposeTensor:wq dimension:0 withDimension:1 name:nil] name:nil]
                                              secondaryTensor:bq name:nil];
        MPSGraphTensor* K = [g additionWithPrimaryTensor:
                             [g matrixMultiplicationWithPrimaryTensor:normed
                                secondaryTensor:[g transposeTensor:wk dimension:0 withDimension:1 name:nil] name:nil]
                                              secondaryTensor:bk name:nil];
        MPSGraphTensor* V = [g additionWithPrimaryTensor:
                             [g matrixMultiplicationWithPrimaryTensor:normed
                                secondaryTensor:[g transposeTensor:wv dimension:0 withDimension:1 name:nil] name:nil]
                                              secondaryTensor:bv name:nil];

        // RoPE
        Q = buildRoPE(g, Q, n, nHeads, headDim, ropeTheta);
        K = buildRoPE(g, K, n, nKVHeads, headDim, ropeTheta);

        // Multi-head reshape: [n, dim] → [nHeads, n, headDim]
        MPSGraphTensor* Qh = [g transposeTensor:[g reshapeTensor:Q withShape:@[@(n),@(nHeads),@(headDim)] name:nil] dimension:0 withDimension:1 name:nil];
        MPSGraphTensor* Kh = [g transposeTensor:[g reshapeTensor:K withShape:@[@(n),@(nKVHeads),@(headDim)] name:nil] dimension:0 withDimension:1 name:nil];
        MPSGraphTensor* Vh = [g transposeTensor:[g reshapeTensor:V withShape:@[@(n),@(nKVHeads),@(headDim)] name:nil] dimension:0 withDimension:1 name:nil];

        // GQA repeat — use concat instead of tile (tile has no gradient in MPSGraph)
        if (nKVHeads < nHeads) {
            int rep = nHeads / nKVHeads;
            NSMutableArray<MPSGraphTensor*>* kParts = [[NSMutableArray alloc] init];
            NSMutableArray<MPSGraphTensor*>* vParts = [[NSMutableArray alloc] init];
            for (int r = 0; r < rep; r++) {
                [kParts addObject:Kh];
                [vParts addObject:Vh];
            }
            Kh = [g concatTensors:kParts dimension:0 name:nil];
            Vh = [g concatTensors:vParts dimension:0 name:nil];
        }

        // Attention
        MPSGraphTensor* scores = [g multiplicationWithPrimaryTensor:
                                  [g matrixMultiplicationWithPrimaryTensor:Qh
                                      secondaryTensor:[g transposeTensor:Kh dimension:1 withDimension:2 name:nil] name:nil]
                                                    secondaryTensor:attnScale name:nil];
        scores = [g additionWithPrimaryTensor:scores secondaryTensor:causalMask name:nil];
        scores = [g softMaxWithTensor:scores axis:-1 name:nil];
        MPSGraphTensor* attn = [g matrixMultiplicationWithPrimaryTensor:scores secondaryTensor:Vh name:nil];

        // Reshape back + WO + residual
        MPSGraphTensor* attnFlat = [g reshapeTensor:[g transposeTensor:attn dimension:0 withDimension:1 name:nil]
                                          withShape:@[@(n),@(dim)] name:nil];
        MPSGraphTensor* proj = [g matrixMultiplicationWithPrimaryTensor:attnFlat
                                   secondaryTensor:[g transposeTensor:wo dimension:0 withDimension:1 name:nil] name:nil];
        hidden = [g additionWithPrimaryTensor:hidden secondaryTensor:proj name:nil];

        // FFN
        MPSGraphTensor* normed2 = buildRMSNorm(g, hidden, norm2W, dim);
        MPSGraphTensor* gate = [g matrixMultiplicationWithPrimaryTensor:normed2
                                   secondaryTensor:[g transposeTensor:wg dimension:0 withDimension:1 name:nil] name:nil];
        MPSGraphTensor* up = [g matrixMultiplicationWithPrimaryTensor:normed2
                                 secondaryTensor:[g transposeTensor:wu dimension:0 withDimension:1 name:nil] name:nil];
        MPSGraphTensor* silu = [g multiplicationWithPrimaryTensor:gate
                                    secondaryTensor:[g sigmoidWithTensor:gate name:nil] name:nil];
        MPSGraphTensor* ffn = [g multiplicationWithPrimaryTensor:silu secondaryTensor:up name:nil];
        MPSGraphTensor* down = [g matrixMultiplicationWithPrimaryTensor:ffn
                                   secondaryTensor:[g transposeTensor:wd dimension:0 withDimension:1 name:nil] name:nil];
        hidden = [g additionWithPrimaryTensor:hidden secondaryTensor:down name:nil];
    }

    // Final norm + LM head (tied weights) + loss
    MPSGraphTensor* finalNormed = buildRMSNorm(g, hidden, cFinalNormW, dim);
    MPSGraphTensor* logits = [g matrixMultiplicationWithPrimaryTensor:finalNormed
                                 secondaryTensor:[g transposeTensor:cEmbedW dimension:0 withDimension:1 name:nil] name:nil];
    MPSGraphTensor* logitsSliced = [g sliceTensor:logits dimension:0 start:0 length:n-1 name:nil];

    // Cross-entropy loss
    MPSGraphTensor* oneHot = [g oneHotWithIndicesTensor:tg.targetTokens depth:vocabSize axis:-1
                                               dataType:MPSDataTypeFloat32 onValue:1.0f offValue:0.0f name:nil];
    MPSGraphTensor* ce = [g softMaxCrossEntropyWithSourceTensor:logitsSliced labelsTensor:oneHot
                                                          axis:-1 reductionType:MPSGraphLossReductionTypeMean name:nil];
    tg.loss = [g meanOfTensor:ce axes:@[@0] name:@"loss"];

    // --- Autograd + Optimizer ---
    NSArray<MPSGraphTensor*>* allWeights = [tg.weightPlaceholders copy];

    NSDictionary<MPSGraphTensor*, MPSGraphTensor*>* grads =
        [g gradientForPrimaryTensor:tg.loss withTensors:allWeights name:nil];

    tg.gradientTensors = [[NSMutableArray alloc] init];
    tg.diffableVars = [[NSMutableArray alloc] init];
    for (MPSGraphTensor* wt in allWeights) {
        MPSGraphTensor* grad = grads[wt];
        if (grad) {
            [tg.diffableVars addObject:wt];
            [tg.gradientTensors addObject:grad];
        }
    }

    if (mode == 1 || mode == 2) {
        // Both graph-Adam (mode 1) and graph-accumulate (mode 2) need Adam ops.
        MPSGraphTensor* lrTensor = [g placeholderWithShape:@[@1] dataType:MPSDataTypeFloat32 name:@"lr"];
        tg.lrPlaceholder = lrTensor;
        MPSGraphTensor* beta1T = [g constantWithScalar:0.9 dataType:MPSDataTypeFloat32];
        MPSGraphTensor* beta2T = [g constantWithScalar:0.95 dataType:MPSDataTypeFloat32];
        MPSGraphTensor* epsT = [g constantWithScalar:1e-8 dataType:MPSDataTypeFloat32];

        if (mode == 2) {
            // Graph-accumulate: accumulator variables collect gradients across
            // micro-batches. Adam reads from accumulators (scaled by 1/N), not
            // raw gradients. This gives graph-Adam the batch size it needs.
            tg.accumVars = [[NSMutableArray alloc] init];
            tg.accumOps = [[NSMutableArray alloc] init];
            tg.adamAccumOps = [[NSMutableArray alloc] init];

            // Scale placeholder: fed as 1/accumSteps when running Adam
            MPSGraphTensor* scalePlaceholder = [g placeholderWithShape:@[@1]
                                                              dataType:MPSDataTypeFloat32
                                                                  name:@"accum_scale"];
            tg.accumScalePlaceholder = scalePlaceholder;

            for (MPSGraphTensor* var in tg.diffableVars) {
                MPSGraphTensor* grad = grads[var];
                if (!grad) continue;
                int sz = 1;
                for (NSNumber* s in var.shape) sz *= [s intValue];
                NSMutableData* zeros = [NSMutableData dataWithLength:sz * sizeof(float)];

                // Gradient accumulator variable (zeroed initially)
                MPSGraphTensor* accum = [g variableWithData:zeros shape:var.shape
                                                   dataType:MPSDataTypeFloat32 name:nil];
                [tg.accumVars addObject:accum];

                // Accumulate op: accum += grad (run each micro-batch)
                MPSGraphTensor* accumSum = [g additionWithPrimaryTensor:accum
                                                       secondaryTensor:grad name:nil];
                [tg.accumOps addObject:[g assignVariable:accum
                                        withValueOfTensor:accumSum name:nil]];

                // Adam state
                MPSGraphTensor* mom = [g variableWithData:zeros shape:var.shape
                                                 dataType:MPSDataTypeFloat32 name:nil];
                MPSGraphTensor* vel = [g variableWithData:zeros shape:var.shape
                                                 dataType:MPSDataTypeFloat32 name:nil];

                // Adam op: reads scaled accumulator (accum * scale), updates weight
                MPSGraphTensor* scaledAccum = [g multiplicationWithPrimaryTensor:accum
                                                                secondaryTensor:scalePlaceholder
                                                                           name:nil];
                NSArray* ar = [g adamWithCurrentLearningRateTensor:lrTensor
                                                     beta1Tensor:beta1T beta2Tensor:beta2T
                                                    epsilonTensor:epsT valuesTensor:var
                                                   momentumTensor:mom velocityTensor:vel
                                              maximumVelocityTensor:nil
                                                    gradientTensor:scaledAccum name:nil];
                [tg.adamAccumOps addObject:[g assignVariable:var
                                            withValueOfTensor:ar[0] name:nil]];
                [tg.adamAccumOps addObject:[g assignVariable:mom
                                            withValueOfTensor:ar[1] name:nil]];
                [tg.adamAccumOps addObject:[g assignVariable:vel
                                            withValueOfTensor:ar[2] name:nil]];

                // Zero the accumulator after Adam step
                MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:MPSDataTypeFloat32];
                MPSGraphTensor* zeroFull = [g multiplicationWithPrimaryTensor:accum
                                                             secondaryTensor:zero name:nil];
                [tg.adamAccumOps addObject:[g assignVariable:accum
                                            withValueOfTensor:zeroFull name:nil]];
            }
            tg.updateOps = nil; // mode 2 doesn't use updateOps
        } else {
            // Mode 1: graph-Adam — optimizer runs every step, no accumulation
            NSMutableArray<MPSGraphOperation*>* ops = [[NSMutableArray alloc] init];
            for (MPSGraphTensor* var in tg.diffableVars) {
                MPSGraphTensor* grad = grads[var];
                if (!grad) continue;
                int sz = 1;
                for (NSNumber* s in var.shape) sz *= [s intValue];
                NSMutableData* zeros = [NSMutableData dataWithLength:sz * sizeof(float)];
                MPSGraphTensor* mom = [g variableWithData:zeros shape:var.shape
                                                 dataType:MPSDataTypeFloat32 name:nil];
                MPSGraphTensor* vel = [g variableWithData:zeros shape:var.shape
                                                 dataType:MPSDataTypeFloat32 name:nil];
                NSArray* ar = [g adamWithCurrentLearningRateTensor:lrTensor
                                                     beta1Tensor:beta1T beta2Tensor:beta2T
                                                    epsilonTensor:epsT valuesTensor:var
                                                   momentumTensor:mom velocityTensor:vel
                                              maximumVelocityTensor:nil
                                                    gradientTensor:grad name:nil];
                [ops addObject:[g assignVariable:var withValueOfTensor:ar[0] name:nil]];
                [ops addObject:[g assignVariable:mom withValueOfTensor:ar[1] name:nil]];
                [ops addObject:[g assignVariable:vel withValueOfTensor:ar[2] name:nil]];
            }
            tg.updateOps = ops;
        }
    } else {
        // Split (mode 0): no graph optimizer, gradients output for external AdamW
        tg.updateOps = nil;
        tg.lrPlaceholder = nil;
    }

    // Mode 0 (split) needs gradientTensors for output. Modes 1/2 handle
    // gradients internally (via Adam ops / accumulators), so clear them.
    if (mode != 0) {
        tg.gradientTensors = [[NSMutableArray alloc] init];
    }

    g_train_graph = tg;
    return 0;
}

// Execute training step.
// mode 0 (split): feeds weightBufs, outputs gradients to gradBufs. Caller does AdamW.
// mode 1 (graph-Adam): feeds lr, variables updated internally. weightBufs/gradBufs ignored.

float mtl_graph_train_step(int* tokens, int* targets, int n,
                           void** weightBufs, void** gradBufs, int nWeights,
                           float learningRate, int mode) {
    MongooseTrainGraph* tg = g_train_graph;
    if (!tg) return -1.0f;

    // @autoreleasepool: cgo threads have no Cocoa run loop, so autoreleased ObjC
    // objects (NSData, MPSGraphTensorData, NSDictionary, etc.) created each step
    // would accumulate indefinitely without this. Drains ~10 objects/step.
    @autoreleasepool {

    int seqN = tg.seqLen + 1;

    // Build feeds: tokens + targets + all weight buffers
    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = [[NSMutableDictionary alloc] init];

    // Token inputs (small — NSData copy is fine)
    NSData* tokData = [NSData dataWithBytes:tokens length:n*sizeof(int)];
    feeds[tg.inputTokens] = [[MPSGraphTensorData alloc] initWithDevice:[MPSGraphDevice deviceWithMTLDevice:g_device]
                                                                  data:tokData shape:@[@(n)] dataType:MPSDataTypeInt32];
    NSData* tgtData = [NSData dataWithBytes:targets length:(n-1)*sizeof(int)];
    feeds[tg.targetTokens] = [[MPSGraphTensorData alloc] initWithDevice:[MPSGraphDevice deviceWithMTLDevice:g_device]
                                                                   data:tgtData shape:@[@(n-1)] dataType:MPSDataTypeInt32];

    if (mode == 0) {
        // SPLIT: feed weight buffers, request gradients
        for (int i = 0; i < nWeights && i < (int)tg.weightPlaceholders.count; i++) {
            MPSGraphTensor* ph = tg.weightPlaceholders[i];
            id<MTLBuffer> buf = (__bridge id<MTLBuffer>)weightBufs[i];
            MPSNDArrayDescriptor* desc = [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32 shape:ph.shape];
            MPSNDArray* nd = [[MPSNDArray alloc] initWithBuffer:buf offset:0 descriptor:desc];
            feeds[ph] = [[MPSGraphTensorData alloc] initWithMPSNDArray:nd];
        }
    } else {
        // GRAPH-ADAM: feed learning rate, variables are internal
        NSData* lrData = [NSData dataWithBytes:&learningRate length:sizeof(float)];
        feeds[tg.lrPlaceholder] = [[MPSGraphTensorData alloc] initWithDevice:[MPSGraphDevice deviceWithMTLDevice:g_device]
                                                                        data:lrData shape:@[@1] dataType:MPSDataTypeFloat32];
    }

    // Target tensors
    NSMutableArray<MPSGraphTensor*>* targetTensors = [[NSMutableArray alloc] init];
    [targetTensors addObject:tg.loss];
    if (mode == 0) {
        [targetTensors addObjectsFromArray:tg.gradientTensors];
    }

    // Dispatch graph ASYNC — GPU starts working immediately
    // For graph-Adam: the graph runs fwd+bwd+Adam, we sync after
    // For split: we could pipeline AdamW during next step's fwd+bwd
    MPSGraphExecutionDescriptor* execDesc = [[MPSGraphExecutionDescriptor alloc] init];

    // Select target operations by mode:
    // mode 0 (split): no ops, gradients returned as tensors
    // mode 1 (graph-Adam): updateOps (fwd+bwd+Adam in one dispatch)
    // mode 2 (graph-accumulate): accumOps (fwd+bwd, gradients added to accumulators)
    NSArray<MPSGraphOperation*>* targetOps = nil;
    if (mode == 1) targetOps = tg.updateOps;
    else if (mode == 2) targetOps = tg.accumOps;

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = nil;
    @try {
        results = [tg.graph runAsyncWithMTLCommandQueue:g_queue
                                                 feeds:feeds
                                         targetTensors:targetTensors
                                      targetOperations:targetOps
                                   executionDescriptor:execDesc];
    } @catch (NSException* e) {
        NSLog(@"mongoose: graph failed: %@", e);
        return -2.0f;
    }

    // Sync — wait for GPU graph to complete.
    // The GPU is executing fwd+bwd+Adam right now. The CPU thread is free.
    // On unified memory, the CPU can do useful work during this window:
    // data prefetch, logging, checkpointing, learning rate scheduling.
    // For now we just sync. The async dispatch itself eliminates the
    // MPSGraph internal dispatch overhead that the sync version has.
    id<MTLCommandBuffer> syncCmd = [g_queue commandBuffer];
    [syncCmd commit];
    [syncCmd waitUntilCompleted];

    float lossVal = 0;
    MPSGraphTensorData* lossResult = results[tg.loss];
    if (lossResult) [[lossResult mpsndarray] readBytes:&lossVal strideBytes:nil];

    // Copy gradients for split mode
    if (mode == 0) {
        for (int i = 0; i < (int)tg.gradientTensors.count && i < nWeights; i++) {
            MPSGraphTensorData* gd = results[tg.gradientTensors[i]];
            if (gd && gradBufs[i]) {
                id<MTLBuffer> dst = (__bridge id<MTLBuffer>)gradBufs[i];
                [[gd mpsndarray] readBytes:[dst contents] strideBytes:nil];
            }
        }
    }

    return lossVal;
    } // @autoreleasepool
}

// Graph-accumulate Adam step: runs Adam using accumulated gradients, then zeros
// the accumulators. Called once after N micro-batches of mtl_graph_train_step(mode=2).
// accumScale = 1.0 / N (averages the accumulated gradients).
int mtl_graph_accum_adam_step(float learningRate, float accumScale) {
    MongooseTrainGraph* tg = g_train_graph;
    if (!tg || !tg.adamAccumOps) return -1;

    @autoreleasepool {

    // Feed lr + accumulation scale
    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = [[NSMutableDictionary alloc] init];
    NSData* lrData = [NSData dataWithBytes:&learningRate length:sizeof(float)];
    feeds[tg.lrPlaceholder] = [[MPSGraphTensorData alloc]
        initWithDevice:[MPSGraphDevice deviceWithMTLDevice:g_device]
                  data:lrData shape:@[@1] dataType:MPSDataTypeFloat32];
    NSData* scaleData = [NSData dataWithBytes:&accumScale length:sizeof(float)];
    feeds[tg.accumScalePlaceholder] = [[MPSGraphTensorData alloc]
        initWithDevice:[MPSGraphDevice deviceWithMTLDevice:g_device]
                  data:scaleData shape:@[@1] dataType:MPSDataTypeFloat32];

    // Also need token/target placeholders with dummy data (graph requires all inputs)
    int n = tg.seqLen + 1;
    int* dummyTok = (int*)calloc(n, sizeof(int));
    int* dummyTgt = (int*)calloc(n-1, sizeof(int));
    NSData* tokData = [NSData dataWithBytes:dummyTok length:n*sizeof(int)];
    NSData* tgtData = [NSData dataWithBytes:dummyTgt length:(n-1)*sizeof(int)];
    feeds[tg.inputTokens] = [[MPSGraphTensorData alloc]
        initWithDevice:[MPSGraphDevice deviceWithMTLDevice:g_device]
                  data:tokData shape:@[@(n)] dataType:MPSDataTypeInt32];
    feeds[tg.targetTokens] = [[MPSGraphTensorData alloc]
        initWithDevice:[MPSGraphDevice deviceWithMTLDevice:g_device]
                  data:tgtData shape:@[@(n-1)] dataType:MPSDataTypeInt32];
    free(dummyTok); free(dummyTgt);

    // Run Adam + zero accumulators (no forward pass computed — just the assign ops)
    @try {
        [tg.graph runWithMTLCommandQueue:g_queue
                                  feeds:feeds
                          targetTensors:@[]
                       targetOperations:tg.adamAccumOps];
    } @catch (NSException* e) {
        NSLog(@"mongoose: accum adam step failed: %@", e);
        return -2;
    }

    return 0;
    } // @autoreleasepool
}

// Split-mode step: graph does forward+backward, returns loss + gradients.
// Caller does AdamW on CPU/AMX. Variables updated via assignVariable after.
// gradBufs: array of shared memory pointers, same order as diffableVars.
float mtl_graph_fwdbwd_step(int* tokens, int* targets, int n,
                            void** gradBufs, int nGrads) {
    MongooseTrainGraph* tg = g_train_graph;
    if (!tg) return -1.0f;

    @autoreleasepool {

    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = [[NSMutableDictionary alloc] init];
    NSData* tokData = [NSData dataWithBytes:tokens length:n*sizeof(int)];
    feeds[tg.inputTokens] = [[MPSGraphTensorData alloc] initWithDevice:[MPSGraphDevice deviceWithMTLDevice:g_device]
                                                                  data:tokData shape:@[@(n)] dataType:MPSDataTypeInt32];
    NSData* tgtData = [NSData dataWithBytes:targets length:(n-1)*sizeof(int)];
    feeds[tg.targetTokens] = [[MPSGraphTensorData alloc] initWithDevice:[MPSGraphDevice deviceWithMTLDevice:g_device]
                                                                   data:tgtData shape:@[@(n-1)] dataType:MPSDataTypeInt32];
    // lr still needed as placeholder even though we don't use Adam ops
    float lr = 0;
    NSData* lrData = [NSData dataWithBytes:&lr length:sizeof(float)];
    feeds[tg.lrPlaceholder] = [[MPSGraphTensorData alloc] initWithDevice:[MPSGraphDevice deviceWithMTLDevice:g_device]
                                                                    data:lrData shape:@[@1] dataType:MPSDataTypeFloat32];

    // Target: loss + all gradient tensors. NO assignVariable ops.
    NSMutableArray<MPSGraphTensor*>* targetTensors = [[NSMutableArray alloc] init];
    [targetTensors addObject:tg.loss];
    [targetTensors addObjectsFromArray:tg.gradientTensors];

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = nil;
    @try {
        results = [tg.graph runWithMTLCommandQueue:g_queue feeds:feeds
                              targetTensors:targetTensors targetOperations:nil];
    } @catch (NSException* e) {
        NSLog(@"mongoose: graph fwdbwd failed: %@", e);
        return -2.0f;
    }

    float lossVal = 0;
    MPSGraphTensorData* lossResult = results[tg.loss];
    if (lossResult) [[lossResult mpsndarray] readBytes:&lossVal strideBytes:nil];

    // Copy gradients to caller's shared memory buffers
    for (int i = 0; i < nGrads && i < (int)tg.gradientTensors.count; i++) {
        MPSGraphTensorData* gd = results[tg.gradientTensors[i]];
        if (gd && gradBufs[i]) {
            id<MTLBuffer> dst = (__bridge id<MTLBuffer>)gradBufs[i];
            [[gd mpsndarray] readBytes:[dst contents] strideBytes:nil];
        }
    }

    return lossVal;
    } // @autoreleasepool
}

// Apply weight updates from AMX AdamW back to graph variables.
// Called after AMX finishes AdamW. Updates the variable for next fwd+bwd.
int mtl_graph_apply_weights(int varIdx, const float* data, int nFloats) {
    MongooseTrainGraph* tg = g_train_graph;
    if (!tg || varIdx >= (int)tg.diffableVars.count) return -1;
    MPSGraphTensor* var = tg.diffableVars[varIdx];
    NSData* nsData = [NSData dataWithBytes:data length:nFloats * sizeof(float)];
    MPSGraphTensor* newVal = [tg.graph constantWithData:nsData shape:var.shape dataType:MPSDataTypeFloat32];
    MPSGraphOperation* assign = [tg.graph assignVariable:var withValueOfTensor:newVal name:nil];
    [tg.graph runWithMTLCommandQueue:g_queue feeds:@{} targetTensors:@[] targetOperations:@[assign]];
    return 0;
}

int mtl_graph_num_diffable(void) {
    return g_train_graph ? (int)g_train_graph.diffableVars.count : 0;
}

int mtl_graph_full_built(void) { return g_train_graph ? 1 : 0; }
int mtl_graph_num_weights(void) { return g_train_graph ? (int)g_train_graph.weightPlaceholders.count : 0; }

// Initialize a variable's data from external buffer.
// Call after build, before first training step.
int mtl_graph_set_variable(int varIdx, const float* data, int nFloats) {
    MongooseTrainGraph* tg = g_train_graph;
    if (!tg || varIdx >= (int)tg.weightPlaceholders.count) return -1;

    MPSGraphTensor* var = tg.weightPlaceholders[varIdx];
    NSData* nsData = [NSData dataWithBytes:data length:nFloats * sizeof(float)];

    // Create a one-shot graph to assign the initial value
    MPSGraph* initGraph = [[MPSGraph alloc] init];
    MPSGraphTensor* initVal = [initGraph constantWithData:nsData shape:var.shape
                                                 dataType:MPSDataTypeFloat32];
    // This doesn't work — can't assign across graphs. Need to re-init the variable.
    // Instead: read the variable, write the data, let the next run pick it up.
    // Variables with variableWithData store their initial data — but once created, how to update?

    // The correct approach: use runWithMTLCommandQueue with an assign op.
    MPSGraphTensor* newVal = [tg.graph constantWithData:nsData shape:var.shape dataType:MPSDataTypeFloat32];
    MPSGraphOperation* assign = [tg.graph assignVariable:var withValueOfTensor:newVal name:nil];
    [tg.graph runWithMTLCommandQueue:g_queue feeds:@{}
                       targetTensors:@[] targetOperations:@[assign]];
    return 0;
}

// Read a variable's current value back to CPU after training.
// Request the variable as a target tensor — MPSGraph returns its current value.
int mtl_graph_read_variable(int varIdx, float* dst, int nFloats) {
    MongooseTrainGraph* tg = g_train_graph;
    if (!tg || varIdx >= (int)tg.weightPlaceholders.count) return -1;

    MPSGraphTensor* var = tg.weightPlaceholders[varIdx];

    // Build minimal feeds — graph-Adam needs lr placeholder
    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = [[NSMutableDictionary alloc] init];
    if (tg.lrPlaceholder) {
        float lr = 0;
        NSData* lrData = [NSData dataWithBytes:&lr length:sizeof(float)];
        feeds[tg.lrPlaceholder] = [[MPSGraphTensorData alloc]
            initWithDevice:[MPSGraphDevice deviceWithMTLDevice:g_device]
                      data:lrData shape:@[@1] dataType:MPSDataTypeFloat32];
    }
    // Also need token/target placeholders with dummy data
    int n = tg.seqLen + 1;
    int* dummyTok = (int*)calloc(n, sizeof(int));
    int* dummyTgt = (int*)calloc(n-1, sizeof(int));
    NSData* tokData = [NSData dataWithBytes:dummyTok length:n*sizeof(int)];
    NSData* tgtData = [NSData dataWithBytes:dummyTgt length:(n-1)*sizeof(int)];
    feeds[tg.inputTokens] = [[MPSGraphTensorData alloc]
        initWithDevice:[MPSGraphDevice deviceWithMTLDevice:g_device]
                  data:tokData shape:@[@(n)] dataType:MPSDataTypeInt32];
    feeds[tg.targetTokens] = [[MPSGraphTensorData alloc]
        initWithDevice:[MPSGraphDevice deviceWithMTLDevice:g_device]
                  data:tgtData shape:@[@(n-1)] dataType:MPSDataTypeInt32];
    free(dummyTok); free(dummyTgt);

    // Run graph requesting the variable as output — no update ops
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
        [tg.graph runWithMTLCommandQueue:g_queue feeds:feeds
                          targetTensors:@[var] targetOperations:nil];

    MPSGraphTensorData* result = results[var];
    if (!result) return -2;
    [[result mpsndarray] readBytes:dst strideBytes:nil];
    return 0;
}
