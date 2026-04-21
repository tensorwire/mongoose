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

// ============================================================
// Inference Graph — forward-only, single dispatch per token
// ============================================================
//
// Architecture: one MPSGraph encodes the full forward pass.
// Weights stored as graph variables (loaded once, persist on GPU).
// Input: single token ID. Output: logits [vocabSize].
// KV cache: CPU-side (small per-token vectors, attention done on CPU).
//
// This matches the training forward pass (lines 184-289) exactly,
// but for n=1 with no loss/backward/optimizer.

@interface MongooseInferGraph : NSObject
@property MPSGraph* graph;
@property MPSGraphTensor* inputToken;   // [1] int32
@property MPSGraphTensor* logits;       // [1, vocabSize]
@property NSMutableArray<MPSGraphTensor*>* weightVars;
@property int dim, kvDim, headDim, nHeads, nKVHeads, ffnDim, vocabSize, nLayers;

// Intermediate outputs for KV cache (per-layer Q, K, V after RoPE — downloaded for CPU attention)
@property NSMutableArray<MPSGraphTensor*>* layerQ;  // [1, dim]
@property NSMutableArray<MPSGraphTensor*>* layerK;  // [1, kvDim]
@property NSMutableArray<MPSGraphTensor*>* layerV;  // [1, kvDim]

// Post-attention input: normed hidden + attnOut uploaded from CPU
@property MPSGraphTensor* attnOutPlaceholder;       // [1, dim] — CPU attention result fed back

// Per-layer: we split the graph into pre-attention and post-attention subgraphs.
// Pre-attention: embed → RMSNorm → QKV + bias → RoPE → output Q,K,V
// CPU: attention with KV cache
// Post-attention: residual + RMSNorm → FFN → residual → next layer
//
// Actually, we can't split a single MPSGraph mid-execution.
// Instead: build TWO graphs per layer approach is too complex.
//
// Better approach: one graph for the full matmul-heavy path.
// CPU handles attention (small, memory-bound) between graph dispatches.
// Per layer: dispatch graph for QKV matmuls, read Q/K/V, do attention on CPU,
// dispatch graph for O-proj + FFN matmuls.
//
// Even better: build the ENTIRE forward including attention as one graph.
// Single-token attention (pos+1 entries) is just dot products — we can
// pass the KV cache as a placeholder and let MPSGraph do the matmul.
// But the KV cache grows each token... variable-length input is tricky.
//
// Pragmatic approach: TWO graph dispatches per layer.
//   Graph A: RMSNorm → QKV + bias → RoPE (outputs Q, K, V)
//   CPU: KV cache update + attention (small, fast for single token)
//   Graph B: O-proj → residual → RMSNorm → FFN → residual
// Plus one dispatch at the end for final norm + LM head.
// Total: 2*nLayers + 1 dispatches per token.
// For TinyLlama (22 layers): 45 dispatches vs 154 individual matmuls.
// Each dispatch does 3-4 fused matmuls — 3-4x fewer GPU round-trips.
@end

@implementation MongooseInferGraph
@end

static MongooseInferGraph* g_infer_graph = nil;

// Per-layer subgraphs for inference
typedef struct {
    // Graph A: norm → QKV
    MPSGraph* graphA;
    MPSGraphTensor* inputA;     // [1, dim] hidden state
    MPSGraphTensor* outQ;       // [1, dim]
    MPSGraphTensor* outK;       // [1, kvDim]
    MPSGraphTensor* outV;       // [1, kvDim]
    // Weight variables inside graphA
    MPSGraphTensor* norm1W;
    MPSGraphTensor* wq;
    MPSGraphTensor* wk;
    MPSGraphTensor* wv;
    MPSGraphTensor* bq;
    MPSGraphTensor* bk;
    MPSGraphTensor* bv;
    MPSGraphTensor* cosP;
    MPSGraphTensor* sinP;

    // Graph B: O-proj + FFN
    MPSGraph* graphB;
    MPSGraphTensor* inputB_hidden;  // [1, dim] hidden state (before residual)
    MPSGraphTensor* inputB_attn;    // [1, dim] attention output
    MPSGraphTensor* outB;           // [1, dim] new hidden state
    // Weight variables inside graphB
    MPSGraphTensor* wo;
    MPSGraphTensor* norm2W;
    MPSGraphTensor* wg;
    MPSGraphTensor* wu;
    MPSGraphTensor* wd;
} infer_layer_t;

static infer_layer_t* g_infer_layers = NULL;

// Final graph: norm + LM head
static MPSGraph* g_infer_final_graph = NULL;
static MPSGraphTensor* g_infer_final_input = NULL;   // [1, dim]
static MPSGraphTensor* g_infer_final_logits = NULL;   // [1, vocabSize]
static MPSGraphTensor* g_infer_final_normW = NULL;
static MPSGraphTensor* g_infer_final_embedW = NULL;

// Build a single-token RMSNorm (reuse the training helper but for n=1)
static MPSGraphTensor* buildRMSNorm1(MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* weight, int dim) {
    MPSGraphTensor* xSq = [g multiplicationWithPrimaryTensor:x secondaryTensor:x name:nil];
    MPSGraphTensor* mean = [g meanOfTensor:xSq axes:@[@(-1)] name:nil];
    MPSGraphTensor* eps = [g constantWithScalar:1e-6 dataType:MPSDataTypeFloat32];
    MPSGraphTensor* variance = [g additionWithPrimaryTensor:mean secondaryTensor:eps name:nil];
    MPSGraphTensor* rms = [g squareRootWithTensor:variance name:nil];
    MPSGraphTensor* xw = [g multiplicationWithPrimaryTensor:x secondaryTensor:weight name:nil];
    return [g divisionWithPrimaryTensor:xw secondaryTensor:rms name:nil];
}

// Build single-token RoPE (rotate_half convention: pairs first/second half of each head)
static MPSGraphTensor* buildRoPE1(MPSGraph* g, MPSGraphTensor* x,
                                   MPSGraphTensor* cosSlice, MPSGraphTensor* sinSlice,
                                   int nHeads, int headDim) {
    int halfDim = headDim / 2;
    int totalDim = nHeads * headDim;
    MPSGraphTensor* xr = [g reshapeTensor:x withShape:@[@1, @(nHeads), @(headDim)] name:nil];
    MPSGraphTensor* xFirst = [g sliceTensor:xr dimension:2 start:0 length:halfDim name:nil];
    MPSGraphTensor* xSecond = [g sliceTensor:xr dimension:2 start:halfDim length:halfDim name:nil];
    MPSGraphTensor* outFirst = [g subtractionWithPrimaryTensor:[g multiplicationWithPrimaryTensor:xFirst secondaryTensor:cosSlice name:nil]
                                               secondaryTensor:[g multiplicationWithPrimaryTensor:xSecond secondaryTensor:sinSlice name:nil] name:nil];
    MPSGraphTensor* outSecond = [g additionWithPrimaryTensor:[g multiplicationWithPrimaryTensor:xFirst secondaryTensor:sinSlice name:nil]
                                             secondaryTensor:[g multiplicationWithPrimaryTensor:xSecond secondaryTensor:cosSlice name:nil] name:nil];
    MPSGraphTensor* result = [g concatTensors:@[outFirst, outSecond] dimension:2 name:nil];
    return [g reshapeTensor:result withShape:@[@1, @(totalDim)] name:nil];
}

int mtl_infer_build(int dim, int kvDim, int headDim,
                    int nHeads, int nKVHeads, int ffnDim,
                    int vocabSize, int nLayers, float ropeTheta) {
    if (g_infer_graph) return 0;

    MongooseInferGraph* ig = [[MongooseInferGraph alloc] init];
    ig.dim = dim; ig.kvDim = kvDim; ig.headDim = headDim;
    ig.nHeads = nHeads; ig.nKVHeads = nKVHeads; ig.ffnDim = ffnDim;
    ig.vocabSize = vocabSize; ig.nLayers = nLayers;
    ig.weightVars = [[NSMutableArray alloc] init];

    int halfDim = headDim / 2;

    g_infer_layers = (infer_layer_t*)calloc(nLayers, sizeof(infer_layer_t));

    for (int l = 0; l < nLayers; l++) {
        infer_layer_t* il = &g_infer_layers[l];

        // === Graph A: RMSNorm → QKV + bias + RoPE ===
        il->graphA = [[MPSGraph alloc] init];
        MPSGraph* gA = il->graphA;

        il->inputA = [gA placeholderWithShape:@[@1, @(dim)] dataType:MPSDataTypeFloat32 name:@"hidden"];

        // RoPE cos/sin as placeholders (different each position)
        il->cosP = [gA placeholderWithShape:@[@1, @1, @(halfDim)] dataType:MPSDataTypeFloat32
                                       name:@"cos"];
        il->sinP = [gA placeholderWithShape:@[@1, @1, @(halfDim)] dataType:MPSDataTypeFloat32
                                       name:@"sin"];
        MPSGraphTensor* cosP = il->cosP;
        MPSGraphTensor* sinP = il->sinP;

        // Weight variables (persist on GPU)
        NSMutableData* z1 = [NSMutableData dataWithLength:dim * sizeof(float)];
        NSMutableData* zqw = [NSMutableData dataWithLength:dim * dim * sizeof(float)];
        NSMutableData* zkw = [NSMutableData dataWithLength:kvDim * dim * sizeof(float)];

        il->norm1W = [gA variableWithData:z1 shape:@[@1, @(dim)] dataType:MPSDataTypeFloat32 name:nil];
        il->wq = [gA variableWithData:zqw shape:@[@(dim), @(dim)] dataType:MPSDataTypeFloat32 name:nil];
        il->wk = [gA variableWithData:zkw shape:@[@(kvDim), @(dim)] dataType:MPSDataTypeFloat32 name:nil];
        il->wv = [gA variableWithData:zkw shape:@[@(kvDim), @(dim)] dataType:MPSDataTypeFloat32 name:nil];
        il->bq = [gA variableWithData:z1 shape:@[@1, @(dim)] dataType:MPSDataTypeFloat32 name:nil];
        il->bk = [gA variableWithData:[NSMutableData dataWithLength:kvDim * sizeof(float)]
                               shape:@[@1, @(kvDim)] dataType:MPSDataTypeFloat32 name:nil];
        il->bv = [gA variableWithData:[NSMutableData dataWithLength:kvDim * sizeof(float)]
                               shape:@[@1, @(kvDim)] dataType:MPSDataTypeFloat32 name:nil];

        [ig.weightVars addObjectsFromArray:@[il->norm1W, il->wq, il->wk, il->wv, il->bq, il->bk, il->bv]];

        MPSGraphTensor* normed = buildRMSNorm1(gA, il->inputA, il->norm1W, dim);

        MPSGraphTensor* Q = [gA additionWithPrimaryTensor:
                             [gA matrixMultiplicationWithPrimaryTensor:normed
                                secondaryTensor:[gA transposeTensor:il->wq dimension:0 withDimension:1 name:nil] name:nil]
                                              secondaryTensor:il->bq name:nil];
        MPSGraphTensor* K = [gA additionWithPrimaryTensor:
                             [gA matrixMultiplicationWithPrimaryTensor:normed
                                secondaryTensor:[gA transposeTensor:il->wk dimension:0 withDimension:1 name:nil] name:nil]
                                              secondaryTensor:il->bk name:nil];
        MPSGraphTensor* V = [gA additionWithPrimaryTensor:
                             [gA matrixMultiplicationWithPrimaryTensor:normed
                                secondaryTensor:[gA transposeTensor:il->wv dimension:0 withDimension:1 name:nil] name:nil]
                                              secondaryTensor:il->bv name:nil];

        Q = buildRoPE1(gA, Q, cosP, sinP, nHeads, headDim);
        K = buildRoPE1(gA, K, cosP, sinP, nKVHeads, headDim);

        il->outQ = Q;
        il->outK = K;
        il->outV = V;

        // === Graph B: O-proj → residual → RMSNorm → FFN → residual ===
        il->graphB = [[MPSGraph alloc] init];
        MPSGraph* gB = il->graphB;

        il->inputB_hidden = [gB placeholderWithShape:@[@1, @(dim)] dataType:MPSDataTypeFloat32 name:@"hidden"];
        il->inputB_attn = [gB placeholderWithShape:@[@1, @(dim)] dataType:MPSDataTypeFloat32 name:@"attn"];

        NSMutableData* zow = [NSMutableData dataWithLength:dim * dim * sizeof(float)];
        NSMutableData* zgw = [NSMutableData dataWithLength:ffnDim * dim * sizeof(float)];
        NSMutableData* zdw = [NSMutableData dataWithLength:dim * ffnDim * sizeof(float)];

        il->wo = [gB variableWithData:zow shape:@[@(dim), @(dim)] dataType:MPSDataTypeFloat32 name:nil];
        il->norm2W = [gB variableWithData:z1 shape:@[@1, @(dim)] dataType:MPSDataTypeFloat32 name:nil];
        il->wg = [gB variableWithData:zgw shape:@[@(ffnDim), @(dim)] dataType:MPSDataTypeFloat32 name:nil];
        il->wu = [gB variableWithData:zgw shape:@[@(ffnDim), @(dim)] dataType:MPSDataTypeFloat32 name:nil];
        il->wd = [gB variableWithData:zdw shape:@[@(dim), @(ffnDim)] dataType:MPSDataTypeFloat32 name:nil];

        [ig.weightVars addObjectsFromArray:@[il->wo, il->norm2W, il->wg, il->wu, il->wd]];

        // O-proj + residual
        MPSGraphTensor* proj = [gB matrixMultiplicationWithPrimaryTensor:il->inputB_attn
                                   secondaryTensor:[gB transposeTensor:il->wo dimension:0 withDimension:1 name:nil] name:nil];
        MPSGraphTensor* h2 = [gB additionWithPrimaryTensor:il->inputB_hidden secondaryTensor:proj name:nil];

        // FFN
        MPSGraphTensor* normed2 = buildRMSNorm1(gB, h2, il->norm2W, dim);
        MPSGraphTensor* gate = [gB matrixMultiplicationWithPrimaryTensor:normed2
                                   secondaryTensor:[gB transposeTensor:il->wg dimension:0 withDimension:1 name:nil] name:nil];
        MPSGraphTensor* up = [gB matrixMultiplicationWithPrimaryTensor:normed2
                                 secondaryTensor:[gB transposeTensor:il->wu dimension:0 withDimension:1 name:nil] name:nil];
        MPSGraphTensor* siluG = [gB multiplicationWithPrimaryTensor:gate
                                    secondaryTensor:[gB sigmoidWithTensor:gate name:nil] name:nil];
        MPSGraphTensor* ffn = [gB multiplicationWithPrimaryTensor:siluG secondaryTensor:up name:nil];
        MPSGraphTensor* down = [gB matrixMultiplicationWithPrimaryTensor:ffn
                                   secondaryTensor:[gB transposeTensor:il->wd dimension:0 withDimension:1 name:nil] name:nil];
        il->outB = [gB additionWithPrimaryTensor:h2 secondaryTensor:down name:nil];
    }

    // === Final graph: RMSNorm + LM head ===
    g_infer_final_graph = [[MPSGraph alloc] init];
    MPSGraph* gF = g_infer_final_graph;
    g_infer_final_input = [gF placeholderWithShape:@[@1, @(dim)] dataType:MPSDataTypeFloat32 name:@"hidden"];
    g_infer_final_normW = [gF variableWithData:[NSMutableData dataWithLength:dim * sizeof(float)]
                                         shape:@[@1, @(dim)] dataType:MPSDataTypeFloat32 name:nil];
    g_infer_final_embedW = [gF variableWithData:[NSMutableData dataWithLength:vocabSize * dim * sizeof(float)]
                                          shape:@[@(vocabSize), @(dim)] dataType:MPSDataTypeFloat32 name:nil];
    [ig.weightVars addObject:g_infer_final_normW];
    [ig.weightVars addObject:g_infer_final_embedW];

    MPSGraphTensor* finalNormed = buildRMSNorm1(gF, g_infer_final_input, g_infer_final_normW, dim);
    g_infer_final_logits = [gF matrixMultiplicationWithPrimaryTensor:finalNormed
                               secondaryTensor:[gF transposeTensor:g_infer_final_embedW
                                                        dimension:0 withDimension:1 name:nil] name:nil];

    g_infer_graph = ig;
    return 0;
}

int mtl_infer_num_weights(void) {
    return g_infer_graph ? (int)g_infer_graph.weightVars.count : 0;
}

int mtl_infer_set_weight(int idx, const float* data, int nFloats) {
    if (!g_infer_graph || idx >= (int)g_infer_graph.weightVars.count) return -1;
    MPSGraphTensor* var = g_infer_graph.weightVars[idx];

    // Find which graph owns this variable
    MPSGraph* ownerGraph = nil;
    int nLayers = g_infer_graph.nLayers;
    int perLayer = 12; // 7 in graphA + 5 in graphB
    if (idx < nLayers * perLayer) {
        int layer = idx / perLayer;
        int within = idx % perLayer;
        if (within < 7) {
            ownerGraph = g_infer_layers[layer].graphA;
        } else {
            ownerGraph = g_infer_layers[layer].graphB;
        }
    } else {
        ownerGraph = g_infer_final_graph;
    }
    if (!ownerGraph) return -2;

    NSData* nsData = [NSData dataWithBytes:data length:nFloats * sizeof(float)];
    MPSGraphTensor* newVal = [ownerGraph constantWithData:nsData shape:var.shape dataType:MPSDataTypeFloat32];
    MPSGraphOperation* assign = [ownerGraph assignVariable:var withValueOfTensor:newVal name:nil];
    [ownerGraph runWithMTLCommandQueue:g_queue feeds:@{} targetTensors:@[] targetOperations:@[assign]];
    return 0;
}

// Run one inference step: returns logits for a single token at given position.
// cosSlice/sinSlice: precomputed RoPE values for this position [halfDim floats each].
// kvCallback: after graphA runs, Q/K/V are downloaded. Caller does attention on CPU,
// writes attnOut. Then graphB runs.
int mtl_infer_step(int tokenID, float* cosData, float* sinData,
                   float* qOut, float* kOut, float* vOut,
                   float* attnIn, float* logitsOut) {
    MongooseInferGraph* ig = g_infer_graph;
    if (!ig) return -1;

    int dim = ig.dim;
    int kvDim = ig.kvDim;
    int headDim = ig.headDim;
    int halfDim = headDim / 2;
    int vocabSize = ig.vocabSize;
    int nLayers = ig.nLayers;

    // Embed: CPU lookup (tiny — one row of embed matrix)
    // We need the embed weight on CPU too. Read from the final graph's variable.
    // For now, caller passes hidden state after embedding.
    // Actually: let's take tokenID, look up in embed weight variable.
    // Read embed row from GPU:
    float* hidden = (float*)malloc(dim * sizeof(float));

    // Read embed weight for this token from the final graph's embed variable
    {
        MPSGraphTensor* embedW = g_infer_final_embedW;
        // Slice row tokenID from embed [vocabSize, dim]
        // Run a tiny graph to extract one row
        MPSGraph* embedGraph = [[MPSGraph alloc] init];
        MPSGraphTensor* idx = [embedGraph constantWithScalar:tokenID dataType:MPSDataTypeInt32];
        // We can't index into a variable from another graph.
        // Simpler: caller passes the embedded hidden state.
        // Let's restructure: caller embeds on CPU (it already has embedData on CPU).
    }
    free(hidden);

    // Restructured API: caller passes hidden=[1,dim] after CPU embedding.
    // This function handles layers + final norm + LM head.
    return -99; // placeholder — see mtl_infer_forward below
}

// Simpler API: caller handles embedding + attention on CPU.
// This function runs the GPU-heavy parts: QKV matmuls, O-proj, FFN.
//
// Per layer:
//   1. GPU dispatch graphA(hidden) → Q, K, V  [3 fused matmuls + RMSNorm + RoPE]
//   2. CPU: KV cache update + attention → attnOut
//   3. GPU dispatch graphB(hidden, attnOut) → new hidden [4 fused matmuls + RMSNorm + SiLU]
//
// Final: GPU dispatch finalGraph(hidden) → logits [1 fused matmul + RMSNorm]
int mtl_infer_forward(float* hiddenIO, float* cosData, float* sinData,
                      float* qOut, float* kOut, float* vOut,
                      float* attnIn,
                      float* logitsOut,
                      int layer) {
    MongooseInferGraph* ig = g_infer_graph;
    if (!ig) return -1;

    int dim = ig.dim;
    int kvDim = ig.kvDim;
    int halfDim = ig.headDim / 2;

    @autoreleasepool {

    if (layer < ig.nLayers) {
        infer_layer_t* il = &g_infer_layers[layer];

        // === Graph A: norm → QKV + RoPE ===
        NSData* hiddenData = [NSData dataWithBytes:hiddenIO length:dim * sizeof(float)];
        NSData* cosNS = [NSData dataWithBytes:cosData length:halfDim * sizeof(float)];
        NSData* sinNS = [NSData dataWithBytes:sinData length:halfDim * sizeof(float)];

        MPSGraphTensorData* hiddenTD = [[MPSGraphTensorData alloc]
            initWithDevice:[MPSGraphDevice deviceWithMTLDevice:g_device]
                      data:hiddenData shape:@[@1, @(dim)] dataType:MPSDataTypeFloat32];
        MPSGraphTensorData* cosTD = [[MPSGraphTensorData alloc]
            initWithDevice:[MPSGraphDevice deviceWithMTLDevice:g_device]
                      data:cosNS shape:@[@1, @1, @(halfDim)] dataType:MPSDataTypeFloat32];
        MPSGraphTensorData* sinTD = [[MPSGraphTensorData alloc]
            initWithDevice:[MPSGraphDevice deviceWithMTLDevice:g_device]
                      data:sinNS shape:@[@1, @1, @(halfDim)] dataType:MPSDataTypeFloat32];

        NSDictionary* feedsA = @{
            il->inputA: hiddenTD,
            il->cosP: cosTD,
            il->sinP: sinTD
        };

        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* resultsA =
            [il->graphA runWithMTLCommandQueue:g_queue feeds:feedsA
                                targetTensors:@[il->outQ, il->outK, il->outV]
                             targetOperations:nil];

        [[resultsA[il->outQ] mpsndarray] readBytes:qOut strideBytes:nil];
        [[resultsA[il->outK] mpsndarray] readBytes:kOut strideBytes:nil];
        [[resultsA[il->outV] mpsndarray] readBytes:vOut strideBytes:nil];

        // Caller does attention on CPU, writes attnIn

    } else {
        // Final: norm + LM head
        NSData* hiddenData = [NSData dataWithBytes:hiddenIO length:dim * sizeof(float)];
        MPSGraphTensorData* hiddenTD = [[MPSGraphTensorData alloc]
            initWithDevice:[MPSGraphDevice deviceWithMTLDevice:g_device]
                      data:hiddenData shape:@[@1, @(dim)] dataType:MPSDataTypeFloat32];

        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
            [g_infer_final_graph runWithMTLCommandQueue:g_queue feeds:@{g_infer_final_input: hiddenTD}
                                         targetTensors:@[g_infer_final_logits]
                                      targetOperations:nil];

        int vocabSize = ig.vocabSize;
        [[results[g_infer_final_logits] mpsndarray] readBytes:logitsOut strideBytes:nil];
        return 0;
    }

    } // @autoreleasepool
    return 0;
}

// Run graph B (O-proj + FFN) for a layer after CPU attention.
int mtl_infer_forward_b(float* hiddenIO, float* attnOut, int layer) {
    MongooseInferGraph* ig = g_infer_graph;
    if (!ig || layer >= ig.nLayers) return -1;

    int dim = ig.dim;
    infer_layer_t* il = &g_infer_layers[layer];

    @autoreleasepool {
    NSData* hiddenData = [NSData dataWithBytes:hiddenIO length:dim * sizeof(float)];
    NSData* attnData = [NSData dataWithBytes:attnOut length:dim * sizeof(float)];

    MPSGraphTensorData* hiddenTD = [[MPSGraphTensorData alloc]
        initWithDevice:[MPSGraphDevice deviceWithMTLDevice:g_device]
                  data:hiddenData shape:@[@1, @(dim)] dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* attnTD = [[MPSGraphTensorData alloc]
        initWithDevice:[MPSGraphDevice deviceWithMTLDevice:g_device]
                  data:attnData shape:@[@1, @(dim)] dataType:MPSDataTypeFloat32];

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
        [il->graphB runWithMTLCommandQueue:g_queue
                                    feeds:@{il->inputB_hidden: hiddenTD, il->inputB_attn: attnTD}
                            targetTensors:@[il->outB]
                         targetOperations:nil];

    [[results[il->outB] mpsndarray] readBytes:hiddenIO strideBytes:nil];

    } // @autoreleasepool
    return 0;
}

// ============================================================
// Fused single-dispatch inference graph
// ============================================================
// One MPSGraph encodes the entire transformer forward pass.
// KV cache lives as GPU variables; attention is computed on-GPU.
// Single graph.run() per token.

typedef struct {
    MPSGraphTensor* kCache;  // variable [maxSeq, kvDim]
    MPSGraphTensor* vCache;  // variable [maxSeq, kvDim]
} fused_kv_t;

static MPSGraph* g_fused_graph = NULL;
static MPSGraphTensor* g_fused_hidden_in = NULL;   // placeholder [1, dim]
static MPSGraphTensor* g_fused_cos = NULL;          // placeholder [1, 1, halfDim]
static MPSGraphTensor* g_fused_sin = NULL;          // placeholder [1, 1, halfDim]
static MPSGraphTensor* g_fused_pos = NULL;          // placeholder [] int32
static MPSGraphTensor* g_fused_logits = NULL;       // output [1, vocabSize]
static NSMutableArray<MPSGraphTensor*>* g_fused_weight_vars = nil;
static fused_kv_t* g_fused_kv = NULL;
static NSMutableArray<MPSGraphOperation*>* g_fused_kv_assigns = nil;
static int g_fused_dim = 0;
static int g_fused_halfDim = 0;
static int g_fused_vocabSize = 0;
static int g_fused_maxSeq = 0;

int mtl_fused_infer_build(int dim, int kvDim, int headDim,
                          int nHeads, int nKVHeads, int ffnDim,
                          int vocabSize, int nLayers, int maxSeq,
                          float ropeTheta) {
    if (g_fused_graph) return 0;

    int halfDim = headDim / 2;
    int kvMul = nHeads / nKVHeads;
    float scale = 1.0f / sqrtf((float)headDim);

    g_fused_dim = dim;
    g_fused_halfDim = halfDim;
    g_fused_vocabSize = vocabSize;
    g_fused_maxSeq = maxSeq;

    MPSGraph* g = [[MPSGraph alloc] init];
    g_fused_weight_vars = [[NSMutableArray alloc] init];
    g_fused_kv_assigns = [[NSMutableArray alloc] init];
    g_fused_kv = (fused_kv_t*)calloc(nLayers, sizeof(fused_kv_t));

    // Placeholders
    g_fused_hidden_in = [g placeholderWithShape:@[@1, @(dim)] dataType:MPSDataTypeFloat32 name:@"hidden"];
    g_fused_cos = [g placeholderWithShape:@[@1, @1, @(halfDim)] dataType:MPSDataTypeFloat32 name:@"cos"];
    g_fused_sin = [g placeholderWithShape:@[@1, @1, @(halfDim)] dataType:MPSDataTypeFloat32 name:@"sin"];
    g_fused_pos = [g placeholderWithShape:@[] dataType:MPSDataTypeInt32 name:@"pos"];

    // seqLen = pos + 1 (number of KV entries to attend over)
    MPSGraphTensor* one_i32 = [g constantWithScalar:1 dataType:MPSDataTypeInt32];
    MPSGraphTensor* seqLen = [g additionWithPrimaryTensor:g_fused_pos secondaryTensor:one_i32 name:nil];

    MPSGraphTensor* scaleTensor = [g constantWithScalar:scale dataType:MPSDataTypeFloat32];

    MPSGraphTensor* x = g_fused_hidden_in;

    for (int l = 0; l < nLayers; l++) {
        // Weight variables — same order as multi-dispatch: norm1, wq, wk, wv, bq, bk, bv, wo, norm2, gate, up, down
        NSMutableData* z1 = [NSMutableData dataWithLength:dim * sizeof(float)];
        NSMutableData* zqw = [NSMutableData dataWithLength:dim * dim * sizeof(float)];
        NSMutableData* zkw = [NSMutableData dataWithLength:kvDim * dim * sizeof(float)];
        NSMutableData* zow = [NSMutableData dataWithLength:dim * dim * sizeof(float)];
        NSMutableData* zgw = [NSMutableData dataWithLength:ffnDim * dim * sizeof(float)];
        NSMutableData* zdw = [NSMutableData dataWithLength:dim * ffnDim * sizeof(float)];

        MPSGraphTensor* norm1W = [g variableWithData:z1 shape:@[@1, @(dim)] dataType:MPSDataTypeFloat32 name:nil];
        MPSGraphTensor* wq = [g variableWithData:zqw shape:@[@(dim), @(dim)] dataType:MPSDataTypeFloat32 name:nil];
        MPSGraphTensor* wk = [g variableWithData:zkw shape:@[@(kvDim), @(dim)] dataType:MPSDataTypeFloat32 name:nil];
        MPSGraphTensor* wv = [g variableWithData:zkw shape:@[@(kvDim), @(dim)] dataType:MPSDataTypeFloat32 name:nil];
        MPSGraphTensor* bq = [g variableWithData:z1 shape:@[@1, @(dim)] dataType:MPSDataTypeFloat32 name:nil];
        MPSGraphTensor* bk = [g variableWithData:[NSMutableData dataWithLength:kvDim * sizeof(float)]
                                           shape:@[@1, @(kvDim)] dataType:MPSDataTypeFloat32 name:nil];
        MPSGraphTensor* bv = [g variableWithData:[NSMutableData dataWithLength:kvDim * sizeof(float)]
                                           shape:@[@1, @(kvDim)] dataType:MPSDataTypeFloat32 name:nil];
        MPSGraphTensor* wo = [g variableWithData:zow shape:@[@(dim), @(dim)] dataType:MPSDataTypeFloat32 name:nil];
        MPSGraphTensor* norm2W = [g variableWithData:z1 shape:@[@1, @(dim)] dataType:MPSDataTypeFloat32 name:nil];
        MPSGraphTensor* wg = [g variableWithData:zgw shape:@[@(ffnDim), @(dim)] dataType:MPSDataTypeFloat32 name:nil];
        MPSGraphTensor* wu = [g variableWithData:zgw shape:@[@(ffnDim), @(dim)] dataType:MPSDataTypeFloat32 name:nil];
        MPSGraphTensor* wd = [g variableWithData:zdw shape:@[@(dim), @(ffnDim)] dataType:MPSDataTypeFloat32 name:nil];

        [g_fused_weight_vars addObjectsFromArray:@[norm1W, wq, wk, wv, bq, bk, bv, wo, norm2W, wg, wu, wd]];

        // KV cache variables
        NSMutableData* kvZeros = [NSMutableData dataWithLength:maxSeq * kvDim * sizeof(float)];
        g_fused_kv[l].kCache = [g variableWithData:kvZeros shape:@[@(maxSeq), @(kvDim)] dataType:MPSDataTypeFloat32 name:nil];
        g_fused_kv[l].vCache = [g variableWithData:kvZeros shape:@[@(maxSeq), @(kvDim)] dataType:MPSDataTypeFloat32 name:nil];

        // === RMSNorm → QKV + bias + RoPE ===
        MPSGraphTensor* normed = buildRMSNorm1(g, x, norm1W, dim);

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

        Q = buildRoPE1(g, Q, g_fused_cos, g_fused_sin, nHeads, headDim);
        K = buildRoPE1(g, K, g_fused_cos, g_fused_sin, nKVHeads, headDim);

        // === Write K, V into cache at position pos ===
        // K is [1, kvDim], scatter into kCache[pos, :]
        MPSGraphTensor* posIdx = [g reshapeTensor:g_fused_pos withShape:@[@1, @1] name:nil];
        MPSGraphTensor* kRow = [g reshapeTensor:K withShape:@[@1, @(kvDim)] name:nil];
        MPSGraphTensor* vRow = [g reshapeTensor:V withShape:@[@1, @(kvDim)] name:nil];

        MPSGraphTensor* newK = [g scatterNDWithUpdatesTensor:kRow
                                               indicesTensor:posIdx
                                                       shape:@[@(maxSeq), @(kvDim)]
                                             batchDimensions:0
                                                        mode:MPSGraphScatterModeSet
                                                        name:nil];
        MPSGraphTensor* newV = [g scatterNDWithUpdatesTensor:vRow
                                               indicesTensor:posIdx
                                                       shape:@[@(maxSeq), @(kvDim)]
                                             batchDimensions:0
                                                        mode:MPSGraphScatterModeSet
                                                        name:nil];

        // kCache = kCache + scatter (additive: old values preserved, new row set)
        // Actually scatterND with Set creates a fresh tensor with zeros + the update.
        // We need: kCache_new = kCache_old with row[pos] replaced.
        // Use: kCache_new = kCache_old - old_row_scattered + new_row_scattered
        // Simpler: read old cache, create mask, replace row.
        // Simplest for MPSGraph: use assignVariable to update the cache after run.
        // But that requires multiple dispatches...
        //
        // Alternative: build it as kCache + scatter_add. But we need to zero the old
        // row first...
        //
        // Best approach: use gatherND to read cache rows 0..pos, concat new K/V,
        // do attention on that slice, then use assignVariable to persist.
        //
        // Actually the cleanest: keep kCache as a variable, and use
        // scatterNDWithDataTensor (which updates IN the existing tensor):
        MPSGraphTensor* kUpdated = [g scatterNDWithDataTensor:g_fused_kv[l].kCache
                                                updatesTensor:kRow
                                                indicesTensor:posIdx
                                              batchDimensions:0
                                                         mode:MPSGraphScatterModeSet
                                                         name:nil];
        MPSGraphTensor* vUpdated = [g scatterNDWithDataTensor:g_fused_kv[l].vCache
                                                updatesTensor:vRow
                                                indicesTensor:posIdx
                                              batchDimensions:0
                                                         mode:MPSGraphScatterModeSet
                                                         name:nil];

        // Assign updated cache back to variables
        MPSGraphOperation* kAssign = [g assignVariable:g_fused_kv[l].kCache withValueOfTensor:kUpdated name:nil];
        MPSGraphOperation* vAssign = [g assignVariable:g_fused_kv[l].vCache withValueOfTensor:vUpdated name:nil];
        [g_fused_kv_assigns addObject:kAssign];
        [g_fused_kv_assigns addObject:vAssign];

        // === Attention: Q @ K^T scaled, masked softmax, @ V ===
        // Slice kCache/vCache to [seqLen, kvDim] using dynamic slicing
        // gatherND with indices [0..pos]
        MPSGraphTensor* indices = [g coordinateAlongAxis:0
                                            withShape:@[@(maxSeq), @(kvDim)]
                                                 name:nil]; // [maxSeq, kvDim] where each row = row_index
        // Simpler: just use the full cache and mask out positions > pos
        // Q: [1, nHeads, headDim] -> per head: [1, headDim]
        // K: [maxSeq, nKVHeads, headDim] -> per kvHead: [maxSeq, headDim]
        // scores[h, t] = Q[h] . K[t, h/kvMul] / sqrt(headDim)  for t <= pos

        // Reshape Q to [nHeads, 1, headDim]
        MPSGraphTensor* Qr = [g reshapeTensor:Q withShape:@[@(nHeads), @1, @(headDim)] name:nil];
        // Use kUpdated [maxSeq, kvDim] -> [nKVHeads, maxSeq, headDim]
        MPSGraphTensor* Kr = [g reshapeTensor:kUpdated withShape:@[@(maxSeq), @(nKVHeads), @(headDim)] name:nil];
        Kr = [g transposeTensor:Kr dimension:0 withDimension:1 name:nil]; // [nKVHeads, maxSeq, headDim]
        MPSGraphTensor* Vr = [g reshapeTensor:vUpdated withShape:@[@(maxSeq), @(nKVHeads), @(headDim)] name:nil];
        Vr = [g transposeTensor:Vr dimension:0 withDimension:1 name:nil]; // [nKVHeads, maxSeq, headDim]

        // Expand KV heads for GQA: [nKVHeads, maxSeq, headDim] -> [nHeads, maxSeq, headDim]
        if (kvMul > 1) {
            // Tile each KV head kvMul times along head dimension
            // [nKVHeads, 1, maxSeq, headDim] -> [nKVHeads, kvMul, maxSeq, headDim] -> [nHeads, maxSeq, headDim]
            Kr = [g reshapeTensor:Kr withShape:@[@(nKVHeads), @1, @(maxSeq), @(headDim)] name:nil];
            Kr = [g tileTensor:Kr withMultiplier:@[@1, @(kvMul), @1, @1] name:nil];
            Kr = [g reshapeTensor:Kr withShape:@[@(nHeads), @(maxSeq), @(headDim)] name:nil];
            Vr = [g reshapeTensor:Vr withShape:@[@(nKVHeads), @1, @(maxSeq), @(headDim)] name:nil];
            Vr = [g tileTensor:Vr withMultiplier:@[@1, @(kvMul), @1, @1] name:nil];
            Vr = [g reshapeTensor:Vr withShape:@[@(nHeads), @(maxSeq), @(headDim)] name:nil];
        }

        // scores = Q @ K^T: [nHeads, 1, headDim] @ [nHeads, headDim, maxSeq] = [nHeads, 1, maxSeq]
        MPSGraphTensor* KrT = [g transposeTensor:Kr dimension:1 withDimension:2 name:nil];
        MPSGraphTensor* scores = [g matrixMultiplicationWithPrimaryTensor:Qr secondaryTensor:KrT name:nil];
        scores = [g multiplicationWithPrimaryTensor:scores secondaryTensor:scaleTensor name:nil];

        // Causal mask: positions > pos get -inf
        // Build mask: [1, 1, maxSeq] where mask[t] = 0 if t <= pos, -inf otherwise
        MPSGraphTensor* posRange = [g coordinateAlongAxis:(-1)
                                              withShape:@[@1, @1, @(maxSeq)]
                                                   name:nil]; // [1,1,maxSeq] values 0..maxSeq-1
        MPSGraphTensor* posBC = [g reshapeTensor:g_fused_pos withShape:@[@1, @1, @1] name:nil];
        posBC = [g castTensor:posBC toType:MPSDataTypeInt32 name:@"posBC"];
        MPSGraphTensor* causalMask = [g greaterThanWithPrimaryTensor:posRange
                                                    secondaryTensor:posBC
                                                               name:nil]; // true where t > pos
        MPSGraphTensor* negInf = [g constantWithScalar:-1e9 dataType:MPSDataTypeFloat32];
        MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:MPSDataTypeFloat32];
        MPSGraphTensor* maskVals = [g selectWithPredicateTensor:causalMask
                                             truePredicateTensor:negInf
                                            falsePredicateTensor:zero
                                                            name:nil];
        scores = [g additionWithPrimaryTensor:scores secondaryTensor:maskVals name:nil];

        // Softmax along last dim
        scores = [g softMaxWithTensor:scores axis:(-1) name:nil]; // [nHeads, 1, maxSeq]

        // attnOut = scores @ V: [nHeads, 1, maxSeq] @ [nHeads, maxSeq, headDim] = [nHeads, 1, headDim]
        MPSGraphTensor* attnOut = [g matrixMultiplicationWithPrimaryTensor:scores secondaryTensor:Vr name:nil];
        attnOut = [g reshapeTensor:attnOut withShape:@[@1, @(dim)] name:nil]; // [1, dim]

        // === O-proj + residual ===
        MPSGraphTensor* proj = [g matrixMultiplicationWithPrimaryTensor:attnOut
                                   secondaryTensor:[g transposeTensor:wo dimension:0 withDimension:1 name:nil] name:nil];
        x = [g additionWithPrimaryTensor:x secondaryTensor:proj name:nil];

        // === FFN: RMSNorm → gate/up → SiLU → down → residual ===
        MPSGraphTensor* normed2 = buildRMSNorm1(g, x, norm2W, dim);
        MPSGraphTensor* gate = [g matrixMultiplicationWithPrimaryTensor:normed2
                                   secondaryTensor:[g transposeTensor:wg dimension:0 withDimension:1 name:nil] name:nil];
        MPSGraphTensor* up = [g matrixMultiplicationWithPrimaryTensor:normed2
                                 secondaryTensor:[g transposeTensor:wu dimension:0 withDimension:1 name:nil] name:nil];
        MPSGraphTensor* siluG = [g multiplicationWithPrimaryTensor:gate
                                    secondaryTensor:[g sigmoidWithTensor:gate name:nil] name:nil];
        MPSGraphTensor* ffn = [g multiplicationWithPrimaryTensor:siluG secondaryTensor:up name:nil];
        MPSGraphTensor* down = [g matrixMultiplicationWithPrimaryTensor:ffn
                                   secondaryTensor:[g transposeTensor:wd dimension:0 withDimension:1 name:nil] name:nil];
        x = [g additionWithPrimaryTensor:x secondaryTensor:down name:nil];
    }

    // === Final: RMSNorm + LM head ===
    MPSGraphTensor* finalNormW = [g variableWithData:[NSMutableData dataWithLength:dim * sizeof(float)]
                                               shape:@[@1, @(dim)] dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* lmHeadW = [g variableWithData:[NSMutableData dataWithLength:vocabSize * dim * sizeof(float)]
                                            shape:@[@(vocabSize), @(dim)] dataType:MPSDataTypeFloat32 name:nil];
    [g_fused_weight_vars addObject:finalNormW];
    [g_fused_weight_vars addObject:lmHeadW];

    MPSGraphTensor* finalNormed = buildRMSNorm1(g, x, finalNormW, dim);
    g_fused_logits = [g matrixMultiplicationWithPrimaryTensor:finalNormed
                         secondaryTensor:[g transposeTensor:lmHeadW dimension:0 withDimension:1 name:nil] name:nil];

    g_fused_graph = g;
    return 0;
}

int mtl_fused_infer_num_weights(void) {
    return g_fused_weight_vars ? (int)g_fused_weight_vars.count : 0;
}

int mtl_fused_infer_set_weight(int idx, const float* data, int nFloats) {
    if (!g_fused_graph || idx >= (int)g_fused_weight_vars.count) return -1;
    MPSGraphTensor* var = g_fused_weight_vars[idx];
    NSData* nsData = [NSData dataWithBytes:data length:nFloats * sizeof(float)];
    MPSGraphTensor* newVal = [g_fused_graph constantWithData:nsData shape:var.shape dataType:MPSDataTypeFloat32];
    MPSGraphOperation* assign = [g_fused_graph assignVariable:var withValueOfTensor:newVal name:nil];
    [g_fused_graph runWithMTLCommandQueue:g_queue feeds:@{} targetTensors:@[] targetOperations:@[assign]];
    return 0;
}

int mtl_fused_infer_step(float* hiddenIn, float* cosData, float* sinData,
                         int pos, float* logitsOut) {
    if (!g_fused_graph) return -1;

    int dim = g_fused_dim;
    int halfDim = g_fused_halfDim;
    int vocabSize = g_fused_vocabSize;

    @autoreleasepool {

    NSData* hiddenNS = [NSData dataWithBytes:hiddenIn length:dim * sizeof(float)];
    NSData* cosNS = [NSData dataWithBytes:cosData length:halfDim * sizeof(float)];
    NSData* sinNS = [NSData dataWithBytes:sinData length:halfDim * sizeof(float)];
    int32_t posVal = (int32_t)pos;
    NSData* posNS = [NSData dataWithBytes:&posVal length:sizeof(int32_t)];

    MPSGraphTensorData* hiddenTD = [[MPSGraphTensorData alloc]
        initWithDevice:[MPSGraphDevice deviceWithMTLDevice:g_device]
                  data:hiddenNS shape:@[@1, @(dim)] dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* cosTD = [[MPSGraphTensorData alloc]
        initWithDevice:[MPSGraphDevice deviceWithMTLDevice:g_device]
                  data:cosNS shape:@[@1, @1, @(halfDim)] dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* sinTD = [[MPSGraphTensorData alloc]
        initWithDevice:[MPSGraphDevice deviceWithMTLDevice:g_device]
                  data:sinNS shape:@[@1, @1, @(halfDim)] dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* posTD = [[MPSGraphTensorData alloc]
        initWithDevice:[MPSGraphDevice deviceWithMTLDevice:g_device]
                  data:posNS shape:@[] dataType:MPSDataTypeInt32];

    NSDictionary* feeds = @{
        g_fused_hidden_in: hiddenTD,
        g_fused_cos: cosTD,
        g_fused_sin: sinTD,
        g_fused_pos: posTD
    };

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
        [g_fused_graph runWithMTLCommandQueue:g_queue
                                       feeds:feeds
                               targetTensors:@[g_fused_logits]
                            targetOperations:g_fused_kv_assigns];

    [[results[g_fused_logits] mpsndarray] readBytes:logitsOut strideBytes:nil];

    } // @autoreleasepool
    return 0;
}

int mtl_fused_infer_reset(void) {
    if (!g_fused_graph || !g_fused_kv) return -1;
    // Zero out all KV cache variables
    // We rebuild by assigning zeros
    // Count KV pairs from the assigns array: 2 per layer
    int nLayers = (int)g_fused_kv_assigns.count / 2;
    for (int l = 0; l < nLayers; l++) {
        int maxSeq = g_fused_maxSeq;
        int kvDim = (int)[g_fused_kv[l].kCache shape][1].intValue;
        NSMutableData* zeros = [NSMutableData dataWithLength:maxSeq * kvDim * sizeof(float)];
        MPSGraphTensor* zeroTensor = [g_fused_graph constantWithData:zeros
                                                               shape:g_fused_kv[l].kCache.shape
                                                            dataType:MPSDataTypeFloat32];
        MPSGraphOperation* ka = [g_fused_graph assignVariable:g_fused_kv[l].kCache withValueOfTensor:zeroTensor name:nil];
        MPSGraphOperation* va = [g_fused_graph assignVariable:g_fused_kv[l].vCache withValueOfTensor:zeroTensor name:nil];
        [g_fused_graph runWithMTLCommandQueue:g_queue feeds:@{} targetTensors:@[] targetOperations:@[ka, va]];
    }
    return 0;
}
