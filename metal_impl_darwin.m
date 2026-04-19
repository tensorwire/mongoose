// Metal MPS implementation for mongoose — Objective-C CGo bridge.
// Called from metal.go via CGo. All GPU operations go through MPS (Metal Performance Shaders).
//
// Key optimization: command buffer batching. Multiple matmuls are encoded into a single
// command buffer and submitted together. Call mtl_begin_batch() before a sequence of ops,
// then mtl_end_batch() to commit and wait. Without batching, each op gets its own commit/wait.

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <string.h>

// Global Metal state (non-static — shared with metal_graph.m)
id<MTLDevice> g_device = nil;
id<MTLCommandQueue> g_queue = nil;
static char g_device_name[256] = {0};

// Batched command buffer — when non-nil, ops encode into this instead of creating their own
static id<MTLCommandBuffer> g_batch_cmd = nil;

int mtl_init(void) {
    if (g_device) return 0;

    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) return -1;

    g_queue = [g_device newCommandQueue];
    if (!g_queue) return -2;

    const char* name = [[g_device name] UTF8String];
    strncpy(g_device_name, name, sizeof(g_device_name) - 1);
    return 0;
}

const char* mtl_device_name(void) {
    return g_device_name;
}

uint64_t mtl_recommended_max_working_set_size(void) {
    if (!g_device) return 0;
    return [g_device recommendedMaxWorkingSetSize];
}

// --- Batched command buffer ---

void mtl_begin_batch(void) {
    if (!g_batch_cmd) {
        g_batch_cmd = [g_queue commandBuffer];
    }
}

void mtl_end_batch(void) {
    if (g_batch_cmd) {
        [g_batch_cmd commit];
        [g_batch_cmd waitUntilCompleted];
        g_batch_cmd = nil;
    }
}

// Get command buffer — returns batch cmd if batching, otherwise creates a new one
static id<MTLCommandBuffer> get_cmd(void) {
    if (g_batch_cmd) return g_batch_cmd;
    return [g_queue commandBuffer];
}

// Commit if not batching
static void maybe_commit(id<MTLCommandBuffer> cmd) {
    if (!g_batch_cmd) {
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

// --- GPU memory ---

void* mtl_alloc(size_t bytes) {
    id<MTLBuffer> buf = [g_device newBufferWithLength:bytes
                                             options:MTLResourceStorageModeShared];
    return (__bridge_retained void*)buf;
}

void mtl_free(void* bufRef) {
    if (!bufRef) return;
    id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)bufRef;
    buf = nil;
}

void mtl_upload(void* bufRef, const void* src, size_t bytes) {
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)bufRef;
    memcpy([buf contents], src, bytes);
}

void mtl_download(void* dst, void* bufRef, size_t bytes) {
    // Must ensure GPU is done before reading
    if (g_batch_cmd) {
        [g_batch_cmd commit];
        [g_batch_cmd waitUntilCompleted];
        g_batch_cmd = [g_queue commandBuffer]; // start new batch
    }
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)bufRef;
    memcpy(dst, [buf contents], bytes);
}

void mtl_zero(void* bufRef, size_t bytes) {
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)bufRef;
    memset([buf contents], 0, bytes);
}

// Return the CPU-accessible pointer for a StorageModeShared buffer.
// On Apple Silicon, this IS the GPU memory — unified architecture, zero copy.
void* mtl_shared_ptr(void* bufRef) {
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)bufRef;
    return [buf contents];
}

// --- MatMul via MPS ---

static void encode_gemm(void* aRef, void* bRef, void* cRef,
                         int m, int k, int n,
                         BOOL transA, BOOL transB,
                         int resultRows, int resultCols, int interiorCols,
                         int aRows, int aCols, int bRows, int bCols) {
    id<MTLBuffer> aBuf = (__bridge id<MTLBuffer>)aRef;
    id<MTLBuffer> bBuf = (__bridge id<MTLBuffer>)bRef;
    id<MTLBuffer> cBuf = (__bridge id<MTLBuffer>)cRef;

    MPSMatrixDescriptor *aDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:aRows
                                                                       columns:aCols
                                                                      rowBytes:aCols * sizeof(float)
                                                                      dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *bDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:bRows
                                                                       columns:bCols
                                                                      rowBytes:bCols * sizeof(float)
                                                                      dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *cDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:resultRows
                                                                       columns:resultCols
                                                                      rowBytes:resultCols * sizeof(float)
                                                                      dataType:MPSDataTypeFloat32];

    MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:aBuf descriptor:aDesc];
    MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:bBuf descriptor:bDesc];
    MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:cBuf descriptor:cDesc];

    MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc]
        initWithDevice:g_device
         transposeLeft:transA
        transposeRight:transB
            resultRows:resultRows
         resultColumns:resultCols
       interiorColumns:interiorCols
                 alpha:1.0
                  beta:0.0];

    id<MTLCommandBuffer> cmd = get_cmd();
    [mm encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
    maybe_commit(cmd);
}

// C = A @ B, row-major. A[m,k], B[k,n], C[m,n].
int mtl_sgemm(void* aRef, void* bRef, void* cRef, int m, int k, int n) {
    encode_gemm(aRef, bRef, cRef, m, k, n,
                NO, NO, m, n, k,
                m, k, k, n);
    return 0;
}

// C = A^T @ B. A is [m,k] stored row-major, result C is [k,n].
int mtl_sgemm_transA(void* aRef, void* bRef, void* cRef, int m, int k, int n) {
    encode_gemm(aRef, bRef, cRef, m, k, n,
                YES, NO, k, n, m,
                m, k, m, n);
    return 0;
}

// C = A @ B^T. A is [m,k], B is [n,k] stored row-major, result C is [m,n].
int mtl_sgemm_transB(void* aRef, void* bRef, void* cRef, int m, int k, int n) {
    encode_gemm(aRef, bRef, cRef, m, k, n,
                NO, YES, m, n, k,
                m, k, n, k);
    return 0;
}

// ============================================================
// MPSGraph MatMul — graph-compiled, hardware-tuned GEMM
// This is the EXACT code path PyTorch MPS uses.
// MPSGraph.matrixMultiplicationWithPrimaryTensor lets Apple's
// internal compiler pick optimal tile sizes per GPU architecture.
// ============================================================

// Cached graph per (transA, transB, shape) signature.
// PyTorch does this too — MPSCachedGraph system.
#define GRAPH_CACHE_SIZE 64

typedef struct {
    int m, k, n;
    int transA, transB;
    MPSGraph* graph;
    MPSGraphTensor* inputA;
    MPSGraphTensor* inputB;
    MPSGraphTensor* result;
    MPSGraphExecutable* executable;
} graph_cache_entry_t;

static graph_cache_entry_t g_graph_cache[GRAPH_CACHE_SIZE];
static int g_graph_cache_count = 0;

static graph_cache_entry_t* find_graph(int m, int k, int n, int transA, int transB) {
    for (int i = 0; i < g_graph_cache_count; i++) {
        graph_cache_entry_t* e = &g_graph_cache[i];
        if (e->m == m && e->k == k && e->n == n && e->transA == transA && e->transB == transB)
            return e;
    }
    return NULL;
}

static graph_cache_entry_t* create_graph(int m, int k, int n, int transA, int transB) {
    if (g_graph_cache_count >= GRAPH_CACHE_SIZE) return NULL;

    graph_cache_entry_t* e = &g_graph_cache[g_graph_cache_count++];
    e->m = m; e->k = k; e->n = n;
    e->transA = transA; e->transB = transB;

    // Determine input shapes based on transpose flags
    // For C = A @ B:
    //   No trans:  A[m,k] @ B[k,n] = C[m,n]
    //   TransA:    A[k,m]^T @ B[m,n] = C[k,n] — stored A is [k,m] but logically A^T is [m,k]...
    //   TransB:    A[m,k] @ B[n,k]^T = C[m,n]
    //
    // MPSGraph matmul: result = primary @ secondary
    // With transpose flags on the tensors themselves.

    e->graph = [[MPSGraph alloc] init];

    if (!transA && !transB) {
        // C[m,n] = A[m,k] @ B[k,n]
        e->inputA = [e->graph placeholderWithShape:@[@(m), @(k)] dataType:MPSDataTypeFloat32 name:@"A"];
        e->inputB = [e->graph placeholderWithShape:@[@(k), @(n)] dataType:MPSDataTypeFloat32 name:@"B"];
        e->result = [e->graph matrixMultiplicationWithPrimaryTensor:e->inputA
                                                   secondaryTensor:e->inputB name:@"C"];
    } else if (transA && !transB) {
        // C[k,n] = A[m,k]^T @ B[m,n]
        e->inputA = [e->graph placeholderWithShape:@[@(m), @(k)] dataType:MPSDataTypeFloat32 name:@"A"];
        e->inputB = [e->graph placeholderWithShape:@[@(m), @(n)] dataType:MPSDataTypeFloat32 name:@"B"];
        MPSGraphTensor* aT = [e->graph transposeTensor:e->inputA dimension:0 withDimension:1 name:@"AT"];
        e->result = [e->graph matrixMultiplicationWithPrimaryTensor:aT
                                                   secondaryTensor:e->inputB name:@"C"];
    } else if (!transA && transB) {
        // C[m,n] = A[m,k] @ B[n,k]^T
        e->inputA = [e->graph placeholderWithShape:@[@(m), @(k)] dataType:MPSDataTypeFloat32 name:@"A"];
        e->inputB = [e->graph placeholderWithShape:@[@(n), @(k)] dataType:MPSDataTypeFloat32 name:@"B"];
        MPSGraphTensor* bT = [e->graph transposeTensor:e->inputB dimension:0 withDimension:1 name:@"BT"];
        e->result = [e->graph matrixMultiplicationWithPrimaryTensor:e->inputA
                                                   secondaryTensor:bT name:@"C"];
    } else {
        // TransA + TransB: C[k,n] = A[m,k]^T @ B[n,m]^T — rare, skip for now
        e->inputA = [e->graph placeholderWithShape:@[@(m), @(k)] dataType:MPSDataTypeFloat32 name:@"A"];
        e->inputB = [e->graph placeholderWithShape:@[@(k), @(n)] dataType:MPSDataTypeFloat32 name:@"B"];
        e->result = [e->graph matrixMultiplicationWithPrimaryTensor:e->inputA
                                                   secondaryTensor:e->inputB name:@"C"];
    }

    // Compilation happens lazily on first run — MPSGraph handles caching internally
    e->executable = nil;
    return e;
}

// Run MPSGraph matmul — graph-compiled, cached, hardware-tuned.
// This is the fast path. First call compiles, subsequent calls reuse.
int mtl_graph_sgemm(void* aRef, void* bRef, void* cRef, int m, int k, int n,
                    int transA, int transB) {
    @autoreleasepool {
    graph_cache_entry_t* e = find_graph(m, k, n, transA, transB);
    if (!e) {
        e = create_graph(m, k, n, transA, transB);
        if (!e) return -1;
    }

    id<MTLBuffer> aBuf = (__bridge id<MTLBuffer>)aRef;
    id<MTLBuffer> bBuf = (__bridge id<MTLBuffer>)bRef;
    id<MTLBuffer> cBuf = (__bridge id<MTLBuffer>)cRef;

    // Wrap Metal buffers as MPSGraphTensorData
    NSArray<NSNumber*>* aShape = e->inputA.shape;
    NSArray<NSNumber*>* bShape = e->inputB.shape;

    MPSGraphTensorData* aData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:aBuf shape:aShape dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* bData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:bBuf shape:bShape dataType:MPSDataTypeFloat32];

    // Determine output shape
    NSArray<NSNumber*>* cShape;
    if (!transA && !transB) cShape = @[@(m), @(n)];
    else if (transA) cShape = @[@(k), @(n)];
    else cShape = @[@(m), @(n)];

    MPSGraphTensorData* cData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:cBuf shape:cShape dataType:MPSDataTypeFloat32];

    // Execute — synchronous, Apple handles compilation caching
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
        [e->graph runWithMTLCommandQueue:g_queue
                                  feeds:@{e->inputA: aData, e->inputB: bData}
                          targetTensors:@[e->result]
                       targetOperations:nil];

    // Copy result to caller's output buffer
    MPSGraphTensorData* resultData = results[e->result];
    if (resultData) {
        MPSNDArray* ndarray = [resultData mpsndarray];
        [ndarray readBytes:[cBuf contents] strideBytes:nil];
    }

    return 0;
    } // @autoreleasepool
}

// Batched matmul via MPSGraph: C[batch,m,n] = A[batch,m,k] @ B[batch,k,n]
// This is the key operation PyTorch uses for multi-head attention.
// One GPU dispatch for all heads simultaneously.
int mtl_graph_bmm(void* aRef, void* bRef, void* cRef,
                  int batch, int m, int k, int n) {
    @autoreleasepool {
    // Build a unique graph for this (batch,m,k,n) shape
    // Using a simple inline graph — no caching needed, MPSGraph caches internally
    MPSGraph* graph = [[MPSGraph alloc] init];

    MPSGraphTensor* inputA = [graph placeholderWithShape:@[@(batch), @(m), @(k)]
                                                dataType:MPSDataTypeFloat32 name:@"A"];
    MPSGraphTensor* inputB = [graph placeholderWithShape:@[@(batch), @(k), @(n)]
                                                dataType:MPSDataTypeFloat32 name:@"B"];
    MPSGraphTensor* result = [graph matrixMultiplicationWithPrimaryTensor:inputA
                                                        secondaryTensor:inputB name:@"C"];

    id<MTLBuffer> aBuf = (__bridge id<MTLBuffer>)aRef;
    id<MTLBuffer> bBuf = (__bridge id<MTLBuffer>)bRef;
    id<MTLBuffer> cBuf = (__bridge id<MTLBuffer>)cRef;

    MPSGraphTensorData* aData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:aBuf shape:@[@(batch), @(m), @(k)] dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* bData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:bBuf shape:@[@(batch), @(k), @(n)] dataType:MPSDataTypeFloat32];

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
        [graph runWithMTLCommandQueue:g_queue
                              feeds:@{inputA: aData, inputB: bData}
                      targetTensors:@[result]
                   targetOperations:nil];

    MPSGraphTensorData* resultData = results[result];
    if (resultData) {
        MPSNDArray* ndarray = [resultData mpsndarray];
        [ndarray readBytes:[cBuf contents] strideBytes:nil];
    }

    return 0;
    } // @autoreleasepool
}

// Batched matmul transposed: C[batch,m,n] = A[batch,m,k] @ B[batch,n,k]^T
int mtl_graph_bmm_transB(void* aRef, void* bRef, void* cRef,
                         int batch, int m, int k, int n) {
    @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];

    MPSGraphTensor* inputA = [graph placeholderWithShape:@[@(batch), @(m), @(k)]
                                                dataType:MPSDataTypeFloat32 name:@"A"];
    MPSGraphTensor* inputB = [graph placeholderWithShape:@[@(batch), @(n), @(k)]
                                                dataType:MPSDataTypeFloat32 name:@"B"];
    MPSGraphTensor* bT = [graph transposeTensor:inputB dimension:1 withDimension:2 name:@"BT"];
    MPSGraphTensor* result = [graph matrixMultiplicationWithPrimaryTensor:inputA
                                                        secondaryTensor:bT name:@"C"];

    id<MTLBuffer> aBuf = (__bridge id<MTLBuffer>)aRef;
    id<MTLBuffer> bBuf = (__bridge id<MTLBuffer>)bRef;
    id<MTLBuffer> cBuf = (__bridge id<MTLBuffer>)cRef;

    MPSGraphTensorData* aData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:aBuf shape:@[@(batch), @(m), @(k)] dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* bData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:bBuf shape:@[@(batch), @(n), @(k)] dataType:MPSDataTypeFloat32];

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
        [graph runWithMTLCommandQueue:g_queue
                              feeds:@{inputA: aData, inputB: bData}
                      targetTensors:@[result]
                   targetOperations:nil];

    MPSGraphTensorData* resultData = results[result];
    if (resultData) {
        MPSNDArray* ndarray = [resultData mpsndarray];
        [ndarray readBytes:[cBuf contents] strideBytes:nil];
    }

    return 0;
    } // @autoreleasepool
}

// Synchronize — wait for all queued MPSGraph work to complete
void mtl_graph_sync(void) {
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    [cmd commit];
    [cmd waitUntilCompleted];
}

// ============================================================
// Graph Training Engine — full layer forward+backward in one dispatch
// ============================================================
//
// Architecture: the graph handles the matmul-heavy path.
// CPU/AMX handles RMSNorm, RoPE, bias, residuals (fast on small vectors).
// Shared memory — zero copy between graph and CPU.
//
// Per layer, the graph computes:
//   Forward:  Q = X @ Wq^T, K = X @ Wk^T, V = X @ Wv^T
//             scores = Q_heads @ K_heads^T, softmax, attn = scores @ V_heads
//             proj = attn_flat @ Wo^T
//             gate = X2 @ Wg^T, up = X2 @ Wu^T, ffn = silu(gate)*up
//             down = ffn @ Wd^T
//   Backward: autograd via MPSGraph gradient API
//   AdamW:    on the 7 weight matrices

// Training graph state — compiled once, reused every step.
typedef struct {
    MPSGraph* graph;

    // Placeholders (fed each step)
    MPSGraphTensor* inputX;    // [n, dim] — normed input for QKV
    MPSGraphTensor* inputX2;   // [n, dim] — normed input for FFN
    MPSGraphTensor* inputMask; // [n, n] — causal mask (-inf upper triangle)

    // Weight placeholders (fed from shared memory each step)
    MPSGraphTensor* pWq, *pWk, *pWv, *pWo;
    MPSGraphTensor* pWg, *pWu, *pWd;

    // Outputs
    MPSGraphTensor* attnOut;  // [n, dim] — attention output (before residual)
    MPSGraphTensor* ffnOut;   // [n, dim] — FFN output (before residual)
    MPSGraphTensor* loss;     // scalar — for gradient computation

    // Adam state per weight
    // (MPSGraph has built-in optimizer ops — use stochasticGradientDescent or adam)

    int dim, kvDim, headDim, nHeads, nKVHeads, ffnDim, seqLen;
    int compiled;
} train_graph_t;

#define MAX_LAYERS 32
static train_graph_t g_layer_graphs[MAX_LAYERS];
static int g_num_layer_graphs = 0;

// Build the graph for one layer's forward pass.
// Takes post-RoPE Q, K, V and normed X2 as inputs (CPU handles norm+proj+bias+RoPE).
// Graph handles: attention (Q@K^T → mask → softmax → @V → WO) + FFN (gate/up/silu/down).
// Returns the layer index.
int mtl_graph_build_layer(int dim, int kvDim, int headDim,
                          int nHeads, int nKVHeads, int ffnDim, int seqLen,
                          void* wqBuf, void* wkBuf, void* wvBuf, void* woBuf,
                          void* wgBuf, void* wuBuf, void* wdBuf) {
    if (g_num_layer_graphs >= MAX_LAYERS) return -1;
    int idx = g_num_layer_graphs++;
    train_graph_t* tg = &g_layer_graphs[idx];
    tg->dim = dim; tg->kvDim = kvDim; tg->headDim = headDim;
    tg->nHeads = nHeads; tg->nKVHeads = nKVHeads; tg->ffnDim = ffnDim;
    tg->seqLen = seqLen;

    int n = seqLen + 1;

    MPSGraph* graph = [[MPSGraph alloc] init];
    tg->graph = graph;

    // Input placeholders:
    // Q[n, dim] — post bias+RoPE, from CPU
    // K[n, kvDim] — post bias+RoPE, from CPU
    // V[n, kvDim] — post bias, from CPU
    // X2[n, dim] — post-attention normed input for FFN, from CPU
    tg->inputX = [graph placeholderWithShape:@[@(n), @(dim)]
                                    dataType:MPSDataTypeFloat32 name:@"Q"];
    tg->pWk = [graph placeholderWithShape:@[@(n), @(kvDim)]
                                 dataType:MPSDataTypeFloat32 name:@"K"];
    tg->pWv = [graph placeholderWithShape:@[@(n), @(kvDim)]
                                 dataType:MPSDataTypeFloat32 name:@"V"];
    tg->inputX2 = [graph placeholderWithShape:@[@(n), @(dim)]
                                     dataType:MPSDataTypeFloat32 name:@"X2"];

    // Weight placeholders (only WO, gate, up, down — QKV done on CPU)
    tg->pWo = [graph placeholderWithShape:@[@(dim), @(dim)]
                                 dataType:MPSDataTypeFloat32 name:@"Wo"];
    tg->pWg = [graph placeholderWithShape:@[@(ffnDim), @(dim)]
                                 dataType:MPSDataTypeFloat32 name:@"Wg"];
    tg->pWu = [graph placeholderWithShape:@[@(ffnDim), @(dim)]
                                 dataType:MPSDataTypeFloat32 name:@"Wu"];
    tg->pWd = [graph placeholderWithShape:@[@(dim), @(ffnDim)]
                                 dataType:MPSDataTypeFloat32 name:@"Wd"];

    // Unused placeholders (QKV projections done on CPU now)
    tg->pWq = nil;

    MPSGraphTensor* Q = tg->inputX;   // [n, dim]
    MPSGraphTensor* K = tg->pWk;      // [n, kvDim]
    MPSGraphTensor* V = tg->pWv;      // [n, kvDim]

    // === Reshape for multi-head attention ===
    MPSGraphTensor* Qr = [graph reshapeTensor:Q withShape:@[@(n), @(nHeads), @(headDim)] name:@"Qr"];
    MPSGraphTensor* Qh = [graph transposeTensor:Qr dimension:0 withDimension:1 name:@"Qh"];

    MPSGraphTensor* Kr = [graph reshapeTensor:K withShape:@[@(n), @(nKVHeads), @(headDim)] name:@"Kr"];
    MPSGraphTensor* Kh = [graph transposeTensor:Kr dimension:0 withDimension:1 name:@"Kh"];
    MPSGraphTensor* Vr = [graph reshapeTensor:V withShape:@[@(n), @(nKVHeads), @(headDim)] name:@"Vr"];
    MPSGraphTensor* Vh = [graph transposeTensor:Vr dimension:0 withDimension:1 name:@"Vh"];

    // GQA: repeat K,V heads
    if (nKVHeads < nHeads) {
        int rep = nHeads / nKVHeads;
        Kh = [graph tileTensor:Kh withMultiplier:@[@(rep), @1, @1] name:@"Kh_rep"];
        Vh = [graph tileTensor:Vh withMultiplier:@[@(rep), @1, @1] name:@"Vh_rep"];
    }

    // === Attention scores ===
    // scores[nHeads, n, n] = Qh[nHeads, n, headDim] @ Kh[nHeads, headDim, n]
    MPSGraphTensor* KhT = [graph transposeTensor:Kh dimension:1 withDimension:2 name:@"KhT"];
    MPSGraphTensor* scores = [graph matrixMultiplicationWithPrimaryTensor:Qh
                                                         secondaryTensor:KhT name:@"scores"];

    // Scale by 1/sqrt(headDim)
    float scaleVal = 1.0f / sqrtf((float)headDim);
    MPSGraphTensor* scale = [graph constantWithScalar:scaleVal dataType:MPSDataTypeFloat32];
    scores = [graph multiplicationWithPrimaryTensor:scores secondaryTensor:scale name:@"scaled"];

    // Causal mask: upper triangle = -inf
    // Build mask as a constant tensor
    int nn = n * n;
    float* maskData = (float*)malloc(nn * sizeof(float));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            maskData[i * n + j] = (j > i) ? -1e9f : 0.0f;
        }
    }
    NSData* maskNSData = [NSData dataWithBytes:maskData length:nn * sizeof(float)];
    MPSGraphTensor* mask = [graph constantWithData:maskNSData shape:@[@1, @(n), @(n)]
                                          dataType:MPSDataTypeFloat32];
    free(maskData);

    scores = [graph additionWithPrimaryTensor:scores secondaryTensor:mask name:@"masked"];

    // Softmax over last dimension
    scores = [graph softMaxWithTensor:scores axis:-1 name:@"attn_weights"];

    // === Attention output ===
    // attn[nHeads, n, headDim] = scores[nHeads, n, n] @ Vh[nHeads, n, headDim]
    MPSGraphTensor* attnHeads = [graph matrixMultiplicationWithPrimaryTensor:scores
                                                            secondaryTensor:Vh name:@"attn"];

    // Reshape back: [nHeads, n, headDim] → [n, nHeads, headDim] → [n, dim]
    MPSGraphTensor* attnT = [graph transposeTensor:attnHeads dimension:0 withDimension:1 name:@"attnT"];
    MPSGraphTensor* attnFlat = [graph reshapeTensor:attnT withShape:@[@(n), @(dim)] name:@"attnFlat"];

    // Output projection: proj[n, dim] = attnFlat[n, dim] @ Wo[dim, dim]^T
    MPSGraphTensor* woT = [graph transposeTensor:tg->pWo dimension:0 withDimension:1 name:@"WoT"];
    tg->attnOut = [graph matrixMultiplicationWithPrimaryTensor:attnFlat
                                              secondaryTensor:woT name:@"proj"];

    // === FFN ===
    // gate[n, ffnDim] = X2[n, dim] @ Wg[ffnDim, dim]^T
    MPSGraphTensor* wgT = [graph transposeTensor:tg->pWg dimension:0 withDimension:1 name:@"WgT"];
    MPSGraphTensor* gate = [graph matrixMultiplicationWithPrimaryTensor:tg->inputX2
                                                       secondaryTensor:wgT name:@"gate"];
    // up[n, ffnDim] = X2[n, dim] @ Wu[ffnDim, dim]^T
    MPSGraphTensor* wuT = [graph transposeTensor:tg->pWu dimension:0 withDimension:1 name:@"WuT"];
    MPSGraphTensor* up = [graph matrixMultiplicationWithPrimaryTensor:tg->inputX2
                                                     secondaryTensor:wuT name:@"up"];

    // SiLU(gate) * up — MPSGraph has sigmoid, multiply ops
    MPSGraphTensor* gateSig = [graph sigmoidWithTensor:gate name:@"gate_sig"];
    MPSGraphTensor* gateSilu = [graph multiplicationWithPrimaryTensor:gate
                                                     secondaryTensor:gateSig name:@"gate_silu"];
    MPSGraphTensor* ffnMid = [graph multiplicationWithPrimaryTensor:gateSilu
                                                    secondaryTensor:up name:@"ffn_mid"];

    // down[n, dim] = ffnMid[n, ffnDim] @ Wd[dim, ffnDim]^T
    MPSGraphTensor* wdT = [graph transposeTensor:tg->pWd dimension:0 withDimension:1 name:@"WdT"];
    tg->ffnOut = [graph matrixMultiplicationWithPrimaryTensor:ffnMid
                                             secondaryTensor:wdT name:@"down"];

    tg->compiled = 1;
    return idx;
}

// Execute one layer's forward pass through the compiled graph.
// CPU provides: Q (post-RoPE), K (post-RoPE), V, X2 (normed FFN input), weights.
// GPU computes: attention (score → mask → softmax → @V → WO proj) + FFN.
// ONE dispatch for all the matmul-heavy work.
int mtl_graph_layer_forward(int layerIdx,
                            void* qBuf, void* kBuf, void* vBuf, void* x2Buf,
                            void* woBuf, void* wgBuf, void* wuBuf, void* wdBuf,
                            void* attnOutBuf, void* ffnOutBuf,
                            int n) {
    train_graph_t* tg = &g_layer_graphs[layerIdx];
    if (!tg->compiled) return -1;

    @autoreleasepool {

    int dim = tg->dim;
    int kvDim = tg->kvDim;
    int ffnDim = tg->ffnDim;

    // Wrap shared memory buffers as MPSGraphTensorData
    MPSGraphTensorData* qData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:(__bridge id<MTLBuffer>)qBuf
                    shape:@[@(n), @(dim)] dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* kData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:(__bridge id<MTLBuffer>)kBuf
                    shape:@[@(n), @(kvDim)] dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* vData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:(__bridge id<MTLBuffer>)vBuf
                    shape:@[@(n), @(kvDim)] dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* x2Data = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:(__bridge id<MTLBuffer>)x2Buf
                    shape:@[@(n), @(dim)] dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* woData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:(__bridge id<MTLBuffer>)woBuf
                    shape:@[@(dim), @(dim)] dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* wgData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:(__bridge id<MTLBuffer>)wgBuf
                    shape:@[@(ffnDim), @(dim)] dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* wuData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:(__bridge id<MTLBuffer>)wuBuf
                    shape:@[@(ffnDim), @(dim)] dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* wdData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:(__bridge id<MTLBuffer>)wdBuf
                    shape:@[@(dim), @(ffnDim)] dataType:MPSDataTypeFloat32];

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
        tg->inputX:  qData,   // Q (repurposed inputX placeholder)
        tg->pWk:     kData,   // K (repurposed pWk placeholder)
        tg->pWv:     vData,   // V (repurposed pWv placeholder)
        tg->inputX2: x2Data,
        tg->pWo: woData, tg->pWg: wgData, tg->pWu: wuData, tg->pWd: wdData,
    };

    // ONE dispatch — Apple's graph compiler fuses and optimizes
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
        [tg->graph runWithMTLCommandQueue:g_queue
                                   feeds:feeds
                           targetTensors:@[tg->attnOut, tg->ffnOut]
                        targetOperations:nil];

    // Results are in graph-managed memory. Copy to caller's shared memory buffers.
    id<MTLBuffer> attnBuf = (__bridge id<MTLBuffer>)attnOutBuf;
    id<MTLBuffer> ffnBufOut = (__bridge id<MTLBuffer>)ffnOutBuf;

    MPSGraphTensorData* attnResult = results[tg->attnOut];
    MPSGraphTensorData* ffnResult = results[tg->ffnOut];
    if (attnResult) {
        [[attnResult mpsndarray] readBytes:[attnBuf contents] strideBytes:nil];
    }
    if (ffnResult) {
        [[ffnResult mpsndarray] readBytes:[ffnBufOut contents] strideBytes:nil];
    }

    return 0;
    } // @autoreleasepool
}

int mtl_graph_num_layers(void) { return g_num_layer_graphs; }

// ============================================================
// Metal Compute Shaders — element-wise training ops on GPU
// ============================================================

// MSL source compiled once at first use via MTLDevice.newLibrary(source:)
static NSString* const g_kernel_source = @"\n"
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"kernel void add_inplace(device float* a [[buffer(0)]],\n"
"                        device const float* b [[buffer(1)]],\n"
"                        uint id [[thread_position_in_grid]]) {\n"
"    a[id] += b[id];\n"
"}\n"
"\n"
"kernel void add_out(device const float* a [[buffer(0)]],\n"
"                    device const float* b [[buffer(1)]],\n"
"                    device float* c [[buffer(2)]],\n"
"                    uint id [[thread_position_in_grid]]) {\n"
"    c[id] = a[id] + b[id];\n"
"}\n"
"\n"
"kernel void scale_inplace(device float* a [[buffer(0)]],\n"
"                          device const float* s [[buffer(1)]],\n"
"                          uint id [[thread_position_in_grid]]) {\n"
"    a[id] *= s[0];\n"
"}\n"
"\n"
"kernel void scale_out(device const float* a [[buffer(0)]],\n"
"                      device float* b [[buffer(1)]],\n"
"                      device const float* s [[buffer(2)]],\n"
"                      uint id [[thread_position_in_grid]]) {\n"
"    b[id] = a[id] * s[0];\n"
"}\n"
"\n"
"kernel void relu_inplace(device float* x [[buffer(0)]],\n"
"                         uint id [[thread_position_in_grid]]) {\n"
"    x[id] = max(x[id], 0.0f);\n"
"}\n"
"\n"
"kernel void relu_out(device const float* x [[buffer(0)]],\n"
"                     device float* y [[buffer(1)]],\n"
"                     uint id [[thread_position_in_grid]]) {\n"
"    y[id] = max(x[id], 0.0f);\n"
"}\n"
"\n"
"kernel void relu_backward(device const float* dOut [[buffer(0)]],\n"
"                          device const float* fwd [[buffer(1)]],\n"
"                          device float* result [[buffer(2)]],\n"
"                          uint id [[thread_position_in_grid]]) {\n"
"    result[id] = fwd[id] > 0.0f ? dOut[id] : 0.0f;\n"
"}\n"
"\n"
"kernel void silu_inplace(device float* x [[buffer(0)]],\n"
"                         uint id [[thread_position_in_grid]]) {\n"
"    float v = x[id];\n"
"    x[id] = v / (1.0f + exp(-v));\n"
"}\n"
"\n"
"kernel void silu_gate_mul(device const float* gate [[buffer(0)]],\n"
"                          device const float* up [[buffer(1)]],\n"
"                          device float* out [[buffer(2)]],\n"
"                          uint id [[thread_position_in_grid]]) {\n"
"    float g = gate[id];\n"
"    out[id] = (g / (1.0f + exp(-g))) * up[id];\n"
"}\n"
"\n"
"kernel void silu_gate_backward(device const float* dOut [[buffer(0)]],\n"
"                               device const float* gatePre [[buffer(1)]],\n"
"                               device const float* upOut [[buffer(2)]],\n"
"                               device const float* gateAct [[buffer(3)]],\n"
"                               device float* dGatePre [[buffer(4)]],\n"
"                               device float* dUp [[buffer(5)]],\n"
"                               uint id [[thread_position_in_grid]]) {\n"
"    float d = dOut[id];\n"
"    float sig = 1.0f / (1.0f + exp(-gatePre[id]));\n"
"    float dGateAct = d * upOut[id];\n"
"    dUp[id] = d * gateAct[id];\n"
"    dGatePre[id] = dGateAct * (sig + gateAct[id] * (1.0f - sig));\n"
"}\n"
"\n"
"// outer_add: G[i*cols+j] += a[i] * b[j] — gradient accumulation\n"
"kernel void outer_add(device float* G [[buffer(0)]],\n"
"                      device const float* a [[buffer(1)]],\n"
"                      device const float* b [[buffer(2)]],\n"
"                      constant uint& cols [[buffer(3)]],\n"
"                      uint2 gid [[thread_position_in_grid]]) {\n"
"    uint i = gid.y;\n"
"    uint j = gid.x;\n"
"    G[i * cols + j] += a[i] * b[j];\n"
"}\n"
"\n"
"// RMSNorm: x = x / rms * weight, one threadgroup per row\n"
"kernel void rmsnorm(device float* x [[buffer(0)]],\n"
"                    device const float* weight [[buffer(1)]],\n"
"                    constant uint& dim [[buffer(2)]],\n"
"                    constant float& eps [[buffer(3)]],\n"
"                    uint row [[threadgroup_position_in_grid]],\n"
"                    uint tid [[thread_index_in_threadgroup]],\n"
"                    uint tpg [[threads_per_threadgroup]]) {\n"
"    uint base = row * dim;\n"
"    // Compute sum of squares (strided reduction)\n"
"    float ss = 0.0f;\n"
"    for (uint i = tid; i < dim; i += tpg) {\n"
"        float v = x[base + i];\n"
"        ss += v * v;\n"
"    }\n"
"    // Warp reduction via simd_sum\n"
"    ss = simd_sum(ss);\n"
"    // Threadgroup reduction (if > 1 simdgroup)\n"
"    threadgroup float shared_ss[32];\n"
"    uint lane = tid % 32;\n"
"    uint warp = tid / 32;\n"
"    if (lane == 0) shared_ss[warp] = ss;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (warp == 0) {\n"
"        ss = (lane < (tpg + 31) / 32) ? shared_ss[lane] : 0.0f;\n"
"        ss = simd_sum(ss);\n"
"    }\n"
"    threadgroup float final_scale;\n"
"    if (tid == 0) {\n"
"        final_scale = rsqrt(ss / float(dim) + eps);\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float s = final_scale;\n"
"    for (uint i = tid; i < dim; i += tpg) {\n"
"        x[base + i] = x[base + i] * s * weight[i];\n"
"    }\n"
"}\n"
"\n"
"// AdamW update — one thread per parameter\n"
"kernel void adamw(device float* param [[buffer(0)]],\n"
"                  device const float* grad [[buffer(1)]],\n"
"                  device float* m [[buffer(2)]],\n"
"                  device float* v [[buffer(3)]],\n"
"                  constant float& lr [[buffer(4)]],\n"
"                  constant float& beta1 [[buffer(5)]],\n"
"                  constant float& beta2 [[buffer(6)]],\n"
"                  constant float& bc1 [[buffer(7)]],\n"
"                  constant float& bc2 [[buffer(8)]],\n"
"                  constant float& eps [[buffer(9)]],\n"
"                  constant float& wd [[buffer(10)]],\n"
"                  uint id [[thread_position_in_grid]]) {\n"
"    float g = grad[id];\n"
"    float m_new = beta1 * m[id] + (1.0f - beta1) * g;\n"
"    float v_new = beta2 * v[id] + (1.0f - beta2) * g * g;\n"
"    m[id] = m_new;\n"
"    v[id] = v_new;\n"
"    float mh = m_new / bc1;\n"
"    float vh = v_new / bc2;\n"
"    param[id] -= lr * (mh / (sqrt(vh) + eps) + wd * param[id]);\n"
"}\n"
"\n"
"// Zero memory\n"
"kernel void zero_mem(device float* ptr [[buffer(0)]],\n"
"                     uint id [[thread_position_in_grid]]) {\n"
"    ptr[id] = 0.0f;\n"
"}\n"
;

// Compute pipeline state objects — lazily initialized
static id<MTLLibrary> g_compute_lib = nil;
static id<MTLComputePipelineState> g_ps_add_inplace = nil;
static id<MTLComputePipelineState> g_ps_add_out = nil;
static id<MTLComputePipelineState> g_ps_scale_inplace = nil;
static id<MTLComputePipelineState> g_ps_scale_out = nil;
static id<MTLComputePipelineState> g_ps_relu_inplace = nil;
static id<MTLComputePipelineState> g_ps_relu_out = nil;
static id<MTLComputePipelineState> g_ps_relu_backward = nil;
static id<MTLComputePipelineState> g_ps_silu_inplace = nil;
static id<MTLComputePipelineState> g_ps_silu_gate_mul = nil;
static id<MTLComputePipelineState> g_ps_silu_gate_backward = nil;
static id<MTLComputePipelineState> g_ps_outer_add = nil;
static id<MTLComputePipelineState> g_ps_rmsnorm = nil;
static id<MTLComputePipelineState> g_ps_adamw = nil;
static id<MTLComputePipelineState> g_ps_zero_mem = nil;

static id<MTLComputePipelineState> make_ps(NSString* name) {
    NSError* err = nil;
    id<MTLFunction> fn = [g_compute_lib newFunctionWithName:name];
    if (!fn) { NSLog(@"mongoose: kernel %@ not found", name); return nil; }
    id<MTLComputePipelineState> ps = [g_device newComputePipelineStateWithFunction:fn error:&err];
    if (err) { NSLog(@"mongoose: pipeline %@: %@", name, err); return nil; }
    return ps;
}

int mtl_init_compute(void) {
    if (g_compute_lib) return 0;
    NSError* err = nil;
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.mathMode = MTLMathModeFast;
    g_compute_lib = [g_device newLibraryWithSource:g_kernel_source options:opts error:&err];
    if (err || !g_compute_lib) {
        NSLog(@"mongoose: compile kernels: %@", err);
        return -1;
    }
    g_ps_add_inplace = make_ps(@"add_inplace");
    g_ps_add_out = make_ps(@"add_out");
    g_ps_scale_inplace = make_ps(@"scale_inplace");
    g_ps_scale_out = make_ps(@"scale_out");
    g_ps_relu_inplace = make_ps(@"relu_inplace");
    g_ps_relu_out = make_ps(@"relu_out");
    g_ps_relu_backward = make_ps(@"relu_backward");
    g_ps_silu_inplace = make_ps(@"silu_inplace");
    g_ps_silu_gate_mul = make_ps(@"silu_gate_mul");
    g_ps_silu_gate_backward = make_ps(@"silu_gate_backward");
    g_ps_outer_add = make_ps(@"outer_add");
    g_ps_rmsnorm = make_ps(@"rmsnorm");
    g_ps_adamw = make_ps(@"adamw");
    g_ps_zero_mem = make_ps(@"zero_mem");
    return 0;
}

int mtl_compute_ready(void) {
    return g_compute_lib != nil ? 1 : 0;
}

// --- Dispatch helpers ---

// Encode a 1D compute dispatch. Buffers set individually per call to avoid ARC array issues.
static void dispatch_1d_setup(id<MTLComputePipelineState> ps, uint n,
                              id<MTLCommandBuffer>* outCmd,
                              id<MTLComputeCommandEncoder>* outEnc) {
    *outCmd = get_cmd();
    *outEnc = [*outCmd computeCommandEncoder];
    [*outEnc setComputePipelineState:ps];
}

static void dispatch_1d_finish(id<MTLComputePipelineState> ps, uint n,
                                id<MTLCommandBuffer> cmd,
                                id<MTLComputeCommandEncoder> enc) {
    NSUInteger tpg = ps.maxTotalThreadsPerThreadgroup;
    if (tpg > n) tpg = n;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    [enc endEncoding];
    maybe_commit(cmd);
}

// --- Kernel dispatch functions (called from Go via CGo) ---

void mtl_add_inplace(void* aRef, void* bRef, int n) {
    id<MTLCommandBuffer> cmd; id<MTLComputeCommandEncoder> enc;
    dispatch_1d_setup(g_ps_add_inplace, n, &cmd, &enc);
    [enc setBuffer:(__bridge id<MTLBuffer>)aRef offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)bRef offset:0 atIndex:1];
    dispatch_1d_finish(g_ps_add_inplace, n, cmd, enc);
}

void mtl_add_out(void* aRef, void* bRef, void* cRef, int n) {
    id<MTLCommandBuffer> cmd; id<MTLComputeCommandEncoder> enc;
    dispatch_1d_setup(g_ps_add_out, n, &cmd, &enc);
    [enc setBuffer:(__bridge id<MTLBuffer>)aRef offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)bRef offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)cRef offset:0 atIndex:2];
    dispatch_1d_finish(g_ps_add_out, n, cmd, enc);
}

void mtl_scale_out(void* aRef, void* bRef, void* sRef, int n) {
    id<MTLCommandBuffer> cmd; id<MTLComputeCommandEncoder> enc;
    dispatch_1d_setup(g_ps_scale_out, n, &cmd, &enc);
    [enc setBuffer:(__bridge id<MTLBuffer>)aRef offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)bRef offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)sRef offset:0 atIndex:2];
    dispatch_1d_finish(g_ps_scale_out, n, cmd, enc);
}

void mtl_relu_out(void* xRef, void* yRef, int n) {
    id<MTLCommandBuffer> cmd; id<MTLComputeCommandEncoder> enc;
    dispatch_1d_setup(g_ps_relu_out, n, &cmd, &enc);
    [enc setBuffer:(__bridge id<MTLBuffer>)xRef offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)yRef offset:0 atIndex:1];
    dispatch_1d_finish(g_ps_relu_out, n, cmd, enc);
}

void mtl_relu_backward_gpu(void* dOutRef, void* fwdRef, void* resultRef, int n) {
    id<MTLCommandBuffer> cmd; id<MTLComputeCommandEncoder> enc;
    dispatch_1d_setup(g_ps_relu_backward, n, &cmd, &enc);
    [enc setBuffer:(__bridge id<MTLBuffer>)dOutRef    offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)fwdRef     offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)resultRef  offset:0 atIndex:2];
    dispatch_1d_finish(g_ps_relu_backward, n, cmd, enc);
}

void mtl_silu_gate_mul(void* gateRef, void* upRef, void* outRef, int n) {
    id<MTLCommandBuffer> cmd; id<MTLComputeCommandEncoder> enc;
    dispatch_1d_setup(g_ps_silu_gate_mul, n, &cmd, &enc);
    [enc setBuffer:(__bridge id<MTLBuffer>)gateRef offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)upRef   offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)outRef  offset:0 atIndex:2];
    dispatch_1d_finish(g_ps_silu_gate_mul, n, cmd, enc);
}

void mtl_silu_gate_backward_gpu(void* dOutRef, void* gatePreRef, void* upOutRef,
                                void* gateActRef, void* dGatePreRef, void* dUpRef, int n) {
    id<MTLCommandBuffer> cmd; id<MTLComputeCommandEncoder> enc;
    dispatch_1d_setup(g_ps_silu_gate_backward, n, &cmd, &enc);
    [enc setBuffer:(__bridge id<MTLBuffer>)dOutRef     offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)gatePreRef  offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)upOutRef    offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)gateActRef  offset:0 atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)dGatePreRef offset:0 atIndex:4];
    [enc setBuffer:(__bridge id<MTLBuffer>)dUpRef      offset:0 atIndex:5];
    dispatch_1d_finish(g_ps_silu_gate_backward, n, cmd, enc);
}

void mtl_outer_add(void* gRef, void* aRef, void* bRef, int rows, int cols) {
    @autoreleasepool {
    id<MTLBuffer> aBuf = (__bridge id<MTLBuffer>)aRef;
    id<MTLBuffer> bBuf = (__bridge id<MTLBuffer>)bRef;
    id<MTLBuffer> gBuf = (__bridge id<MTLBuffer>)gRef;

    // Allocate a tiny constant buffer for cols
    uint32_t colsVal = (uint32_t)cols;
    id<MTLBuffer> colsBuf = [g_device newBufferWithBytes:&colsVal length:sizeof(uint32_t)
                                                 options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cmd = get_cmd();
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_ps_outer_add];
    [enc setBuffer:gBuf offset:0 atIndex:0];
    [enc setBuffer:aBuf offset:0 atIndex:1];
    [enc setBuffer:bBuf offset:0 atIndex:2];
    [enc setBuffer:colsBuf offset:0 atIndex:3];

    NSUInteger tpg = 16; // 16x16 threadgroup for 2D
    MTLSize threads = MTLSizeMake(cols, rows, 1);
    MTLSize tgSize = MTLSizeMake(tpg, tpg, 1);
    [enc dispatchThreads:threads threadsPerThreadgroup:tgSize];
    [enc endEncoding];
    maybe_commit(cmd);
    } // @autoreleasepool
}

void mtl_rmsnorm_gpu(void* xRef, void* weightRef, int seqLen, int dim) {
    @autoreleasepool {
    id<MTLBuffer> xBuf = (__bridge id<MTLBuffer>)xRef;
    id<MTLBuffer> wBuf = (__bridge id<MTLBuffer>)weightRef;

    // Constant buffers
    uint32_t dimVal = (uint32_t)dim;
    float epsVal = 1e-6f;
    id<MTLBuffer> dimBuf = [g_device newBufferWithBytes:&dimVal length:sizeof(uint32_t)
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> epsBuf = [g_device newBufferWithBytes:&epsVal length:sizeof(float)
                                               options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cmd = get_cmd();
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_ps_rmsnorm];
    [enc setBuffer:xBuf offset:0 atIndex:0];
    [enc setBuffer:wBuf offset:0 atIndex:1];
    [enc setBuffer:dimBuf offset:0 atIndex:2];
    [enc setBuffer:epsBuf offset:0 atIndex:3];

    // One threadgroup per row, threads within threadgroup reduce over dim
    NSUInteger tpg = g_ps_rmsnorm.maxTotalThreadsPerThreadgroup;
    if (tpg > (NSUInteger)dim) tpg = (NSUInteger)dim;
    // Round down to multiple of 32 for simd_sum
    tpg = (tpg / 32) * 32;
    if (tpg == 0) tpg = 32;
    [enc dispatchThreadgroups:MTLSizeMake(seqLen, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    [enc endEncoding];
    maybe_commit(cmd);
    } // @autoreleasepool
}

void mtl_adamw_gpu(void* paramRef, void* gradRef, void* mRef, void* vRef,
                   float lr, float beta1, float beta2, float bc1, float bc2,
                   float eps, float wd, int n) {
    @autoreleasepool {
    // Scalar constant buffers
    id<MTLBuffer> lrBuf   = [g_device newBufferWithBytes:&lr    length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> b1Buf   = [g_device newBufferWithBytes:&beta1 length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> b2Buf   = [g_device newBufferWithBytes:&beta2 length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bc1Buf  = [g_device newBufferWithBytes:&bc1   length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bc2Buf  = [g_device newBufferWithBytes:&bc2   length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> epsBuf  = [g_device newBufferWithBytes:&eps   length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> wdBuf   = [g_device newBufferWithBytes:&wd    length:sizeof(float) options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cmd = get_cmd();
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_ps_adamw];
    [enc setBuffer:(__bridge id<MTLBuffer>)paramRef offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)gradRef  offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)mRef     offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)vRef     offset:0 atIndex:3];
    [enc setBuffer:lrBuf  offset:0 atIndex:4];
    [enc setBuffer:b1Buf  offset:0 atIndex:5];
    [enc setBuffer:b2Buf  offset:0 atIndex:6];
    [enc setBuffer:bc1Buf offset:0 atIndex:7];
    [enc setBuffer:bc2Buf offset:0 atIndex:8];
    [enc setBuffer:epsBuf offset:0 atIndex:9];
    [enc setBuffer:wdBuf  offset:0 atIndex:10];

    NSUInteger tpg = g_ps_adamw.maxTotalThreadsPerThreadgroup;
    if (tpg > (NSUInteger)n) tpg = (NSUInteger)n;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    [enc endEncoding];
    maybe_commit(cmd);
    } // @autoreleasepool
}

void mtl_zero_gpu(void* ptrRef, int n) {
    id<MTLCommandBuffer> cmd; id<MTLComputeCommandEncoder> enc;
    dispatch_1d_setup(g_ps_zero_mem, n, &cmd, &enc);
    [enc setBuffer:(__bridge id<MTLBuffer>)ptrRef offset:0 atIndex:0];
    dispatch_1d_finish(g_ps_zero_mem, n, cmd, enc);
}
