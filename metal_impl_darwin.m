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
#define MTL_MAX_QUEUES 4
id<MTLCommandQueue> g_queues[MTL_MAX_QUEUES] = {nil, nil, nil, nil};
id<MTLCommandQueue> g_queue = nil;   // alias for g_queues[0]
id<MTLCommandQueue> g_queue2 = nil;  // alias for g_queues[1]
id<MTLSharedEvent> g_coil_event = nil;
static char g_device_name[256] = {0};

static id<MTLCommandBuffer> g_batch_cmd = nil;

// Fused command buffers — 4 slots for parallel forward passes.
static id<MTLCommandBuffer> g_fused_cmd[MTL_MAX_QUEUES] = {nil, nil, nil, nil};
static id<MTLComputeCommandEncoder> g_fused_enc[MTL_MAX_QUEUES] = {nil, nil, nil, nil};
static int g_active_fused_slot = 0;

int mtl_init(void) {
    if (g_device) return 0;

    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) return -1;

    for (int i = 0; i < MTL_MAX_QUEUES; i++) {
        g_queues[i] = [g_device newCommandQueue];
        if (!g_queues[i]) return -2;
    }
    g_queue = g_queues[0];
    g_queue2 = g_queues[1];
    g_coil_event = [g_device newSharedEvent];

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

// End batch without waiting — commit to GPU queue, return immediately.
// The GPU executes async. Next command buffer on the same queue will
// implicitly wait for this one to finish (Metal command queue ordering).
void mtl_end_batch_async(void) {
    if (g_batch_cmd) {
        [g_batch_cmd commit];
        g_batch_cmd = nil;
    }
}

// Get command buffer — returns fused cmd if fusing, batch cmd if batching, otherwise new one.
static id<MTLCommandBuffer> get_cmd(void) {
    int s = g_active_fused_slot;
    if (g_fused_cmd[s]) return g_fused_cmd[s];
    if (g_batch_cmd) return g_batch_cmd;
    return [g_queue commandBuffer];
}

static id<MTLComputeCommandEncoder> get_enc(id<MTLCommandBuffer> cmd) {
    int s = g_active_fused_slot;
    if (g_fused_enc[s]) return g_fused_enc[s];
    return [cmd computeCommandEncoder];
}

static void end_enc(id<MTLComputeCommandEncoder> enc) {
    for (int i = 0; i < MTL_MAX_QUEUES; i++) {
        if (enc == g_fused_enc[i]) return;
    }
    [enc endEncoding];
}

static void maybe_commit(id<MTLCommandBuffer> cmd) {
    int s = g_active_fused_slot;
    if (g_fused_cmd[s]) return;
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
"    float ga = gatePre[id] * sig;  // silu(gatePre) — recomputed, gateAct buffer optional\n"
"    float dGateAct = d * upOut[id];\n"
"    dUp[id] = d * ga;\n"
"    dGatePre[id] = dGateAct * (sig + ga * (1.0f - sig));\n"
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
"\n"
"// ============================================================\n"
"// Forward-gradient LM head — no backward pass needed.\n"
"//\n"
"// Two-pass streaming over the embedding matrix:\n"
"//   Pass 1: dot products → max + sum_exp (2 floats per position)\n"
"//   Pass 2: dot products → softmax → gradient → accumulate dHidden\n"
"//\n"
"// No logits buffer allocated. dHidden computed during forward.\n"
"// The backward pass through the LM head is eliminated entirely.\n"
"// ============================================================\n"
"\n"
"// Cross-entropy loss — no gradient. One threadgroup per position.\n"
"// Reads logits[n, vocabSize], targets[n]. Writes losses[n].\n"
"// Cooperative reduction for numerical stability (log-sum-exp trick).\n"
"kernel void ce_loss(\n"
"    device const float* logits  [[buffer(0)]],  // [n, vocabSize]\n"
"    device const int* targets   [[buffer(1)]],   // [n] target token IDs\n"
"    device float* losses        [[buffer(2)]],   // [n] per-position loss\n"
"    constant uint& vocabSize   [[buffer(3)]],\n"
"    uint pos [[threadgroup_position_in_grid]],\n"
"    uint tid [[thread_index_in_threadgroup]],\n"
"    uint tpg [[threads_per_threadgroup]])\n"
"{\n"
"    uint off = pos * vocabSize;\n"
"    int target = targets[pos];\n"
"\n"
"    // Pass 1: find max logit (cooperative)\n"
"    float localMax = -1e30f;\n"
"    for (uint i = tid; i < vocabSize; i += tpg) {\n"
"        float v = logits[off + i];\n"
"        localMax = max(localMax, v);\n"
"    }\n"
"    localMax = simd_max(localMax);\n"
"    threadgroup float shared_max[32];\n"
"    uint lane = tid % 32, warp = tid / 32;\n"
"    if (lane == 0) shared_max[warp] = localMax;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (warp == 0) {\n"
"        localMax = (lane < (tpg + 31) / 32) ? shared_max[lane] : -1e30f;\n"
"        localMax = simd_max(localMax);\n"
"    }\n"
"    threadgroup float finalMax;\n"
"    if (tid == 0) finalMax = localMax;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"\n"
"    // Pass 2: sum of exp(logit - max)\n"
"    float localSum = 0.0f;\n"
"    for (uint i = tid; i < vocabSize; i += tpg) {\n"
"        localSum += exp(logits[off + i] - finalMax);\n"
"    }\n"
"    localSum = simd_sum(localSum);\n"
"    threadgroup float shared_sum[32];\n"
"    if (lane == 0) shared_sum[warp] = localSum;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (warp == 0) {\n"
"        localSum = (lane < (tpg + 31) / 32) ? shared_sum[lane] : 0.0f;\n"
"        localSum = simd_sum(localSum);\n"
"    }\n"
"    threadgroup float finalSum;\n"
"    if (tid == 0) finalSum = localSum;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"\n"
"    // Loss = -(logit[target] - max - log(sumExp))\n"
"    if (tid == 0) {\n"
"        float logProb = logits[off + target] - finalMax - log(finalSum);\n"
"        losses[pos] = -logProb;\n"
"    }\n"
"}\n"
"\n"
"// Pass 1: compute max logit and sum of exp for each position.\n"
"// One threadgroup per position. Threads cooperatively scan all vocab rows.\n"
"kernel void lm_head_pass1(\n"
"    device const float* hidden [[buffer(0)]],  // [n, dim]\n"
"    device const float* embed  [[buffer(1)]],  // [vocabSize, dim]\n"
"    device float* maxLogit     [[buffer(2)]],  // [n] output\n"
"    device float* sumExp       [[buffer(3)]],  // [n] output\n"
"    constant uint& dim        [[buffer(4)]],\n"
"    constant uint& vocabSize  [[buffer(5)]],\n"
"    constant uint& nPositions [[buffer(6)]],\n"
"    uint pos [[threadgroup_position_in_grid]],\n"
"    uint tid [[thread_index_in_threadgroup]],\n"
"    uint tpg [[threads_per_threadgroup]])\n"
"{\n"
"    if (pos >= nPositions) return;\n"
"    uint hOff = pos * dim;\n"
"\n"
"    // Each thread scans a strided subset of vocab rows\n"
"    float localMax = -1e30f;\n"
"    for (uint v = tid; v < vocabSize; v += tpg) {\n"
"        float dot = 0.0f;\n"
"        for (uint d = 0; d < dim; d++) {\n"
"            dot += hidden[hOff + d] * embed[v * dim + d];\n"
"        }\n"
"        localMax = max(localMax, dot);\n"
"    }\n"
"\n"
"    // Reduce max across threadgroup via simd_max\n"
"    localMax = simd_max(localMax);\n"
"    threadgroup float shared_max[32];\n"
"    uint lane = tid % 32;\n"
"    uint warp = tid / 32;\n"
"    if (lane == 0) shared_max[warp] = localMax;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (warp == 0) {\n"
"        localMax = (lane < (tpg + 31) / 32) ? shared_max[lane] : -1e30f;\n"
"        localMax = simd_max(localMax);\n"
"    }\n"
"    threadgroup float finalMax;\n"
"    if (tid == 0) finalMax = localMax;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float mx = finalMax;\n"
"\n"
"    // Sum exp(logit - max)\n"
"    float localSum = 0.0f;\n"
"    for (uint v = tid; v < vocabSize; v += tpg) {\n"
"        float dot = 0.0f;\n"
"        for (uint d = 0; d < dim; d++) {\n"
"            dot += hidden[hOff + d] * embed[v * dim + d];\n"
"        }\n"
"        localSum += exp(dot - mx);\n"
"    }\n"
"\n"
"    // Reduce sum across threadgroup\n"
"    localSum = simd_sum(localSum);\n"
"    threadgroup float shared_sum[32];\n"
"    if (lane == 0) shared_sum[warp] = localSum;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (warp == 0) {\n"
"        localSum = (lane < (tpg + 31) / 32) ? shared_sum[lane] : 0.0f;\n"
"        localSum = simd_sum(localSum);\n"
"    }\n"
"\n"
"    if (tid == 0) {\n"
"        maxLogit[pos] = mx;\n"
"        sumExp[pos] = localSum;\n"
"    }\n"
"}\n"
"\n"
"// Pass 2: compute loss + dHidden using max/sumExp from pass 1.\n"
"// Recomputes dot products (no logits buffer). Accumulates gradient\n"
"// into dHidden during the same scan. One threadgroup per position.\n"
"kernel void lm_head_pass2(\n"
"    device const float* hidden   [[buffer(0)]],  // [n, dim]\n"
"    device const float* embed    [[buffer(1)]],  // [vocabSize, dim]\n"
"    device const float* maxLogit [[buffer(2)]],  // [n] from pass 1\n"
"    device const float* sumExp   [[buffer(3)]],  // [n] from pass 1\n"
"    device const int* targets    [[buffer(4)]],  // [n] target token IDs\n"
"    device float* dHidden        [[buffer(5)]],  // [n, dim] output gradient\n"
"    device atomic_float* loss    [[buffer(6)]],  // scalar output (atomic add)\n"
"    constant uint& dim          [[buffer(7)]],\n"
"    constant uint& vocabSize    [[buffer(8)]],\n"
"    constant uint& nPositions   [[buffer(9)]],\n"
"    constant float& invN        [[buffer(10)]],  // 1.0 / nPositions\n"
"    uint pos [[threadgroup_position_in_grid]],\n"
"    uint tid [[thread_index_in_threadgroup]],\n"
"    uint tpg [[threads_per_threadgroup]])\n"
"{\n"
"    if (pos >= nPositions) return;\n"
"    uint hOff = pos * dim;\n"
"    float mx = maxLogit[pos];\n"
"    float se = sumExp[pos];\n"
"    float invSe = 1.0f / se;\n"
"    int target = targets[pos];\n"
"\n"
"    // Zero dHidden for this position\n"
"    for (uint d = tid; d < dim; d += tpg) {\n"
"        dHidden[hOff + d] = 0.0f;\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"\n"
"    // Loss from target logit\n"
"    if (tid == 0 && target >= 0 && (uint)target < vocabSize) {\n"
"        float targetDot = 0.0f;\n"
"        for (uint d = 0; d < dim; d++) {\n"
"            targetDot += hidden[hOff + d] * embed[target * dim + d];\n"
"        }\n"
"        float posLoss = -(targetDot - mx) + log(se);\n"
"        atomic_fetch_add_explicit(loss, posLoss * invN, memory_order_relaxed);\n"
"    }\n"
"\n"
"    // Gradient accumulation: scan all vocab rows, accumulate grad * embed into dHidden.\n"
"    // grad[v] = softmax(v) * invN, grad[target] -= invN\n"
"    // dHidden[pos] += sum_v grad[v] * embed[v]\n"
"    for (uint v = tid; v < vocabSize; v += tpg) {\n"
"        float dot = 0.0f;\n"
"        for (uint d = 0; d < dim; d++) {\n"
"            dot += hidden[hOff + d] * embed[v * dim + d];\n"
"        }\n"
"        float sv = exp(dot - mx) * invSe;  // softmax value\n"
"        float grad = sv * invN;\n"
"        if ((int)v == target) grad -= invN;\n"
"\n"
"        // Skip near-zero gradients (the sparsity win)\n"
"        if (abs(grad) < 1e-7f) continue;\n"
"\n"
"        // Accumulate grad * embed[v] into dHidden[pos]\n"
"        for (uint d = 0; d < dim; d++) {\n"
"            // Atomic add — multiple threads may write to same dHidden element\n"
"            atomic_fetch_add_explicit(\n"
"                (device atomic_float*)&dHidden[hOff + d],\n"
"                grad * embed[v * dim + d],\n"
"                memory_order_relaxed);\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"// ============================================================\n"
"// Sparse LM head — conductor-masked forward gradient\n"
"// ============================================================\n"
"// Same as lm_head_pass1/pass2 but only scans hot embedding rows.\n"
"// hotIndices[nHot] contains the active row indices from the conductor.\n"
"// At 99.7% sparsity (148/50257 hot), this is 340x fewer dot products.\n"
"\n"
"kernel void lm_head_sparse_pass1(\n"
"    device const float* hidden    [[buffer(0)]],\n"
"    device const float* embed     [[buffer(1)]],\n"
"    device float* maxLogit        [[buffer(2)]],\n"
"    device float* sumExp          [[buffer(3)]],\n"
"    device const int* hotIndices  [[buffer(4)]],\n"
"    constant uint& dim           [[buffer(5)]],\n"
"    constant uint& nHot          [[buffer(6)]],\n"
"    constant uint& nPositions    [[buffer(7)]],\n"
"    uint pos [[threadgroup_position_in_grid]],\n"
"    uint tid [[thread_index_in_threadgroup]],\n"
"    uint tpg [[threads_per_threadgroup]])\n"
"{\n"
"    if (pos >= nPositions) return;\n"
"    uint hOff = pos * dim;\n"
"\n"
"    float localMax = -1e30f;\n"
"    for (uint i = tid; i < nHot; i += tpg) {\n"
"        uint v = (uint)hotIndices[i];\n"
"        float dot = 0.0f;\n"
"        for (uint d = 0; d < dim; d++) {\n"
"            dot += hidden[hOff + d] * embed[v * dim + d];\n"
"        }\n"
"        localMax = max(localMax, dot);\n"
"    }\n"
"\n"
"    localMax = simd_max(localMax);\n"
"    threadgroup float shared_max[32];\n"
"    uint lane = tid % 32;\n"
"    uint warp = tid / 32;\n"
"    if (lane == 0) shared_max[warp] = localMax;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (warp == 0) {\n"
"        localMax = (lane < (tpg + 31) / 32) ? shared_max[lane] : -1e30f;\n"
"        localMax = simd_max(localMax);\n"
"    }\n"
"    threadgroup float finalMax;\n"
"    if (tid == 0) finalMax = localMax;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float mx = finalMax;\n"
"\n"
"    float localSum = 0.0f;\n"
"    for (uint i = tid; i < nHot; i += tpg) {\n"
"        uint v = (uint)hotIndices[i];\n"
"        float dot = 0.0f;\n"
"        for (uint d = 0; d < dim; d++) {\n"
"            dot += hidden[hOff + d] * embed[v * dim + d];\n"
"        }\n"
"        localSum += exp(dot - mx);\n"
"    }\n"
"\n"
"    localSum = simd_sum(localSum);\n"
"    threadgroup float shared_sum[32];\n"
"    if (lane == 0) shared_sum[warp] = localSum;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (warp == 0) {\n"
"        localSum = (lane < (tpg + 31) / 32) ? shared_sum[lane] : 0.0f;\n"
"        localSum = simd_sum(localSum);\n"
"    }\n"
"\n"
"    if (tid == 0) {\n"
"        maxLogit[pos] = mx;\n"
"        sumExp[pos] = localSum;\n"
"    }\n"
"}\n"
"\n"
"kernel void lm_head_sparse_pass2(\n"
"    device const float* hidden    [[buffer(0)]],\n"
"    device const float* embed     [[buffer(1)]],\n"
"    device const float* maxLogit  [[buffer(2)]],\n"
"    device const float* sumExp    [[buffer(3)]],\n"
"    device const int* targets     [[buffer(4)]],\n"
"    device float* dHidden         [[buffer(5)]],\n"
"    device atomic_float* loss     [[buffer(6)]],\n"
"    device const int* hotIndices  [[buffer(7)]],\n"
"    constant uint& dim           [[buffer(8)]],\n"
"    constant uint& nHot          [[buffer(9)]],\n"
"    constant uint& nPositions    [[buffer(10)]],\n"
"    constant float& invN         [[buffer(11)]],\n"
"    uint pos [[threadgroup_position_in_grid]],\n"
"    uint tid [[thread_index_in_threadgroup]],\n"
"    uint tpg [[threads_per_threadgroup]])\n"
"{\n"
"    if (pos >= nPositions) return;\n"
"    uint hOff = pos * dim;\n"
"    float mx = maxLogit[pos];\n"
"    float se = sumExp[pos];\n"
"    float invSe = 1.0f / se;\n"
"    int target = targets[pos];\n"
"\n"
"    for (uint d = tid; d < dim; d += tpg) {\n"
"        dHidden[hOff + d] = 0.0f;\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"\n"
"    if (tid == 0 && target >= 0) {\n"
"        float targetDot = 0.0f;\n"
"        for (uint d = 0; d < dim; d++) {\n"
"            targetDot += hidden[hOff + d] * embed[target * dim + d];\n"
"        }\n"
"        float posLoss = -(targetDot - mx) + log(se);\n"
"        atomic_fetch_add_explicit(loss, posLoss * invN, memory_order_relaxed);\n"
"    }\n"
"\n"
"    for (uint i = tid; i < nHot; i += tpg) {\n"
"        uint v = (uint)hotIndices[i];\n"
"        float dot = 0.0f;\n"
"        for (uint d = 0; d < dim; d++) {\n"
"            dot += hidden[hOff + d] * embed[v * dim + d];\n"
"        }\n"
"        float sv = exp(dot - mx) * invSe;\n"
"        float grad = sv * invN;\n"
"        if ((int)v == target) grad -= invN;\n"
"        if (abs(grad) < 1e-7f) continue;\n"
"        for (uint d = 0; d < dim; d++) {\n"
"            atomic_fetch_add_explicit(\n"
"                (device atomic_float*)&dHidden[hOff + d],\n"
"                grad * embed[v * dim + d],\n"
"                memory_order_relaxed);\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"// ============================================================\n"
"// DNA Gradient Descent — rung kernel\n"
"// ============================================================\n"
"//\n"
"// One thread per parameter in the coupled region.\n"
"// Computes the 6-point base pair update for both strands simultaneously.\n"
"//\n"
"//   [backbone1]--[glyco1]--[hbond1 | hbond2]--[glyco2]--[backbone2]\n"
"//       wd1        signal1    coupling           signal2      wd2\n"
"//\n"
"// Each thread reads one element from both strands, applies the rung\n"
"// geometry, runs Adam, writes back. Zero CPU involvement.\n"
"\n"
"kernel void dna_rung_paired(\n"
"    device float* d1     [[buffer(0)]],   // strand 1 weights\n"
"    device const float* g1 [[buffer(1)]], // strand 1 gradients\n"
"    device float* m1     [[buffer(2)]],   // strand 1 momentum\n"
"    device float* v1     [[buffer(3)]],   // strand 1 velocity\n"
"    device float* d2     [[buffer(4)]],   // strand 2 weights\n"
"    device const float* g2 [[buffer(5)]], // strand 2 gradients\n"
"    device float* m2     [[buffer(6)]],   // strand 2 momentum\n"
"    device float* v2     [[buffer(7)]],   // strand 2 velocity\n"
"    constant float& backbone1 [[buffer(8)]],\n"
"    constant float& glyco1    [[buffer(9)]],\n"
"    constant float& hbond1    [[buffer(10)]],\n"
"    constant float& hbond2    [[buffer(11)]],\n"
"    constant float& glyco2    [[buffer(12)]],\n"
"    constant float& backbone2 [[buffer(13)]],\n"
"    constant float& bondStr   [[buffer(14)]],\n"
"    constant float& lr        [[buffer(15)]],\n"
"    constant float& beta1     [[buffer(16)]],\n"
"    constant float& beta2     [[buffer(17)]],\n"
"    constant float& bc1       [[buffer(18)]],\n"
"    constant float& bc2       [[buffer(19)]],\n"
"    constant float& eps       [[buffer(20)]],\n"
"    constant float& wd        [[buffer(21)]],\n"
"    uint id [[thread_position_in_grid]])\n"
"{\n"
"    float ob1 = 1.0f - beta1;\n"
"    float ob2 = 1.0f - beta2;\n"
"\n"
"    // Read both strands\n"
"    float grad1 = g1[id];\n"
"    float grad2 = g2[id];\n"
"\n"
"    // 6-point rung: gradient flows through the base pair\n"
"    float signal1  = grad1 * glyco1;                  // glycosidic bond 1\n"
"    float crossMom = grad2 * hbond1 * bondStr;        // H-bond: strand2 -> strand1\n"
"    float crossVel = grad1 * hbond2 * bondStr;        // H-bond: strand1 -> strand2\n"
"    float signal2  = grad2 * glyco2;                  // glycosidic bond 2\n"
"\n"
"    // Strand 1: Adam with cross-strand coupling\n"
"    float eff1 = signal1 + crossMom;\n"
"    float mi1 = beta1 * m1[id] + ob1 * eff1;\n"
"    float vi1 = beta2 * v1[id] + ob2 * eff1 * eff1;\n"
"    m1[id] = mi1;\n"
"    v1[id] = vi1;\n"
"    d1[id] -= lr * (mi1 / bc1 / (sqrt(vi1 / bc2) + eps) + wd * backbone1 * d1[id]);\n"
"\n"
"    // Strand 2: Adam with cross-strand coupling\n"
"    float eff2 = signal2 + crossVel;\n"
"    float mi2 = beta1 * m2[id] + ob1 * eff2;\n"
"    float vi2 = beta2 * v2[id] + ob2 * eff2 * eff2;\n"
"    m2[id] = mi2;\n"
"    v2[id] = vi2;\n"
"    d2[id] -= lr * (mi2 / bc1 / (sqrt(vi2 / bc2) + eps) + wd * backbone2 * d2[id]);\n"
"}\n"
"\n"
"// Forward-only inline needle — no backward pass, momentum IS gradient.\n"
"// Fires BEFORE the matmul reads the weight. Updates dequant scratch in-place.\n"
"// signalScale: +1 = loss improved, -ratio = worsening.\n"
"// On Metal unified memory, mask is INT8 (0=frozen, 1=active). No compact indexing.\n"
"kernel void helix_needle_inline(\n"
"    device char* data_int8        [[buffer(0)]],  // [n] INT8 weights\n"
"    device float* scales          [[buffer(1)]],  // [nRows] per-row absmax scales\n"
"    device float* fp32_cache      [[buffer(2)]],  // [n] FP32 dequant scratch — updated in-place\n"
"    device half* mom              [[buffer(3)]],  // [n] FP16 momentum\n"
"    device float* delta           [[buffer(4)]],  // [n] FP32 delta residual\n"
"    device const char* mask       [[buffer(5)]],  // [n] 0=frozen, 1=active\n"
"    constant float& signalScale  [[buffer(6)]],  // +1 improving, -ratio worsening\n"
"    constant float& lr           [[buffer(7)]],\n"
"    constant float& beta1        [[buffer(8)]],\n"
"    constant float& wd           [[buffer(9)]],\n"
"    constant uint& cols          [[buffer(10)]],\n"
"    uint i [[thread_position_in_grid]])\n"
"{\n"
"    if (mask[i] == 0) return;\n"
"\n"
"    float mi = float(mom[i]);\n"
"    float d = delta[i];\n"
"\n"
"    // Synthetic gradient: momentum * signalScale\n"
"    float sg = mi * signalScale;\n"
"    mi = beta1 * mi + (1.0f - beta1) * sg;\n"
"    d -= lr * (mi + wd * d);\n"
"\n"
"    uint row = i / cols;\n"
"    float scale = scales[row] / 127.0f;\n"
"\n"
"    // Fold into INT8 when delta crosses half a quant bucket\n"
"    float bucket = scale;\n"
"    if (d > 0.5f * bucket || d < -0.5f * bucket) {\n"
"        float w = float(data_int8[i]) * scale + d;\n"
"        float qi = w / scale;\n"
"        qi = clamp(qi, -127.0f, 127.0f);\n"
"        char q = char(rint(qi));\n"
"        data_int8[i] = q;\n"
"        d = w - float(q) * scale;\n"
"        fp32_cache[i] = float(q) * scale + d;\n"
"    } else {\n"
"        fp32_cache[i] = float(data_int8[i]) * scale + d;\n"
"    }\n"
"\n"
"    mom[i] = half(mi);\n"
"    delta[i] = d;\n"
"}\n"
"\n"
"// Gradient L2 norm squared — parallel reduction via simd_sum.\n"
"// Each threadgroup reduces its chunk, atomically adds to output[0].\n"
"kernel void grad_norm_sq(\n"
"    device const float* grad [[buffer(0)]],\n"
"    device atomic_float* out [[buffer(1)]],\n"
"    uint id  [[thread_position_in_grid]],\n"
"    uint tid [[thread_index_in_threadgroup]],\n"
"    uint n   [[threads_per_grid]])\n"
"{\n"
"    float val = grad[id];\n"
"    float sq = val * val;\n"
"    sq = simd_sum(sq);\n"
"    if (tid % 32 == 0) {\n"
"        atomic_fetch_add_explicit(out, sq, memory_order_relaxed);\n"
"    }\n"
"}\n"
"\n"
"// Gradient clip scale: grad[i] *= min(1, maxNorm / sqrt(sumSq[0])).\n"
"kernel void grad_clip_scale(\n"
"    device float* grad            [[buffer(0)]],\n"
"    device const float* sumSq     [[buffer(1)]],\n"
"    constant float& maxNorm       [[buffer(2)]],\n"
"    uint id [[thread_position_in_grid]])\n"
"{\n"
"    float norm = sqrt(sumSq[0]);\n"
"    float scale = (norm > maxNorm) ? (maxNorm / norm) : 1.0f;\n"
"    grad[id] *= scale;\n"
"}\n"
"\n"
"// Memory copy — compute kernel for use within a fused encoder.\n"
"kernel void copy_mem(\n"
"    device const float* src [[buffer(0)]],\n"
"    device float* dst       [[buffer(1)]],\n"
"    uint id [[thread_position_in_grid]])\n"
"{\n"
"    dst[id] = src[id];\n"
"}\n"
"\n"
"// Debug: write 42.0 to all elements of an FP32 buffer.\n"
"kernel void debug_fill_42(\n"
"    device float* dst [[buffer(0)]],\n"
"    uint id [[thread_position_in_grid]])\n"
"{\n"
"    dst[id] = 42.0f;\n"
"}\n"
"\n"
"// ============================================================\n"
"// ============================================================\n"
"// Tiled GEMM: C[M,N] = A[M,K] @ B[N,K]^T (B transposed)\n"
"// ============================================================\n"
"// Cooperative threadgroup tiling. TILE=16 for Apple Silicon.\n"
"// Each threadgroup computes one 16×16 output tile.\n"
"// Shared memory holds tiles of A and B for data reuse.\n"
"#define TILE 32\n"
"\n"
"// Device function: tiled GEMM into a sub-region of output buffer.\n"
"// Called from within a kernel — not a standalone kernel.\n"
"// A[M,K] row-major, B[N,K] row-major (transposed access), C[M,N] row-major.\n"
"// Caller must ensure threadgroup size = (TILE, TILE).\n"
"inline void tiled_gemm_bt(\n"
"    device const float* A, device const float* B, device float* C,\n"
"    uint M, uint K, uint N,\n"
"    uint2 tid, uint2 gid,\n"
"    threadgroup float* sA, threadgroup float* sB)\n"
"{\n"
"    uint row = gid.y * TILE + tid.y;\n"
"    uint col = gid.x * TILE + tid.x;\n"
"    float acc = 0.0f;\n"
"\n"
"    for (uint t = 0; t < (K + TILE - 1) / TILE; t++) {\n"
"        uint aCol = t * TILE + tid.x;\n"
"        uint bCol = t * TILE + tid.y;\n"
"        // sA: tile of A, row-major. sA[r][c] = A[row, t*TILE+c]\n"
"        sA[tid.y * TILE + tid.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;\n"
"        // sB: tile of B^T, stored row-major. sB[c][k] = B[col, t*TILE+k]\n"
"        // tid.y iterates k-tiles, tid.x iterates columns\n"
"        sB[tid.y * TILE + tid.x] = (col < N && bCol < K) ? B[col * K + bCol] : 0.0f;\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"\n"
"        for (uint i = 0; i < TILE; i++) {\n"
"            // sA[row_local, i] * sB[i, col_local] — but sB is stored transposed\n"
"            // sB[i][col_local] = sB[i * TILE + tid.x] — NO, sB layout is [tid.y][tid.x]\n"
"            // where tid.y=k-index, tid.x=col-index.\n"
"            // So sB[k][col] = sB[k * TILE + col_local]. Access: sB[i * TILE + tid.x]\n"
"            acc += sA[tid.y * TILE + i] * sB[i * TILE + tid.x];\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (row < M && col < N) C[row * N + col] = acc;\n"
"}\n"
"\n"
"// ============================================================\n"
"// Fused per-layer forward transformer kernel\n"
"// ============================================================\n"
"// One threadgroup per TILE×TILE output region.\n"
"// Buffer layout: all INT8 weights packed flat, offset table in constants.\n"
"// Activation cache written to device memory for backward coil.\n"
"//\n"
"// Per-layer ops (all in-kernel, no CPU round-trip):\n"
"//   dequant INT8→FP32 → RMSNorm → QKV GEMM → bias → RoPE →\n"
"//   fused causal attention → WO GEMM → residual →\n"
"//   RMSNorm → gate+up GEMM → SiLU gate mul → down GEMM → residual\n"
"//\n"
"// NOTE: This kernel handles the GEMM portions. Attention (which requires\n"
"// cross-position dependencies) is dispatched as a separate fused kernel\n"
"// between the QKV and WO phases.\n"
"\n"
"// RMSNorm device function: in-place, one row at a time.\n"
"// weight[dim], x[dim] modified in-place. Returns 1/rms for backward.\n"
"inline float device_rmsnorm(device float* x, device const float* w, uint dim) {\n"
"    float ss = 0.0f;\n"
"    for (uint i = 0; i < dim; i++) ss += x[i] * x[i];\n"
"    float rms = 1.0f / sqrt(ss / float(dim) + 1e-6f);\n"
"    for (uint i = 0; i < dim; i++) x[i] = x[i] * rms * w[i];\n"
"    return rms;\n"
"}\n"
"\n"
"// RoPE device function: rotate pairs in-place.\n"
"inline void device_rope(device float* x, uint pos, uint headDim, uint nHeads, float theta) {\n"
"    for (uint h = 0; h < nHeads; h++) {\n"
"        for (uint i = 0; i < headDim; i += 2) {\n"
"            float freq = 1.0f / pow(theta, float(i) / float(headDim));\n"
"            float angle = float(pos) * freq;\n"
"            float cosA = cos(angle), sinA = sin(angle);\n"
"            uint off = h * headDim + i;\n"
"            float x0 = x[off], x1 = x[off + 1];\n"
"            x[off]     = x0 * cosA - x1 * sinA;\n"
"            x[off + 1] = x0 * sinA + x1 * cosA;\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"// Fused causal attention: one threadgroup per head.\n"
"// Q[n,headDim], K[n,headDim], V[n,headDim] → out[n,headDim]\n"
"// Online safe softmax — no materialized score matrix.\n"
"kernel void fused_causal_attention(\n"
"    device const float* Q    [[buffer(0)]],  // [n, dim] (strided by dim, head offset externally)\n"
"    device const float* K    [[buffer(1)]],  // [n, kvDim]\n"
"    device const float* V    [[buffer(2)]],  // [n, kvDim]\n"
"    device float* out        [[buffer(3)]],  // [n, dim]\n"
"    device float* scores_out [[buffer(4)]],  // [nHeads, n, n] — saved for backward\n"
"    constant uint& dim      [[buffer(5)]],\n"
"    constant uint& kvDim    [[buffer(6)]],\n"
"    constant uint& headDim  [[buffer(7)]],\n"
"    constant uint& nHeads   [[buffer(8)]],\n"
"    constant uint& nKVHeads [[buffer(9)]],\n"
"    constant uint& seqLen   [[buffer(10)]],\n"
"    uint h [[threadgroup_position_in_grid]],\n"
"    uint tid [[thread_index_in_threadgroup]])\n"
"{\n"
"    if (h >= nHeads) return;\n"
"    uint kvH = h / (nHeads / nKVHeads);\n"
"    float scale = 1.0f / sqrt(float(headDim));\n"
"    uint n = seqLen;\n"
"\n"
"    // One thread per position (tid = position index)\n"
"    if (tid >= n) return;\n"
"    uint i = tid;\n"
"\n"
"    // Score, softmax, accumulate for position i\n"
"    float maxScore = -1e30f;\n"
"    // Pass 1: compute scores and find max\n"
"    for (uint j = 0; j <= i; j++) {\n"
"        float dot = 0.0f;\n"
"        for (uint d = 0; d < headDim; d++) {\n"
"            dot += Q[i * dim + h * headDim + d] * K[j * kvDim + kvH * headDim + d];\n"
"        }\n"
"        dot *= scale;\n"
"        scores_out[h * n * n + i * n + j] = dot;\n"
"        maxScore = max(maxScore, dot);\n"
"    }\n"
"\n"
"    // Pass 2: exp and sum\n"
"    float sumExp = 0.0f;\n"
"    for (uint j = 0; j <= i; j++) {\n"
"        float s = exp(scores_out[h * n * n + i * n + j] - maxScore);\n"
"        scores_out[h * n * n + i * n + j] = s;\n"
"        sumExp += s;\n"
"    }\n"
"\n"
"    // Normalize + accumulate output\n"
"    float invSum = 1.0f / sumExp;\n"
"    for (uint j = 0; j <= i; j++) {\n"
"        scores_out[h * n * n + i * n + j] *= invSum;\n"
"    }\n"
"\n"
"    // Output: weighted sum of V\n"
"    for (uint d = 0; d < headDim; d++) {\n"
"        float acc = 0.0f;\n"
"        for (uint j = 0; j <= i; j++) {\n"
"            acc += scores_out[h * n * n + i * n + j] * V[j * kvDim + kvH * headDim + d];\n"
"        }\n"
"        out[i * dim + h * headDim + d] = acc;\n"
"    }\n"
"}\n"
"\n"
"// Fused causal attention backward: one thread per position per head.\n"
"// Consumes saved scores, produces dQ, dK, dV.\n"
"kernel void fused_causal_attention_backward(\n"
"    device const float* dOut    [[buffer(0)]],  // [n, dim]\n"
"    device const float* Q       [[buffer(1)]],\n"
"    device const float* K       [[buffer(2)]],\n"
"    device const float* V       [[buffer(3)]],\n"
"    device const float* scores  [[buffer(4)]],  // [nHeads, n, n] softmax output\n"
"    device float* dQ            [[buffer(5)]],   // [n, dim] output\n"
"    device float* dK            [[buffer(6)]],   // [n, kvDim] output (atomic add for GQA)\n"
"    device float* dV            [[buffer(7)]],   // [n, kvDim] output (atomic add for GQA)\n"
"    constant uint& dim         [[buffer(8)]],\n"
"    constant uint& kvDim      [[buffer(9)]],\n"
"    constant uint& headDim    [[buffer(10)]],\n"
"    constant uint& nHeads     [[buffer(11)]],\n"
"    constant uint& nKVHeads   [[buffer(12)]],\n"
"    constant uint& seqLen     [[buffer(13)]],\n"
"    uint h [[threadgroup_position_in_grid]],\n"
"    uint tid [[thread_index_in_threadgroup]])\n"
"{\n"
"    if (h >= nHeads) return;\n"
"    uint kvH = h / (nHeads / nKVHeads);\n"
"    float scale = 1.0f / sqrt(float(headDim));\n"
"    uint n = seqLen;\n"
"    if (tid >= n) return;\n"
"    uint i = tid;\n"
"\n"
"    // For position i, head h:\n"
"    // dScores[j] = softmax_backward(dOut_i @ V_j^T, scores_i)\n"
"    // Then dQ_i += dScores @ K, dK_j += dScores^T @ Q_i, dV_j += scores^T @ dOut_i\n"
"\n"
"    // Compute dot products: dOut_i dot V_j for each j <= i\n"
"    float dotSum = 0.0f;\n"
"    for (uint j = 0; j <= i; j++) {\n"
"        float dv = 0.0f;\n"
"        for (uint d = 0; d < headDim; d++) {\n"
"            dv += dOut[i * dim + h * headDim + d] * V[j * kvDim + kvH * headDim + d];\n"
"        }\n"
"        dotSum += scores[h * n * n + i * n + j] * dv;\n"
"    }\n"
"\n"
"    // Softmax backward + accumulate dQ, dK, dV\n"
"    for (uint j = 0; j <= i; j++) {\n"
"        float dv = 0.0f;\n"
"        for (uint d = 0; d < headDim; d++) {\n"
"            dv += dOut[i * dim + h * headDim + d] * V[j * kvDim + kvH * headDim + d];\n"
"        }\n"
"        float ds = scores[h * n * n + i * n + j] * (dv - dotSum) * scale;\n"
"\n"
"        // dQ[i] += ds * K[j]\n"
"        for (uint d = 0; d < headDim; d++) {\n"
"            dQ[i * dim + h * headDim + d] += ds * K[j * kvDim + kvH * headDim + d];\n"
"        }\n"
"        // dK[j] += ds * Q[i] (atomic for GQA — multiple heads share same KV head)\n"
"        for (uint d = 0; d < headDim; d++) {\n"
"            atomic_fetch_add_explicit(\n"
"                (device atomic_float*)&dK[j * kvDim + kvH * headDim + d],\n"
"                ds * Q[i * dim + h * headDim + d],\n"
"                memory_order_relaxed);\n"
"        }\n"
"        // dV[j] += scores[i,j] * dOut[i] (atomic for GQA)\n"
"        float s = scores[h * n * n + i * n + j];\n"
"        for (uint d = 0; d < headDim; d++) {\n"
"            atomic_fetch_add_explicit(\n"
"                (device atomic_float*)&dV[j * kvDim + kvH * headDim + d],\n"
"                s * dOut[i * dim + h * headDim + d],\n"
"                memory_order_relaxed);\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"// Bias add: x[i] += bias[i % cols] for all positions\n"
"kernel void bias_add(\n"
"    device float* x        [[buffer(0)]],\n"
"    device const float* b  [[buffer(1)]],\n"
"    constant uint& cols   [[buffer(2)]],\n"
"    uint i [[thread_position_in_grid]])\n"
"{\n"
"    x[i] += b[i % cols];\n"
"}\n"
"\n"
"// RoPE forward: rotate pairs in Q and K.\n"
"// One thread per (position, pair). dim_pairs = headDim/2 * nHeads.\n"
"kernel void rope_forward(\n"
"    device float* x        [[buffer(0)]],  // [n, dim] (Q or K)\n"
"    constant uint& headDim [[buffer(1)]],\n"
"    constant uint& nHeads  [[buffer(2)]],\n"
"    constant float& theta  [[buffer(3)]],\n"
"    constant uint& stride  [[buffer(4)]],  // dim for Q, kvDim for K\n"
"    uint2 gid [[thread_position_in_grid]])  // (pair_idx, position)\n"
"{\n"
"    uint pos = gid.y;\n"
"    uint pairIdx = gid.x;  // which pair across all heads\n"
"    uint h = pairIdx / (headDim / 2);\n"
"    uint i = (pairIdx % (headDim / 2)) * 2;\n"
"    if (h >= nHeads) return;\n"
"\n"
"    float freq = 1.0f / pow(theta, float(i) / float(headDim));\n"
"    float angle = float(pos) * freq;\n"
"    float cosA = cos(angle), sinA = sin(angle);\n"
"\n"
"    uint off = pos * stride + h * headDim + i;\n"
"    float x0 = x[off], x1 = x[off + 1];\n"
"    x[off]     = x0 * cosA - x1 * sinA;\n"
"    x[off + 1] = x0 * sinA + x1 * cosA;\n"
"}\n"
"\n"
"// RoPE backward: same rotation but negate sinA.\n"
"kernel void rope_backward(\n"
"    device float* dx       [[buffer(0)]],\n"
"    constant uint& headDim [[buffer(1)]],\n"
"    constant uint& nHeads  [[buffer(2)]],\n"
"    constant float& theta  [[buffer(3)]],\n"
"    constant uint& stride  [[buffer(4)]],\n"
"    uint2 gid [[thread_position_in_grid]])\n"
"{\n"
"    uint pos = gid.y;\n"
"    uint pairIdx = gid.x;\n"
"    uint h = pairIdx / (headDim / 2);\n"
"    uint i = (pairIdx % (headDim / 2)) * 2;\n"
"    if (h >= nHeads) return;\n"
"\n"
"    float freq = 1.0f / pow(theta, float(i) / float(headDim));\n"
"    float angle = float(pos) * freq;\n"
"    float cosA = cos(angle), sinA = -sin(angle);\n"
"\n"
"    uint off = pos * stride + h * headDim + i;\n"
"    float x0 = dx[off], x1 = dx[off + 1];\n"
"    dx[off]     = x0 * cosA - x1 * sinA;\n"
"    dx[off + 1] = x0 * sinA + x1 * cosA;\n"
"}\n"
"\n"
"// GEMM kernel: C = A @ B^T. A[M,K], B[N,K] → C[M,N].\n"
"// Tiled cooperative: threadgroup=(TILE,TILE), grid=(N/TILE, M/TILE).\n"
"kernel void gemm_bt(\n"
"    device const float* A [[buffer(0)]],\n"
"    device const float* B [[buffer(1)]],\n"
"    device float* C       [[buffer(2)]],\n"
"    constant uint& M     [[buffer(3)]],\n"
"    constant uint& K     [[buffer(4)]],\n"
"    constant uint& N     [[buffer(5)]],\n"
"    uint2 tid [[thread_position_in_threadgroup]],\n"
"    uint2 gid [[threadgroup_position_in_grid]])\n"
"{\n"
"    threadgroup float sA[TILE * TILE];\n"
"    threadgroup float sB[TILE * TILE];\n"
"    tiled_gemm_bt(A, B, C, M, K, N, tid, gid, sA, sB);\n"
"}\n"
"\n"
"// GEMM accumulate: C += A @ B^T.\n"
"kernel void gemm_bt_acc(\n"
"    device const float* A [[buffer(0)]],\n"
"    device const float* B [[buffer(1)]],\n"
"    device float* C       [[buffer(2)]],\n"
"    constant uint& M     [[buffer(3)]],\n"
"    constant uint& K     [[buffer(4)]],\n"
"    constant uint& N     [[buffer(5)]],\n"
"    uint2 tid [[thread_position_in_threadgroup]],\n"
"    uint2 gid [[threadgroup_position_in_grid]])\n"
"{\n"
"    threadgroup float sA[TILE * TILE];\n"
"    threadgroup float sB[TILE * TILE];\n"
"\n"
"    uint row = gid.y * TILE + tid.y;\n"
"    uint col = gid.x * TILE + tid.x;\n"
"    float acc = 0.0f;\n"
"\n"
"    for (uint t = 0; t < (K + TILE - 1) / TILE; t++) {\n"
"        uint aCol = t * TILE + tid.x;\n"
"        uint bCol = t * TILE + tid.y;\n"
"        sA[tid.y * TILE + tid.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;\n"
"        sB[tid.y * TILE + tid.x] = (col < N && bCol < K) ? B[col * K + bCol] : 0.0f;\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        for (uint i = 0; i < TILE; i++) acc += sA[tid.y * TILE + i] * sB[i * TILE + tid.x];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (row < M && col < N) C[row * N + col] += acc;\n"
"}\n"
"\n"
"// GEMM: C = A^T @ B. A[K,M], B[K,N] → C[M,N]. For weight gradients.\n"
"kernel void gemm_tn(\n"
"    device const float* A [[buffer(0)]],\n"
"    device const float* B [[buffer(1)]],\n"
"    device float* C       [[buffer(2)]],\n"
"    constant uint& M     [[buffer(3)]],\n"
"    constant uint& K     [[buffer(4)]],\n"
"    constant uint& N     [[buffer(5)]],\n"
"    uint2 tid [[thread_position_in_threadgroup]],\n"
"    uint2 gid [[threadgroup_position_in_grid]])\n"
"{\n"
"    threadgroup float sA[TILE * TILE];\n"
"    threadgroup float sB[TILE * TILE];\n"
"\n"
"    uint row = gid.y * TILE + tid.y;  // output row (M dimension)\n"
"    uint col = gid.x * TILE + tid.x;  // output col (N dimension)\n"
"    float acc = 0.0f;\n"
"\n"
"    for (uint t = 0; t < (K + TILE - 1) / TILE; t++) {\n"
"        uint aRow = t * TILE + tid.x;  // K dimension\n"
"        uint bRow = t * TILE + tid.y;  // K dimension\n"
"        // A is [K,M], transposed access: A^T[row, aRow] = A[aRow, row]\n"
"        sA[tid.y * TILE + tid.x] = (row < M && aRow < K) ? A[aRow * M + row] : 0.0f;\n"
"        sB[tid.y * TILE + tid.x] = (col < N && bRow < K) ? B[bRow * N + col] : 0.0f;\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        for (uint i = 0; i < TILE; i++) acc += sA[tid.y * TILE + i] * sB[i * TILE + tid.x];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (row < M && col < N) C[row * N + col] = acc;\n"
"}\n"
"\n"
"// GEMM: C = A @ B (no transpose). A[M,K], B[K,N] → C[M,N].\n"
"kernel void gemm_nn(\n"
"    device const float* A [[buffer(0)]],\n"
"    device const float* B [[buffer(1)]],\n"
"    device float* C       [[buffer(2)]],\n"
"    constant uint& M     [[buffer(3)]],\n"
"    constant uint& K     [[buffer(4)]],\n"
"    constant uint& N     [[buffer(5)]],\n"
"    uint2 tid [[thread_position_in_threadgroup]],\n"
"    uint2 gid [[threadgroup_position_in_grid]])\n"
"{\n"
"    threadgroup float sA[TILE * TILE];\n"
"    threadgroup float sB[TILE * TILE];\n"
"\n"
"    uint row = gid.y * TILE + tid.y;\n"
"    uint col = gid.x * TILE + tid.x;\n"
"    float acc = 0.0f;\n"
"\n"
"    for (uint t = 0; t < (K + TILE - 1) / TILE; t++) {\n"
"        uint aCol = t * TILE + tid.x;\n"
"        uint bRow = t * TILE + tid.y;\n"
"        sA[tid.y * TILE + tid.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;\n"
"        sB[tid.y * TILE + tid.x] = (col < N && bRow < K) ? B[bRow * N + col] : 0.0f;\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        for (uint i = 0; i < TILE; i++) acc += sA[tid.y * TILE + i] * sB[i * TILE + tid.x];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (row < M && col < N) C[row * N + col] = acc;\n"
"}\n"
"\n"
"// RMSNorm with scale output for backward. One threadgroup per position.\n"
"// Writes normed values in-place and scale[pos] = 1/rms for backward.\n"
"kernel void rmsnorm_save(\n"
"    device float* x         [[buffer(0)]],   // [n, dim] modified in-place\n"
"    device const float* w   [[buffer(1)]],   // [dim] weights\n"
"    device float* scale     [[buffer(2)]],   // [n] output: 1/rms per position\n"
"    constant uint& dim     [[buffer(3)]],\n"
"    constant float& eps    [[buffer(4)]],\n"
"    uint pos [[threadgroup_position_in_grid]],\n"
"    uint tid [[thread_index_in_threadgroup]],\n"
"    uint tpg [[threads_per_threadgroup]])\n"
"{\n"
"    uint off = pos * dim;\n"
"    // Cooperative sum of squares\n"
"    float localSS = 0.0f;\n"
"    for (uint i = tid; i < dim; i += tpg) localSS += x[off + i] * x[off + i];\n"
"    localSS = simd_sum(localSS);\n"
"    threadgroup float shared_ss[32];\n"
"    uint lane = tid % 32, warp = tid / 32;\n"
"    if (lane == 0) shared_ss[warp] = localSS;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (warp == 0) {\n"
"        localSS = (lane < (tpg + 31) / 32) ? shared_ss[lane] : 0.0f;\n"
"        localSS = simd_sum(localSS);\n"
"    }\n"
"    threadgroup float finalSS;\n"
"    if (tid == 0) finalSS = localSS;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"\n"
"    float rms = 1.0f / sqrt(finalSS / float(dim) + eps);\n"
"    if (tid == 0) scale[pos] = rms;\n"
"\n"
"    for (uint i = tid; i < dim; i += tpg) {\n"
"        x[off + i] = x[off + i] * rms * w[i];\n"
"    }\n"
"}\n"
"\n"
"// RMSNorm backward: dx = rmsScale * w * dOut - dot * xIn\n"
"// where dot = sum(dOut * w * xIn) * rmsScale^3 / dim.\n"
"// One threadgroup per position. Cooperative reduction for dot product.\n"
"kernel void rmsnorm_backward(\n"
"    device const float* dOut   [[buffer(0)]],  // [n, dim] upstream gradient\n"
"    device const float* xIn    [[buffer(1)]],  // [n, dim] saved input\n"
"    device const float* w      [[buffer(2)]],  // [dim] norm weights\n"
"    device const float* scale  [[buffer(3)]],  // [n] saved 1/rms per position\n"
"    device float* dx           [[buffer(4)]],  // [n, dim] output gradient\n"
"    constant uint& dim        [[buffer(5)]],\n"
"    uint pos [[threadgroup_position_in_grid]],\n"
"    uint tid [[thread_index_in_threadgroup]],\n"
"    uint tpg [[threads_per_threadgroup]])\n"
"{\n"
"    uint off = pos * dim;\n"
"    float rms = scale[pos];\n"
"\n"
"    // Pass 1: cooperative dot product sum(dOut * w * xIn)\n"
"    float localDot = 0.0f;\n"
"    for (uint i = tid; i < dim; i += tpg) {\n"
"        localDot += dOut[off + i] * w[i] * xIn[off + i];\n"
"    }\n"
"    localDot = simd_sum(localDot);\n"
"    threadgroup float shared_dot[32];\n"
"    uint lane = tid % 32, warp = tid / 32;\n"
"    if (lane == 0) shared_dot[warp] = localDot;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (warp == 0) {\n"
"        localDot = (lane < (tpg + 31) / 32) ? shared_dot[lane] : 0.0f;\n"
"        localDot = simd_sum(localDot);\n"
"    }\n"
"    threadgroup float finalDot;\n"
"    if (tid == 0) finalDot = localDot;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"\n"
"    float coeff = finalDot * rms * rms * rms / float(dim);\n"
"\n"
"    // Pass 2: compute dx\n"
"    for (uint i = tid; i < dim; i += tpg) {\n"
"        dx[off + i] = rms * w[i] * dOut[off + i] - coeff * xIn[off + i];\n"
"    }\n"
"}\n"
"\n"
"// Dequant INT8 → FP32 for graph consumption.\n"
"// One thread per element. Reads INT8 weight + per-row scale, writes FP32.\n"
"kernel void dequant_int8_fp32(\n"
"    device const char* src      [[buffer(0)]],  // [n] INT8 weights\n"
"    device const float* scales  [[buffer(1)]],  // [nRows] per-row scales\n"
"    device float* dst           [[buffer(2)]],  // [n] FP32 output\n"
"    constant uint& cols        [[buffer(3)]],\n"
"    uint i [[thread_position_in_grid]])\n"
"{\n"
"    uint row = i / cols;\n"
"    float scale = scales[row] / 127.0f;\n"
"    dst[i] = float(src[i]) * scale;\n"
"}\n"
"\n"
"// Sparse dequant: only update rows where mask[row] != 0.\n"
"kernel void dequant_int8_delta_sparse(\n"
"    device const char* src      [[buffer(0)]],\n"
"    device const float* scales  [[buffer(1)]],\n"
"    device const float* delta   [[buffer(2)]],\n"
"    device float* dst           [[buffer(3)]],\n"
"    device const char* mask     [[buffer(4)]],\n"
"    constant uint& cols        [[buffer(5)]],\n"
"    uint i [[thread_position_in_grid]])\n"
"{\n"
"    uint row = i / cols;\n"
"    if (mask[row] == 0) return;\n"
"    float scale = scales[row] / 127.0f;\n"
"    dst[i] = float(src[i]) * scale + delta[i];\n"
"}\n"
"\n"
"// Dequant INT8 → FP32 with delta residual.\n"
"// effective_weight = int8 * scale + delta\n"
"// The delta accumulates sub-quant-level precision from the optimizer.\n"
"kernel void dequant_int8_delta(\n"
"    device const char* src      [[buffer(0)]],\n"
"    device const float* scales  [[buffer(1)]],\n"
"    device const float* delta   [[buffer(2)]],  // [n] FP32 delta residual\n"
"    device float* dst           [[buffer(3)]],\n"
"    constant uint& cols        [[buffer(4)]],\n"
"    uint i [[thread_position_in_grid]])\n"
"{\n"
"    uint row = i / cols;\n"
"    float scale = scales[row] / 127.0f;\n"
"    dst[i] = float(src[i]) * scale + delta[i];\n"
"}\n"
"\n"
"// Dequant INT8 → FP16 for Metal 4 matmul2d.\n"
"kernel void dequant_int8_fp16(\n"
"    device const char* src      [[buffer(0)]],\n"
"    device const float* scales  [[buffer(1)]],\n"
"    device half* dst            [[buffer(2)]],\n"
"    constant uint& cols        [[buffer(3)]],\n"
"    uint i [[thread_position_in_grid]])\n"
"{\n"
"    uint row = i / cols;\n"
"    float scale = scales[row] / 127.0f;\n"
"    dst[i] = half(float(src[i]) * scale);\n"
"}\n"
"\n"
"// FP32 → FP16 narrowing conversion.\n"
"kernel void fp32_to_fp16(\n"
"    device const float* src [[buffer(0)]],\n"
"    device half* dst        [[buffer(1)]],\n"
"    uint i [[thread_position_in_grid]])\n"
"{\n"
"    dst[i] = half(src[i]);\n"
"}\n"
"\n"
"// FP16 → FP32 widening conversion.\n"
"kernel void fp16_to_fp32(\n"
"    device const half* src [[buffer(0)]],\n"
"    device float* dst      [[buffer(1)]],\n"
"    uint i [[thread_position_in_grid]])\n"
"{\n"
"    dst[i] = float(src[i]);\n"
"}\n"
"\n"
"// Helix Needle — fused INT8 dequant → Adam → requant\n"
"// ============================================================\n"
"// Kill LoRA. Train actual INT8 weights directly.\n"
"// INT8 weight → dequant to FP32 in register → Adam/Helix update\n"
"// → requant FP32 → INT8. Momentum/velocity in FP16.\n"
"// Sparse mask from Conductor: 0=frozen, 1=active.\n"
"//\n"
"// Metal note: no native half on older devices, but Apple Silicon\n"
"// has full FP16 support. We use half for momentum/velocity.\n"
"\n"
"kernel void helix_needle(\n"
"    device char* data_int8        [[buffer(0)]],  // [n] INT8 weights\n"
"    device float* scales          [[buffer(1)]],  // [nRows] per-row absmax scales\n"
"    device const float* grad      [[buffer(2)]],  // [n] FP32 gradient\n"
"    device half* mom              [[buffer(3)]],  // [n] FP16 momentum\n"
"    device half* vel              [[buffer(4)]],  // [n] FP16 velocity\n"
"    device const char* mask       [[buffer(5)]],  // [n] 0=frozen, 1=active\n"
"    device float* delta           [[buffer(6)]],  // [n] FP32 delta residual\n"
"    constant float& lr           [[buffer(7)]],\n"
"    constant float& beta1        [[buffer(8)]],\n"
"    constant float& beta2        [[buffer(9)]],\n"
"    constant float& bc1          [[buffer(10)]],\n"
"    constant float& bc2          [[buffer(11)]],\n"
"    constant float& eps          [[buffer(12)]],\n"
"    constant float& wd           [[buffer(13)]],\n"
"    constant uint& cols          [[buffer(14)]],\n"
"    uint i [[thread_position_in_grid]])\n"
"{\n"
"    uint row = i / cols;\n"
"    if (mask[row] == 0) return;\n"
"\n"
"    float scale = scales[row] / 127.0f;\n"
"\n"
"    // Effective weight = INT8 * scale + delta\n"
"    float w = float(data_int8[i]) * scale + delta[i];\n"
"\n"
"    // Read momentum/velocity from FP16\n"
"    float mi = float(mom[i]);\n"
"    float vi = float(vel[i]);\n"
"\n"
"    // Adam update\n"
"    float g = grad[i];\n"
"    float ob1 = 1.0f - beta1, ob2 = 1.0f - beta2;\n"
"    mi = beta1 * mi + ob1 * g;\n"
"    vi = beta2 * vi + ob2 * g * g;\n"
"    float mhat = mi / bc1;\n"
"    float vhat = vi / bc2;\n"
"    w -= lr * (mhat / (sqrt(vhat) + eps) + wd * w);\n"
"\n"
"    // Write back FP16 momentum/velocity\n"
"    mom[i] = half(mi);\n"
"    vel[i] = half(vi);\n"
"\n"
"    // Delta residual: compute what INT8 can represent vs what we want\n"
"    float inv_scale = 127.0f / (scales[row] + 1e-10f);\n"
"    float qi = w * inv_scale;\n"
"    qi = clamp(qi, -127.0f, 127.0f);\n"
"    char q_int8 = char(rint(qi));\n"
"    float w_quantized = float(q_int8) * scale;\n"
"    delta[i] = w - w_quantized;  // sub-quant-level precision lives here\n"
"    data_int8[i] = q_int8;\n"
"\n"
"    // Scale update deferred — delta residual handles sub-quant precision.\n"
"    // Periodic scale refresh done on CPU between steps if needed.\n"
"}\n"
"\n"
"// Paired-strand needle: DNA rung coupling for wq↔wk (AT), gate↔up (GC).\n"
"// Both strands updated in one kernel — cross-gradient coupling through H-bonds.\n"
"kernel void helix_needle_paired(\n"
"    device char* d1_int8          [[buffer(0)]],\n"
"    device char* d2_int8          [[buffer(1)]],\n"
"    device float* s1              [[buffer(2)]],\n"
"    device float* s2              [[buffer(3)]],\n"
"    device const float* g1        [[buffer(4)]],\n"
"    device const float* g2        [[buffer(5)]],\n"
"    device half* m1               [[buffer(6)]],\n"
"    device half* m2               [[buffer(7)]],\n"
"    device half* v1               [[buffer(8)]],\n"
"    device half* v2               [[buffer(9)]],\n"
"    device const char* mask       [[buffer(10)]],\n"
"    device float* delta1          [[buffer(11)]],  // FP32 delta residual strand 1\n"
"    device float* delta2          [[buffer(12)]],  // FP32 delta residual strand 2\n"
"    constant float& lr           [[buffer(13)]],\n"
"    constant float& beta1        [[buffer(14)]],\n"
"    constant float& beta2        [[buffer(15)]],\n"
"    constant float& bc1          [[buffer(16)]],\n"
"    constant float& bc2          [[buffer(17)]],\n"
"    constant float& eps          [[buffer(18)]],\n"
"    constant float& wd           [[buffer(19)]],\n"
"    constant float& backbone1    [[buffer(20)]],\n"
"    constant float& glyco1       [[buffer(21)]],\n"
"    constant float& hbond1       [[buffer(22)]],\n"
"    constant float& hbond2       [[buffer(23)]],\n"
"    constant float& glyco2       [[buffer(24)]],\n"
"    constant float& backbone2    [[buffer(25)]],\n"
"    constant float& bondStrength [[buffer(26)]],\n"
"    constant uint& cols          [[buffer(27)]],\n"
"    uint i [[thread_position_in_grid]])\n"
"{\n"
"    uint row = i / cols;\n"
"    if (mask[row] == 0) return;\n"
"\n"
"    float scale1 = s1[row] / 127.0f;\n"
"    float scale2 = s2[row] / 127.0f;\n"
"\n"
"    // Effective weight = INT8 * scale + delta\n"
"    float w1 = float(d1_int8[i]) * scale1 + delta1[i];\n"
"    float w2 = float(d2_int8[i]) * scale2 + delta2[i];\n"
"\n"
"    float mi1 = float(m1[i]), vi1 = float(v1[i]);\n"
"    float mi2 = float(m2[i]), vi2 = float(v2[i]);\n"
"\n"
"    float ob1 = 1.0f - beta1, ob2 = 1.0f - beta2;\n"
"\n"
"    // Strand 1: signal + cross-coupling from strand 2\n"
"    float signal1 = g1[i] * glyco1;\n"
"    float crossMom = g2[i] * hbond1 * bondStrength;\n"
"    float effGrad1 = signal1 + crossMom;\n"
"    mi1 = beta1 * mi1 + ob1 * effGrad1;\n"
"    vi1 = beta2 * vi1 + ob2 * effGrad1 * effGrad1;\n"
"    w1 -= lr * (mi1 / bc1 / (sqrt(vi1 / bc2) + eps) + wd * backbone1 * w1);\n"
"\n"
"    // Strand 2: signal + cross-coupling from strand 1\n"
"    float signal2 = g2[i] * glyco2;\n"
"    float crossVel = g1[i] * hbond2 * bondStrength;\n"
"    float effGrad2 = signal2 + crossVel;\n"
"    mi2 = beta1 * mi2 + ob1 * effGrad2;\n"
"    vi2 = beta2 * vi2 + ob2 * effGrad2 * effGrad2;\n"
"    w2 -= lr * (mi2 / bc1 / (sqrt(vi2 / bc2) + eps) + wd * backbone2 * w2);\n"
"\n"
"    m1[i] = half(mi1); v1[i] = half(vi1);\n"
"    m2[i] = half(mi2); v2[i] = half(vi2);\n"
"\n"
"    // Delta residual + requant for both strands\n"
"    float inv1 = 127.0f / (s1[row] + 1e-10f);\n"
"    float inv2 = 127.0f / (s2[row] + 1e-10f);\n"
"    float q1 = clamp(w1 * inv1, -127.0f, 127.0f);\n"
"    float q2 = clamp(w2 * inv2, -127.0f, 127.0f);\n"
"    char qi1 = char(rint(q1)), qi2 = char(rint(q2));\n"
"    delta1[i] = w1 - float(qi1) * scale1;\n"
"    delta2[i] = w2 - float(qi2) * scale2;\n"
"    d1_int8[i] = qi1;\n"
"    d2_int8[i] = qi2;\n"
"    // Scale update deferred — delta handles sub-quant precision.\n"
"}\n"
;

// Compute pipeline state objects — lazily initialized
static id<MTLLibrary> g_compute_lib = nil;
static id<MTLLibrary> g_metal4_lib = nil;  // pre-compiled Metal 4 matmul2d kernels
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
static id<MTLComputePipelineState> g_ps_dna_rung = nil;
static id<MTLComputePipelineState> g_ps_grad_norm = nil;
static id<MTLComputePipelineState> g_ps_grad_clip = nil;
static id<MTLComputePipelineState> g_ps_copy_mem = nil;
static id<MTLComputePipelineState> g_ps_lm_pass1 = nil;
static id<MTLComputePipelineState> g_ps_lm_pass2 = nil;
static id<MTLComputePipelineState> g_ps_lm_sparse1 = nil;
static id<MTLComputePipelineState> g_ps_lm_sparse2 = nil;
static id<MTLComputePipelineState> g_ps_needle = nil;
static id<MTLComputePipelineState> g_ps_needle_paired = nil;
static id<MTLComputePipelineState> g_ps_dequant_int8 = nil;
static id<MTLComputePipelineState> g_ps_dequant_delta = nil;
static id<MTLComputePipelineState> g_ps_dequant_delta_sparse = nil;
static id<MTLComputePipelineState> g_ps_needle_inline = nil;
static id<MTLComputePipelineState> g_ps_ce_loss = nil;
static id<MTLComputePipelineState> g_ps_rmsnorm_bwd = nil;
static id<MTLComputePipelineState> g_ps_dequant_fp16 = nil;
static id<MTLComputePipelineState> g_ps_fp32_to_fp16 = nil;
static id<MTLComputePipelineState> g_ps_fp16_to_fp32 = nil;
static id<MTLComputePipelineState> g_ps_gemm_bt = nil;
static id<MTLComputePipelineState> g_ps_gemm_bt_acc = nil;
static id<MTLComputePipelineState> g_ps_gemm_tn = nil;
static id<MTLComputePipelineState> g_ps_gemm_nn = nil;
// Metal 4 matmul2d replacements (loaded from .metallib if available)
static id<MTLComputePipelineState> g_ps_gemm4_bt = nil;
static id<MTLComputePipelineState> g_ps_gemm4_nn = nil;
static id<MTLComputePipelineState> g_ps_gemm4_tn = nil;
// FP32 cooperative variants — no FP16 conversion needed
static id<MTLComputePipelineState> g_ps_gemm4f_bt = nil;
static id<MTLComputePipelineState> g_ps_gemm4f_nn = nil;
static id<MTLComputePipelineState> g_ps_gemm4f_tn = nil;
static bool g_use_metal4_gemm = false;
static id<MTLComputePipelineState> g_ps_rmsnorm_save = nil;
static id<MTLComputePipelineState> g_ps_fused_attn = nil;
static id<MTLComputePipelineState> g_ps_fused_attn_bwd = nil;
static id<MTLComputePipelineState> g_ps_bias_add = nil;
static id<MTLComputePipelineState> g_ps_rope_fwd = nil;
static id<MTLComputePipelineState> g_ps_rope_bwd = nil;

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
    g_ps_dna_rung = make_ps(@"dna_rung_paired");
    g_ps_grad_norm = make_ps(@"grad_norm_sq");
    g_ps_grad_clip = make_ps(@"grad_clip_scale");
    g_ps_copy_mem = make_ps(@"copy_mem");
    g_ps_lm_pass1 = make_ps(@"lm_head_pass1");
    g_ps_lm_pass2 = make_ps(@"lm_head_pass2");
    g_ps_lm_sparse1 = make_ps(@"lm_head_sparse_pass1");
    g_ps_lm_sparse2 = make_ps(@"lm_head_sparse_pass2");
    g_ps_needle = make_ps(@"helix_needle");
    g_ps_needle_paired = make_ps(@"helix_needle_paired");
    g_ps_dequant_int8 = make_ps(@"dequant_int8_fp32");
    g_ps_dequant_delta = make_ps(@"dequant_int8_delta");
    g_ps_dequant_delta_sparse = make_ps(@"dequant_int8_delta_sparse");
    g_ps_needle_inline = make_ps(@"helix_needle_inline");
    g_ps_ce_loss = make_ps(@"ce_loss");
    g_ps_rmsnorm_bwd = make_ps(@"rmsnorm_backward");
    g_ps_dequant_fp16 = make_ps(@"dequant_int8_fp16");
    g_ps_fp32_to_fp16 = make_ps(@"fp32_to_fp16");
    g_ps_fp16_to_fp32 = make_ps(@"fp16_to_fp32");
    g_ps_gemm_bt = make_ps(@"gemm_bt");
    g_ps_gemm_bt_acc = make_ps(@"gemm_bt_acc");
    g_ps_gemm_tn = make_ps(@"gemm_tn");
    g_ps_gemm_nn = make_ps(@"gemm_nn");
    g_ps_rmsnorm_save = make_ps(@"rmsnorm_save");
    g_ps_fused_attn = make_ps(@"fused_causal_attention");
    g_ps_fused_attn_bwd = make_ps(@"fused_causal_attention_backward");
    g_ps_bias_add = make_ps(@"bias_add");
    g_ps_rope_fwd = make_ps(@"rope_forward");
    g_ps_rope_bwd = make_ps(@"rope_backward");

    // Try loading Metal 4 matmul2d kernels from pre-compiled .metallib.
    // Search: executable dir, then kernels/ relative to executable, then cwd/kernels/.
    g_use_metal4_gemm = false;
    NSArray* searchPaths = @[
        [[[NSBundle mainBundle] executablePath] stringByDeletingLastPathComponent],
        [[[[NSBundle mainBundle] executablePath] stringByDeletingLastPathComponent] stringByAppendingPathComponent:@"kernels"],
        @"kernels",
        @"."
    ];
    for (NSString* dir in searchPaths) {
        NSString* libPath = [dir stringByAppendingPathComponent:@"gemm_metal4.metallib"];
        if ([[NSFileManager defaultManager] fileExistsAtPath:libPath]) {
            NSURL* libURL = [NSURL fileURLWithPath:libPath];
            NSError* m4err = nil;
            g_metal4_lib = [g_device newLibraryWithURL:libURL error:&m4err];
            if (g_metal4_lib) {
                id<MTLFunction> fn_bt = [g_metal4_lib newFunctionWithName:@"gemm4_bt"];
                id<MTLFunction> fn_nn = [g_metal4_lib newFunctionWithName:@"gemm4_nn"];
                id<MTLFunction> fn_tn = [g_metal4_lib newFunctionWithName:@"gemm4_tn"];
                if (fn_bt && fn_nn && fn_tn) {
                    NSError* psErr = nil;
                    g_ps_gemm4_bt = [g_device newComputePipelineStateWithFunction:fn_bt error:&psErr];
                    g_ps_gemm4_nn = [g_device newComputePipelineStateWithFunction:fn_nn error:&psErr];
                    g_ps_gemm4_tn = [g_device newComputePipelineStateWithFunction:fn_tn error:&psErr];
                    // FP32 cooperative variants
                    id<MTLFunction> fn_fbt = [g_metal4_lib newFunctionWithName:@"gemm4f_bt"];
                    id<MTLFunction> fn_fnn = [g_metal4_lib newFunctionWithName:@"gemm4f_nn"];
                    id<MTLFunction> fn_ftn = [g_metal4_lib newFunctionWithName:@"gemm4f_tn"];
                    if (fn_fbt) g_ps_gemm4f_bt = [g_device newComputePipelineStateWithFunction:fn_fbt error:&psErr];
                    if (fn_fnn) g_ps_gemm4f_nn = [g_device newComputePipelineStateWithFunction:fn_fnn error:&psErr];
                    if (fn_ftn) g_ps_gemm4f_tn = [g_device newComputePipelineStateWithFunction:fn_ftn error:&psErr];
                    if (g_ps_gemm4_bt && g_ps_gemm4_nn && g_ps_gemm4_tn) {
                        g_use_metal4_gemm = true;
                        NSLog(@"mongoose: Metal 4 matmul2d GEMM loaded from %@ (FP32: %s)",
                              libPath, (g_ps_gemm4f_bt && g_ps_gemm4f_nn && g_ps_gemm4f_tn) ? "yes" : "no");
                    }
                }
            }
            if (g_use_metal4_gemm) break;
        }
    }
    if (!g_use_metal4_gemm) {
        NSLog(@"mongoose: Metal 4 .metallib not found — using tiled GEMM fallback");
    }

    return 0;
}

int mtl_compute_ready(void) {
    return g_compute_lib != nil ? 1 : 0;
}

int mtl_has_metal4_gemm(void) {
    return g_use_metal4_gemm ? 1 : 0;
}

// --- Dispatch helpers ---

// Encode a 1D compute dispatch. Buffers set individually per call to avoid ARC array issues.
static void dispatch_1d_setup(id<MTLComputePipelineState> ps, uint n,
                              id<MTLCommandBuffer>* outCmd,
                              id<MTLComputeCommandEncoder>* outEnc) {
    *outCmd = get_cmd();
    *outEnc = get_enc(*outCmd);
    [*outEnc setComputePipelineState:ps];
}

static void dispatch_1d_finish(id<MTLComputePipelineState> ps, uint n,
                                id<MTLCommandBuffer> cmd,
                                id<MTLComputeCommandEncoder> enc) {
    NSUInteger tpg = ps.maxTotalThreadsPerThreadgroup;
    if (tpg > n) tpg = n;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    end_enc(enc);
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

// DNA rung kernel dispatch — paired parameters, 6-point base pair update.
// One GPU dispatch updates both strands simultaneously. Zero CPU involvement.
void mtl_dna_rung_gpu(
    void* d1Ref, void* g1Ref, void* m1Ref, void* v1Ref,
    void* d2Ref, void* g2Ref, void* m2Ref, void* v2Ref,
    float backbone1, float glyco1, float hbond1,
    float hbond2, float glyco2, float backbone2,
    float bondStr,
    float lr, float beta1, float beta2, float bc1, float bc2,
    float eps, float wd, int n) {
    @autoreleasepool {

    // Pack all 8 scalar rung/adam constants into tiny buffers
    id<MTLBuffer> bb1Buf  = [g_device newBufferWithBytes:&backbone1 length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> gl1Buf  = [g_device newBufferWithBytes:&glyco1    length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> hb1Buf  = [g_device newBufferWithBytes:&hbond1    length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> hb2Buf  = [g_device newBufferWithBytes:&hbond2    length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> gl2Buf  = [g_device newBufferWithBytes:&glyco2    length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bb2Buf  = [g_device newBufferWithBytes:&backbone2 length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bsBuf   = [g_device newBufferWithBytes:&bondStr   length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> lrBuf   = [g_device newBufferWithBytes:&lr        length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> b1Buf   = [g_device newBufferWithBytes:&beta1     length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> b2Buf   = [g_device newBufferWithBytes:&beta2     length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bc1Buf  = [g_device newBufferWithBytes:&bc1       length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bc2Buf  = [g_device newBufferWithBytes:&bc2       length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> epsBuf  = [g_device newBufferWithBytes:&eps       length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> wdBuf   = [g_device newBufferWithBytes:&wd        length:sizeof(float) options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cmd = get_cmd();
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_ps_dna_rung];

    // Strand 1: D, G, M, V
    [enc setBuffer:(__bridge id<MTLBuffer>)d1Ref offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)g1Ref offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)m1Ref offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)v1Ref offset:0 atIndex:3];
    // Strand 2: D, G, M, V
    [enc setBuffer:(__bridge id<MTLBuffer>)d2Ref offset:0 atIndex:4];
    [enc setBuffer:(__bridge id<MTLBuffer>)g2Ref offset:0 atIndex:5];
    [enc setBuffer:(__bridge id<MTLBuffer>)m2Ref offset:0 atIndex:6];
    [enc setBuffer:(__bridge id<MTLBuffer>)v2Ref offset:0 atIndex:7];
    // Rung geometry
    [enc setBuffer:bb1Buf offset:0 atIndex:8];
    [enc setBuffer:gl1Buf offset:0 atIndex:9];
    [enc setBuffer:hb1Buf offset:0 atIndex:10];
    [enc setBuffer:hb2Buf offset:0 atIndex:11];
    [enc setBuffer:gl2Buf offset:0 atIndex:12];
    [enc setBuffer:bb2Buf offset:0 atIndex:13];
    [enc setBuffer:bsBuf  offset:0 atIndex:14];
    // Adam constants
    [enc setBuffer:lrBuf  offset:0 atIndex:15];
    [enc setBuffer:b1Buf  offset:0 atIndex:16];
    [enc setBuffer:b2Buf  offset:0 atIndex:17];
    [enc setBuffer:bc1Buf offset:0 atIndex:18];
    [enc setBuffer:bc2Buf offset:0 atIndex:19];
    [enc setBuffer:epsBuf offset:0 atIndex:20];
    [enc setBuffer:wdBuf  offset:0 atIndex:21];

    NSUInteger tpg = g_ps_dna_rung.maxTotalThreadsPerThreadgroup;
    if (tpg > (NSUInteger)n) tpg = (NSUInteger)n;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    [enc endEncoding];
    maybe_commit(cmd);
    } // @autoreleasepool
}

// DNA rung kernel dispatch — warm cache variant.
// Weights (d1, d2) and gradients (g1, g2) are separate MTLBuffers.
// Momentum and velocity for both strands live in a single warm cache MTLBuffer
// at byte offsets. On Apple Silicon unified memory, the CPU (helix) and GPU
// read/write the same physical pages — no copy.
void mtl_dna_rung_warm(
    void* d1Ref, void* g1Ref,
    void* d2Ref, void* g2Ref,
    void* cacheRef,
    int m1Off, int v1Off, int m2Off, int v2Off,
    float backbone1, float glyco1, float hbond1,
    float hbond2, float glyco2, float backbone2,
    float bondStr,
    float lr, float beta1, float beta2, float bc1, float bc2,
    float eps, float wd, int n) {
    @autoreleasepool {

    id<MTLBuffer> bb1Buf  = [g_device newBufferWithBytes:&backbone1 length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> gl1Buf  = [g_device newBufferWithBytes:&glyco1    length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> hb1Buf  = [g_device newBufferWithBytes:&hbond1    length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> hb2Buf  = [g_device newBufferWithBytes:&hbond2    length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> gl2Buf  = [g_device newBufferWithBytes:&glyco2    length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bb2Buf  = [g_device newBufferWithBytes:&backbone2 length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bsBuf   = [g_device newBufferWithBytes:&bondStr   length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> lrBuf   = [g_device newBufferWithBytes:&lr        length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> b1Buf   = [g_device newBufferWithBytes:&beta1     length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> b2Buf   = [g_device newBufferWithBytes:&beta2     length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bc1Buf  = [g_device newBufferWithBytes:&bc1       length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bc2Buf  = [g_device newBufferWithBytes:&bc2       length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> epsBuf  = [g_device newBufferWithBytes:&eps       length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> wdBuf   = [g_device newBufferWithBytes:&wd        length:sizeof(float) options:MTLResourceStorageModeShared];

    id<MTLBuffer> cache = (__bridge id<MTLBuffer>)cacheRef;

    id<MTLCommandBuffer> cmd = get_cmd();
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_ps_dna_rung];

    // Strand 1: D, G from separate buffers; M, V from warm cache at offsets
    [enc setBuffer:(__bridge id<MTLBuffer>)d1Ref offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)g1Ref offset:0 atIndex:1];
    [enc setBuffer:cache offset:(NSUInteger)m1Off atIndex:2];
    [enc setBuffer:cache offset:(NSUInteger)v1Off atIndex:3];
    // Strand 2: D, G from separate buffers; M, V from warm cache at offsets
    [enc setBuffer:(__bridge id<MTLBuffer>)d2Ref offset:0 atIndex:4];
    [enc setBuffer:(__bridge id<MTLBuffer>)g2Ref offset:0 atIndex:5];
    [enc setBuffer:cache offset:(NSUInteger)m2Off atIndex:6];
    [enc setBuffer:cache offset:(NSUInteger)v2Off atIndex:7];
    // Rung geometry
    [enc setBuffer:bb1Buf offset:0 atIndex:8];
    [enc setBuffer:gl1Buf offset:0 atIndex:9];
    [enc setBuffer:hb1Buf offset:0 atIndex:10];
    [enc setBuffer:hb2Buf offset:0 atIndex:11];
    [enc setBuffer:gl2Buf offset:0 atIndex:12];
    [enc setBuffer:bb2Buf offset:0 atIndex:13];
    [enc setBuffer:bsBuf  offset:0 atIndex:14];
    // Adam constants
    [enc setBuffer:lrBuf  offset:0 atIndex:15];
    [enc setBuffer:b1Buf  offset:0 atIndex:16];
    [enc setBuffer:b2Buf  offset:0 atIndex:17];
    [enc setBuffer:bc1Buf offset:0 atIndex:18];
    [enc setBuffer:bc2Buf offset:0 atIndex:19];
    [enc setBuffer:epsBuf offset:0 atIndex:20];
    [enc setBuffer:wdBuf  offset:0 atIndex:21];

    NSUInteger tpg = g_ps_dna_rung.maxTotalThreadsPerThreadgroup;
    if (tpg > (NSUInteger)n) tpg = (NSUInteger)n;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    [enc endEncoding];
    maybe_commit(cmd);
    } // @autoreleasepool
}

// AdamW warm cache variant — single-strand update, m/v from warm cache offsets.
void mtl_adamw_warm(
    void* paramRef, void* gradRef,
    void* cacheRef, int mOff, int vOff,
    float lr, float beta1, float beta2, float bc1, float bc2,
    float eps, float wd, int n) {
    @autoreleasepool {

    id<MTLBuffer> lrBuf   = [g_device newBufferWithBytes:&lr    length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> b1Buf   = [g_device newBufferWithBytes:&beta1 length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> b2Buf   = [g_device newBufferWithBytes:&beta2 length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bc1Buf  = [g_device newBufferWithBytes:&bc1   length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bc2Buf  = [g_device newBufferWithBytes:&bc2   length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> epsBuf  = [g_device newBufferWithBytes:&eps   length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> wdBuf   = [g_device newBufferWithBytes:&wd    length:sizeof(float) options:MTLResourceStorageModeShared];

    id<MTLBuffer> cache = (__bridge id<MTLBuffer>)cacheRef;

    id<MTLCommandBuffer> cmd = get_cmd();
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_ps_adamw];

    [enc setBuffer:(__bridge id<MTLBuffer>)paramRef offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)gradRef  offset:0 atIndex:1];
    [enc setBuffer:cache offset:(NSUInteger)mOff atIndex:2];
    [enc setBuffer:cache offset:(NSUInteger)vOff atIndex:3];
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

// Gradient norm squared — GPU reduction.
// Adds sum-of-squares of grad into out[0] (must be zeroed before first call).
void mtl_grad_norm_sq(void* gradRef, void* outRef, int n) {
    @autoreleasepool {
    id<MTLCommandBuffer> cmd = get_cmd();
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_ps_grad_norm];
    [enc setBuffer:(__bridge id<MTLBuffer>)gradRef offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)outRef  offset:0 atIndex:1];

    NSUInteger tpg = g_ps_grad_norm.maxTotalThreadsPerThreadgroup;
    if (tpg > (NSUInteger)n) tpg = (NSUInteger)n;
    // Round down to multiple of 32 for simd_sum
    tpg = (tpg / 32) * 32;
    if (tpg == 0) tpg = 32;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    [enc endEncoding];
    maybe_commit(cmd);
    } // @autoreleasepool
}

// Forward-gradient LM head: two-pass, no logits buffer, no backward.
// Pass 1: max + sum_exp. Pass 2: loss + dHidden.
// dHidden computed during forward — backward pass eliminated for LM head.
void mtl_lm_head_forward_grad(
    void* hiddenRef, void* embedRef,
    void* maxRef, void* sumExpRef,
    void* targetsRef, void* dHiddenRef, void* lossRef,
    int dim, int vocabSize, int nPositions) {
    @autoreleasepool {

    uint32_t udim = (uint32_t)dim;
    uint32_t uvocab = (uint32_t)vocabSize;
    uint32_t un = (uint32_t)nPositions;
    float invN = 1.0f / (float)nPositions;

    id<MTLBuffer> dimBuf   = [g_device newBufferWithBytes:&udim   length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> vocabBuf = [g_device newBufferWithBytes:&uvocab length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> nBuf     = [g_device newBufferWithBytes:&un     length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> invNBuf  = [g_device newBufferWithBytes:&invN   length:sizeof(float)    options:MTLResourceStorageModeShared];

    // Zero the loss buffer
    float zero = 0.0f;
    memcpy([(__bridge id<MTLBuffer>)lossRef contents], &zero, sizeof(float));

    // Pass 1: max + sum_exp
    {
        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_ps_lm_pass1];
        [enc setBuffer:(__bridge id<MTLBuffer>)hiddenRef offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)embedRef  offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)maxRef    offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)sumExpRef offset:0 atIndex:3];
        [enc setBuffer:dimBuf   offset:0 atIndex:4];
        [enc setBuffer:vocabBuf offset:0 atIndex:5];
        [enc setBuffer:nBuf     offset:0 atIndex:6];

        NSUInteger tpg = g_ps_lm_pass1.maxTotalThreadsPerThreadgroup;
        tpg = (tpg / 32) * 32;
        if (tpg == 0) tpg = 32;
        [enc dispatchThreadgroups:MTLSizeMake(nPositions, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    // Pass 2: loss + dHidden
    {
        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_ps_lm_pass2];
        [enc setBuffer:(__bridge id<MTLBuffer>)hiddenRef   offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)embedRef    offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)maxRef      offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)sumExpRef   offset:0 atIndex:3];
        [enc setBuffer:(__bridge id<MTLBuffer>)targetsRef  offset:0 atIndex:4];
        [enc setBuffer:(__bridge id<MTLBuffer>)dHiddenRef  offset:0 atIndex:5];
        [enc setBuffer:(__bridge id<MTLBuffer>)lossRef     offset:0 atIndex:6];
        [enc setBuffer:dimBuf   offset:0 atIndex:7];
        [enc setBuffer:vocabBuf offset:0 atIndex:8];
        [enc setBuffer:nBuf     offset:0 atIndex:9];
        [enc setBuffer:invNBuf  offset:0 atIndex:10];

        NSUInteger tpg = g_ps_lm_pass2.maxTotalThreadsPerThreadgroup;
        tpg = (tpg / 32) * 32;
        if (tpg == 0) tpg = 32;
        [enc dispatchThreadgroups:MTLSizeMake(nPositions, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    } // @autoreleasepool
}

// Sparse LM head: only scans hot embedding rows from the conductor.
// hotIndicesRef points to a Metal buffer of int32 hot row indices.
void mtl_lm_head_sparse_forward_grad(
    void* hiddenRef, void* embedRef,
    void* maxRef, void* sumExpRef,
    void* targetsRef, void* dHiddenRef, void* lossRef,
    void* hotIndicesRef,
    int dim, int nHot, int nPositions) {
    @autoreleasepool {

    uint32_t udim = (uint32_t)dim;
    uint32_t uhot = (uint32_t)nHot;
    uint32_t un = (uint32_t)nPositions;
    float invN = 1.0f / (float)nPositions;

    id<MTLBuffer> dimBuf  = [g_device newBufferWithBytes:&udim length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> hotBuf  = [g_device newBufferWithBytes:&uhot length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> nBuf    = [g_device newBufferWithBytes:&un   length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> invNBuf = [g_device newBufferWithBytes:&invN length:sizeof(float)    options:MTLResourceStorageModeShared];

    float zero = 0.0f;
    memcpy([(__bridge id<MTLBuffer>)lossRef contents], &zero, sizeof(float));

    // Pass 1: max + sum_exp (sparse)
    {
        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_ps_lm_sparse1];
        [enc setBuffer:(__bridge id<MTLBuffer>)hiddenRef     offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)embedRef      offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)maxRef        offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)sumExpRef     offset:0 atIndex:3];
        [enc setBuffer:(__bridge id<MTLBuffer>)hotIndicesRef offset:0 atIndex:4];
        [enc setBuffer:dimBuf offset:0 atIndex:5];
        [enc setBuffer:hotBuf offset:0 atIndex:6];
        [enc setBuffer:nBuf   offset:0 atIndex:7];

        NSUInteger tpg = g_ps_lm_sparse1.maxTotalThreadsPerThreadgroup;
        tpg = (tpg / 32) * 32;
        if (tpg == 0) tpg = 32;
        [enc dispatchThreadgroups:MTLSizeMake(nPositions, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    // Pass 2: loss + dHidden (sparse)
    {
        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_ps_lm_sparse2];
        [enc setBuffer:(__bridge id<MTLBuffer>)hiddenRef     offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)embedRef      offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)maxRef        offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)sumExpRef     offset:0 atIndex:3];
        [enc setBuffer:(__bridge id<MTLBuffer>)targetsRef    offset:0 atIndex:4];
        [enc setBuffer:(__bridge id<MTLBuffer>)dHiddenRef    offset:0 atIndex:5];
        [enc setBuffer:(__bridge id<MTLBuffer>)lossRef       offset:0 atIndex:6];
        [enc setBuffer:(__bridge id<MTLBuffer>)hotIndicesRef offset:0 atIndex:7];
        [enc setBuffer:dimBuf  offset:0 atIndex:8];
        [enc setBuffer:hotBuf  offset:0 atIndex:9];
        [enc setBuffer:nBuf    offset:0 atIndex:10];
        [enc setBuffer:invNBuf offset:0 atIndex:11];

        NSUInteger tpg = g_ps_lm_sparse2.maxTotalThreadsPerThreadgroup;
        tpg = (tpg / 32) * 32;
        if (tpg == 0) tpg = 32;
        [enc dispatchThreadgroups:MTLSizeMake(nPositions, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    } // @autoreleasepool
}

// Dequant INT8 → FP32 for graph weight placeholders.
void mtl_dequant_int8(void* srcRef, void* scalesRef, void* dstRef, int n, int cols) {
    @autoreleasepool {
    uint32_t ucols = (uint32_t)cols;
    id<MTLBuffer> colsBuf = [g_device newBufferWithBytes:&ucols length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_ps_dequant_int8];
    [enc setBuffer:(__bridge id<MTLBuffer>)srcRef    offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)scalesRef offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)dstRef    offset:0 atIndex:2];
    [enc setBuffer:colsBuf offset:0 atIndex:3];

    NSUInteger tpg = g_ps_dequant_int8.maxTotalThreadsPerThreadgroup;
    if (tpg > 1024) tpg = 1024;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    } // @autoreleasepool
}

// Dequant with delta residual: dst = int8 * scale + delta.
void mtl_dequant_int8_delta(void* srcRef, void* scalesRef, void* deltaRef, void* dstRef, int n, int cols) {
    @autoreleasepool {
    uint32_t ucols = (uint32_t)cols;
    id<MTLBuffer> colsBuf = [g_device newBufferWithBytes:&ucols length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_ps_dequant_delta];
    [enc setBuffer:(__bridge id<MTLBuffer>)srcRef    offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)scalesRef offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)deltaRef  offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)dstRef    offset:0 atIndex:3];
    [enc setBuffer:colsBuf offset:0 atIndex:4];
    NSUInteger tpg = g_ps_dequant_delta.maxTotalThreadsPerThreadgroup;
    if (tpg > 1024) tpg = 1024;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    [enc endEncoding];
    [cmd commit]; [cmd waitUntilCompleted];
    } // @autoreleasepool
}

// Helix Needle: fused INT8 dequant → Adam → requant.
// data is int8, mom/vel are float16, scales/grad are float32, mask is int8.
void mtl_helix_needle(
    void* dataRef, void* scalesRef, void* gradRef,
    void* momRef, void* velRef, void* maskRef, void* deltaRef,
    float lr, float beta1, float beta2, float bc1, float bc2,
    float eps, float wd, int n, int cols) {
    @autoreleasepool {

    uint32_t ucols = (uint32_t)cols;
    id<MTLBuffer> lrBuf   = [g_device newBufferWithBytes:&lr    length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> b1Buf   = [g_device newBufferWithBytes:&beta1 length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> b2Buf   = [g_device newBufferWithBytes:&beta2 length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bc1Buf  = [g_device newBufferWithBytes:&bc1   length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bc2Buf  = [g_device newBufferWithBytes:&bc2   length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> epsBuf  = [g_device newBufferWithBytes:&eps   length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> wdBuf   = [g_device newBufferWithBytes:&wd    length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> colsBuf = [g_device newBufferWithBytes:&ucols length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_ps_needle];
    [enc setBuffer:(__bridge id<MTLBuffer>)dataRef   offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)scalesRef offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)gradRef   offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)momRef    offset:0 atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)velRef    offset:0 atIndex:4];
    [enc setBuffer:(__bridge id<MTLBuffer>)maskRef   offset:0 atIndex:5];
    [enc setBuffer:(__bridge id<MTLBuffer>)deltaRef  offset:0 atIndex:6];
    [enc setBuffer:lrBuf   offset:0 atIndex:7];
    [enc setBuffer:b1Buf   offset:0 atIndex:8];
    [enc setBuffer:b2Buf   offset:0 atIndex:9];
    [enc setBuffer:bc1Buf  offset:0 atIndex:10];
    [enc setBuffer:bc2Buf  offset:0 atIndex:11];
    [enc setBuffer:epsBuf  offset:0 atIndex:12];
    [enc setBuffer:wdBuf   offset:0 atIndex:13];
    [enc setBuffer:colsBuf offset:0 atIndex:14];

    NSUInteger tpg = g_ps_needle.maxTotalThreadsPerThreadgroup;
    if (tpg > 1024) tpg = 1024;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    } // @autoreleasepool
}

// Helix Needle Paired: fused INT8 + DNA rung coupling for paired params.
void mtl_helix_needle_paired(
    void* d1Ref, void* d2Ref,
    void* s1Ref, void* s2Ref,
    void* g1Ref, void* g2Ref,
    void* m1Ref, void* m2Ref,
    void* v1Ref, void* v2Ref,
    void* maskRef, void* delta1Ref, void* delta2Ref,
    float lr, float beta1, float beta2, float bc1, float bc2,
    float eps, float wd,
    float backbone1, float glyco1, float hbond1,
    float hbond2, float glyco2, float backbone2,
    float bondStrength, int n, int cols) {
    @autoreleasepool {

    uint32_t ucols = (uint32_t)cols;
    id<MTLBuffer> lrBuf   = [g_device newBufferWithBytes:&lr          length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> b1Buf   = [g_device newBufferWithBytes:&beta1       length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> b2Buf   = [g_device newBufferWithBytes:&beta2       length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bc1Buf  = [g_device newBufferWithBytes:&bc1         length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bc2Buf  = [g_device newBufferWithBytes:&bc2         length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> epsBuf  = [g_device newBufferWithBytes:&eps         length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> wdBuf   = [g_device newBufferWithBytes:&wd          length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bb1Buf  = [g_device newBufferWithBytes:&backbone1   length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> gl1Buf  = [g_device newBufferWithBytes:&glyco1      length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> hb1Buf  = [g_device newBufferWithBytes:&hbond1      length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> hb2Buf  = [g_device newBufferWithBytes:&hbond2      length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> gl2Buf  = [g_device newBufferWithBytes:&glyco2      length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bb2Buf  = [g_device newBufferWithBytes:&backbone2   length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bsBuf   = [g_device newBufferWithBytes:&bondStrength length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> colsBuf = [g_device newBufferWithBytes:&ucols       length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_ps_needle_paired];
    [enc setBuffer:(__bridge id<MTLBuffer>)d1Ref     offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)d2Ref     offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)s1Ref     offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)s2Ref     offset:0 atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)g1Ref     offset:0 atIndex:4];
    [enc setBuffer:(__bridge id<MTLBuffer>)g2Ref     offset:0 atIndex:5];
    [enc setBuffer:(__bridge id<MTLBuffer>)m1Ref     offset:0 atIndex:6];
    [enc setBuffer:(__bridge id<MTLBuffer>)m2Ref     offset:0 atIndex:7];
    [enc setBuffer:(__bridge id<MTLBuffer>)v1Ref     offset:0 atIndex:8];
    [enc setBuffer:(__bridge id<MTLBuffer>)v2Ref     offset:0 atIndex:9];
    [enc setBuffer:(__bridge id<MTLBuffer>)maskRef   offset:0 atIndex:10];
    [enc setBuffer:(__bridge id<MTLBuffer>)delta1Ref offset:0 atIndex:11];
    [enc setBuffer:(__bridge id<MTLBuffer>)delta2Ref offset:0 atIndex:12];
    [enc setBuffer:lrBuf   offset:0 atIndex:13];
    [enc setBuffer:b1Buf   offset:0 atIndex:14];
    [enc setBuffer:b2Buf   offset:0 atIndex:15];
    [enc setBuffer:bc1Buf  offset:0 atIndex:16];
    [enc setBuffer:bc2Buf  offset:0 atIndex:17];
    [enc setBuffer:epsBuf  offset:0 atIndex:18];
    [enc setBuffer:wdBuf   offset:0 atIndex:19];
    [enc setBuffer:bb1Buf  offset:0 atIndex:20];
    [enc setBuffer:gl1Buf  offset:0 atIndex:21];
    [enc setBuffer:hb1Buf  offset:0 atIndex:22];
    [enc setBuffer:hb2Buf  offset:0 atIndex:23];
    [enc setBuffer:gl2Buf  offset:0 atIndex:24];
    [enc setBuffer:bb2Buf  offset:0 atIndex:25];
    [enc setBuffer:bsBuf   offset:0 atIndex:26];
    [enc setBuffer:colsBuf offset:0 atIndex:27];

    NSUInteger tpg = g_ps_needle_paired.maxTotalThreadsPerThreadgroup;
    if (tpg > 1024) tpg = 1024;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    } // @autoreleasepool
}

// === Coiled pipeline infrastructure ===

// Encode a command buffer on queue 2 (backward coil).
// Returns a command buffer that the caller can encode compute into.
void* mtl_coil_begin_bwd(void) {
    id<MTLCommandBuffer> cmd = [g_queue2 commandBuffer];
    return (__bridge_retained void*)cmd;
}

// Signal the shared event from a command buffer (marks a layer as done).
void mtl_coil_signal(void* cmdRef, uint64_t value) {
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)cmdRef;
    [cmd encodeSignalEvent:g_coil_event value:value];
}

// Wait for the shared event in a command buffer (wait for forward layer to complete).
void mtl_coil_wait(void* cmdRef, uint64_t value) {
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)cmdRef;
    [cmd encodeWaitForEvent:g_coil_event value:value];
}

// Commit a coiled command buffer (non-blocking).
void mtl_coil_commit(void* cmdRef) {
    id<MTLCommandBuffer> cmd = (__bridge_transfer id<MTLCommandBuffer>)cmdRef;
    [cmd commit];
}

// Wait for all coiled operations to complete.
void mtl_coil_sync(void) {
    id<MTLCommandBuffer> cmd = [g_queue2 commandBuffer];
    [cmd commit];
    [cmd waitUntilCompleted];
    // Also drain queue 1
    id<MTLCommandBuffer> cmd1 = [g_queue commandBuffer];
    [cmd1 commit];
    [cmd1 waitUntilCompleted];
}

// Reset the shared event counter for a new training step.
void mtl_coil_reset(void) {
    g_coil_event.signaledValue = 0;
}

// Non-graph matmul: C = A @ B^T using MPS (for per-layer dispatch outside MPSGraph).
// A[M,K], B[N,K] → C[M,N]. All row-major FP32 shared buffers.
// queueIdx: 0 = primary queue, 1 = backward queue.
void mtl_matmul_raw(void* aRef, void* bRef, void* cRef,
                    int M, int K, int N, int queueIdx) {
    @autoreleasepool {
    id<MTLCommandQueue> q = (queueIdx == 0) ? g_queue : g_queue2;

    // MPS matrix descriptors — row-major FP32
    MPSMatrixDescriptor* descA = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K
        rowBytes:K*sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* descB = [MPSMatrixDescriptor matrixDescriptorWithRows:N columns:K
        rowBytes:K*sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N
        rowBytes:N*sizeof(float) dataType:MPSDataTypeFloat32];

    MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)aRef descriptor:descA];
    MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)bRef descriptor:descB];
    MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)cRef descriptor:descC];

    // C = A @ B^T (transposeRight = YES)
    MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc]
        initWithDevice:g_device transposeLeft:NO transposeRight:YES
        resultRows:M resultColumns:N interiorColumns:K alpha:1.0 beta:0.0];

    id<MTLCommandBuffer> cmd = [q commandBuffer];
    [mm encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
    [cmd commit];
    [cmd waitUntilCompleted];
    } // @autoreleasepool
}

// Non-graph matmul (no transpose): C = A @ B.
// A[M,K], B[K,N] → C[M,N].
void mtl_matmul_raw_nn(void* aRef, void* bRef, void* cRef,
                       int M, int K, int N, int queueIdx) {
    @autoreleasepool {
    id<MTLCommandQueue> q = (queueIdx == 0) ? g_queue : g_queue2;

    MPSMatrixDescriptor* descA = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K
        rowBytes:K*sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* descB = [MPSMatrixDescriptor matrixDescriptorWithRows:K columns:N
        rowBytes:N*sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N
        rowBytes:N*sizeof(float) dataType:MPSDataTypeFloat32];

    MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)aRef descriptor:descA];
    MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)bRef descriptor:descB];
    MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)cRef descriptor:descC];

    MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc]
        initWithDevice:g_device transposeLeft:NO transposeRight:NO
        resultRows:M resultColumns:N interiorColumns:K alpha:1.0 beta:0.0];

    id<MTLCommandBuffer> cmd = [q commandBuffer];
    [mm encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
    [cmd commit];
    [cmd waitUntilCompleted];
    } // @autoreleasepool
}

// C += A @ B^T (accumulate into C). beta=1.0.
void mtl_matmul_raw_acc(void* aRef, void* bRef, void* cRef,
                        int M, int K, int N, int queueIdx) {
    @autoreleasepool {
    id<MTLCommandQueue> q = (queueIdx == 0) ? g_queue : g_queue2;

    MPSMatrixDescriptor* descA = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K
        rowBytes:K*sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* descB = [MPSMatrixDescriptor matrixDescriptorWithRows:N columns:K
        rowBytes:K*sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N
        rowBytes:N*sizeof(float) dataType:MPSDataTypeFloat32];

    MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)aRef descriptor:descA];
    MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)bRef descriptor:descB];
    MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)cRef descriptor:descC];

    MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc]
        initWithDevice:g_device transposeLeft:NO transposeRight:YES
        resultRows:M resultColumns:N interiorColumns:K alpha:1.0 beta:1.0];

    id<MTLCommandBuffer> cmd = [q commandBuffer];
    [mm encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
    [cmd commit];
    [cmd waitUntilCompleted];
    } // @autoreleasepool
}

// Transpose matmul: C = A^T @ B. For weight gradient dW = act^T @ dOut.
void mtl_matmul_raw_tn(void* aRef, void* bRef, void* cRef,
                       int M, int K, int N, int queueIdx) {
    @autoreleasepool {
    id<MTLCommandQueue> q = (queueIdx == 0) ? g_queue : g_queue2;

    // A^T: original A is [K, M], transposed to [M, K]
    MPSMatrixDescriptor* descA = [MPSMatrixDescriptor matrixDescriptorWithRows:K columns:M
        rowBytes:M*sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* descB = [MPSMatrixDescriptor matrixDescriptorWithRows:K columns:N
        rowBytes:N*sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N
        rowBytes:N*sizeof(float) dataType:MPSDataTypeFloat32];

    MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)aRef descriptor:descA];
    MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)bRef descriptor:descB];
    MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)cRef descriptor:descC];

    MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc]
        initWithDevice:g_device transposeLeft:YES transposeRight:NO
        resultRows:M resultColumns:N interiorColumns:K alpha:1.0 beta:0.0];

    id<MTLCommandBuffer> cmd = [q commandBuffer];
    [mm encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
    [cmd commit];
    [cmd waitUntilCompleted];
    } // @autoreleasepool
}

void mtl_zero_gpu(void* ptrRef, int n) {
    id<MTLCommandBuffer> cmd; id<MTLComputeCommandEncoder> enc;
    dispatch_1d_setup(g_ps_zero_mem, n, &cmd, &enc);
    [enc setBuffer:(__bridge id<MTLBuffer>)ptrRef offset:0 atIndex:0];
    dispatch_1d_finish(g_ps_zero_mem, n, cmd, enc);
}

// === Batched command buffer for fused dispatch ===
// Two slots for parallel forward passes. Slot 0 = g_queue, slot 1 = g_queue2.

void mtl_fused_begin(void) {
    g_active_fused_slot = 0;
    g_fused_cmd[0] = [g_queue commandBuffer];
    g_fused_enc[0] = [g_fused_cmd[0] computeCommandEncoder];
}

void mtl_fused_end(void) {
    [g_fused_enc[0] endEncoding];
    [g_fused_cmd[0] commit];
    [g_fused_cmd[0] waitUntilCompleted];
    g_fused_enc[0] = nil;
    g_fused_cmd[0] = nil;
}

// Begin fused dispatch on a specific slot (0-3).
void mtl_fused_begin_slot(int slot) {
    if (slot < 0 || slot >= MTL_MAX_QUEUES) return;
    g_active_fused_slot = slot;
    g_fused_cmd[slot] = [g_queues[slot] commandBuffer];
    g_fused_enc[slot] = [g_fused_cmd[slot] computeCommandEncoder];
}

// End fused dispatch on a slot — commits but does NOT wait.
void mtl_fused_end_slot(int slot) {
    if (!g_fused_enc[slot]) return;
    [g_fused_enc[slot] endEncoding];
    [g_fused_cmd[slot] commit];
    g_fused_enc[slot] = nil;
    g_fused_cmd[slot] = nil;
}

// Set the active fused slot (for routing get_cmd/get_enc).
void mtl_fused_set_slot(int slot) {
    g_active_fused_slot = slot;
}

// Wait for all fused slots to complete.
void mtl_fused_sync_all(void) {
    for (int i = 0; i < MTL_MAX_QUEUES; i++) {
        if (g_fused_cmd[i]) {
            [g_fused_cmd[i] waitUntilCompleted];
            g_fused_enc[i] = nil;
            g_fused_cmd[i] = nil;
        }
    }
    for (int i = 0; i < MTL_MAX_QUEUES; i++) {
        id<MTLCommandBuffer> c = [g_queues[i] commandBuffer];
        [c commit]; [c waitUntilCompleted];
    }
}

// Fused GEMM: C = A @ B^T, encoded into the active fused command buffer.
void mtl_fused_gemm_bt(void* aRef, void* bRef, void* cRef, int M, int K, int N) {
    uint32_t um = M, uk = K, un = N;
    id<MTLBuffer> mBuf = [g_device newBufferWithBytes:&um length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> kBuf = [g_device newBufferWithBytes:&uk length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> nBuf = [g_device newBufferWithBytes:&un length:4 options:MTLResourceStorageModeShared];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)aRef offset:0 atIndex:0];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)bRef offset:0 atIndex:1];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)cRef offset:0 atIndex:2];
    [g_fused_enc[g_active_fused_slot] setBuffer:mBuf offset:0 atIndex:3];
    [g_fused_enc[g_active_fused_slot] setBuffer:kBuf offset:0 atIndex:4];
    [g_fused_enc[g_active_fused_slot] setBuffer:nBuf offset:0 atIndex:5];
    if (g_use_metal4_gemm) {
        // Metal 4 matmul2d: 4 simdgroups per threadgroup, 64x32 tiles
        [g_fused_enc[g_active_fused_slot] setComputePipelineState:g_ps_gemm4_bt];
        NSUInteger simdWidth = g_ps_gemm4_bt.threadExecutionWidth;
        [g_fused_enc[g_active_fused_slot] dispatchThreadgroups:MTLSizeMake((N+31)/32, (M+63)/64, 1)
                    threadsPerThreadgroup:MTLSizeMake(simdWidth * 4, 1, 1)];
    } else {
        [g_fused_enc[g_active_fused_slot] setComputePipelineState:g_ps_gemm_bt];
        [g_fused_enc[g_active_fused_slot] dispatchThreadgroups:MTLSizeMake((N+31)/32, (M+31)/32, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 32, 1)];
    }
}

void mtl_fused_gemm_nn(void* aRef, void* bRef, void* cRef, int M, int K, int N) {
    uint32_t um = M, uk = K, un = N;
    id<MTLBuffer> mBuf = [g_device newBufferWithBytes:&um length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> kBuf = [g_device newBufferWithBytes:&uk length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> nBuf = [g_device newBufferWithBytes:&un length:4 options:MTLResourceStorageModeShared];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)aRef offset:0 atIndex:0];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)bRef offset:0 atIndex:1];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)cRef offset:0 atIndex:2];
    [g_fused_enc[g_active_fused_slot] setBuffer:mBuf offset:0 atIndex:3];
    [g_fused_enc[g_active_fused_slot] setBuffer:kBuf offset:0 atIndex:4];
    [g_fused_enc[g_active_fused_slot] setBuffer:nBuf offset:0 atIndex:5];
    if (g_use_metal4_gemm) {
        [g_fused_enc[g_active_fused_slot] setComputePipelineState:g_ps_gemm4_nn];
        NSUInteger simdWidth = g_ps_gemm4_nn.threadExecutionWidth;
        [g_fused_enc[g_active_fused_slot] dispatchThreadgroups:MTLSizeMake((N+31)/32, (M+63)/64, 1)
                    threadsPerThreadgroup:MTLSizeMake(simdWidth * 4, 1, 1)];
    } else {
        [g_fused_enc[g_active_fused_slot] setComputePipelineState:g_ps_gemm_nn];
        [g_fused_enc[g_active_fused_slot] dispatchThreadgroups:MTLSizeMake((N+31)/32, (M+31)/32, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 32, 1)];
    }
}

void mtl_fused_gemm_tn(void* aRef, void* bRef, void* cRef, int M, int K, int N) {
    uint32_t um = M, uk = K, un = N;
    id<MTLBuffer> mBuf = [g_device newBufferWithBytes:&um length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> kBuf = [g_device newBufferWithBytes:&uk length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> nBuf = [g_device newBufferWithBytes:&un length:4 options:MTLResourceStorageModeShared];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)aRef offset:0 atIndex:0];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)bRef offset:0 atIndex:1];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)cRef offset:0 atIndex:2];
    [g_fused_enc[g_active_fused_slot] setBuffer:mBuf offset:0 atIndex:3];
    [g_fused_enc[g_active_fused_slot] setBuffer:kBuf offset:0 atIndex:4];
    [g_fused_enc[g_active_fused_slot] setBuffer:nBuf offset:0 atIndex:5];
    if (g_use_metal4_gemm) {
        [g_fused_enc[g_active_fused_slot] setComputePipelineState:g_ps_gemm4_tn];
        NSUInteger simdWidth = g_ps_gemm4_tn.threadExecutionWidth;
        [g_fused_enc[g_active_fused_slot] dispatchThreadgroups:MTLSizeMake((N+31)/32, (M+63)/64, 1)
                    threadsPerThreadgroup:MTLSizeMake(simdWidth * 4, 1, 1)];
    } else {
        [g_fused_enc[g_active_fused_slot] setComputePipelineState:g_ps_gemm_tn];
        [g_fused_enc[g_active_fused_slot] dispatchThreadgroups:MTLSizeMake((N+31)/32, (M+31)/32, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 32, 1)];
    }
}

void mtl_fused_dequant(void* srcRef, void* scalesRef, void* dstRef, int n, int cols) {
    uint32_t ucols = (uint32_t)cols;
    id<MTLBuffer> colsBuf = [g_device newBufferWithBytes:&ucols length:4 options:MTLResourceStorageModeShared];
    [g_fused_enc[g_active_fused_slot] setComputePipelineState:g_ps_dequant_int8];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)srcRef    offset:0 atIndex:0];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)scalesRef offset:0 atIndex:1];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)dstRef    offset:0 atIndex:2];
    [g_fused_enc[g_active_fused_slot] setBuffer:colsBuf offset:0 atIndex:3];
    NSUInteger tpg = g_ps_dequant_int8.maxTotalThreadsPerThreadgroup;
    if (tpg > 1024) tpg = 1024;
    [g_fused_enc[g_active_fused_slot] dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
}

// Fused inline needle — fires before matmul reads weight, updates dequant scratch.
void mtl_fused_needle_inline(void* dataRef, void* scalesRef, void* cacheRef,
                             void* momRef, void* deltaRef, void* maskRef,
                             float signalScale, float lr, float beta1, float wd,
                             int n, int cols) {
    float ss = signalScale, lr2 = lr, b1 = beta1, wd2 = wd;
    uint32_t ucols = (uint32_t)cols;
    id<MTLBuffer> ssBuf   = [g_device newBufferWithBytes:&ss   length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> lrBuf   = [g_device newBufferWithBytes:&lr2  length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> b1Buf   = [g_device newBufferWithBytes:&b1   length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> wdBuf   = [g_device newBufferWithBytes:&wd2  length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> colsBuf = [g_device newBufferWithBytes:&ucols length:4 options:MTLResourceStorageModeShared];
    [g_fused_enc[g_active_fused_slot] setComputePipelineState:g_ps_needle_inline];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)dataRef   offset:0 atIndex:0];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)scalesRef offset:0 atIndex:1];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)cacheRef  offset:0 atIndex:2];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)momRef    offset:0 atIndex:3];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)deltaRef  offset:0 atIndex:4];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)maskRef   offset:0 atIndex:5];
    [g_fused_enc[g_active_fused_slot] setBuffer:ssBuf   offset:0 atIndex:6];
    [g_fused_enc[g_active_fused_slot] setBuffer:lrBuf   offset:0 atIndex:7];
    [g_fused_enc[g_active_fused_slot] setBuffer:b1Buf   offset:0 atIndex:8];
    [g_fused_enc[g_active_fused_slot] setBuffer:wdBuf   offset:0 atIndex:9];
    [g_fused_enc[g_active_fused_slot] setBuffer:colsBuf offset:0 atIndex:10];
    NSUInteger tpg = g_ps_needle_inline.maxTotalThreadsPerThreadgroup;
    if (tpg > 1024) tpg = 1024;
    [g_fused_enc[g_active_fused_slot] dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
}

// Cross-entropy loss on GPU — no gradient, no download of logits.
void mtl_ce_loss(void* logitsRef, void* targetsRef, void* lossesRef, int seqLen, int vocabSize) {
    @autoreleasepool {
    uint32_t uv = (uint32_t)vocabSize;
    id<MTLBuffer> vBuf = [g_device newBufferWithBytes:&uv length:4 options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_ps_ce_loss];
    [enc setBuffer:(__bridge id<MTLBuffer>)logitsRef  offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)targetsRef offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)lossesRef  offset:0 atIndex:2];
    [enc setBuffer:vBuf offset:0 atIndex:3];
    NSUInteger tpg = g_ps_ce_loss.maxTotalThreadsPerThreadgroup;
    if (tpg > 1024) tpg = 1024;
    tpg = (tpg / 32) * 32;
    if (tpg == 0) tpg = 32;
    [enc dispatchThreadgroups:MTLSizeMake(seqLen, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    [enc endEncoding];
    [cmd commit]; [cmd waitUntilCompleted];
    } // @autoreleasepool
}

// Fused FP32 GEMM: C = A @ B^T (FP32 in, FP32 out, Metal 4 cooperative path).
void mtl_fused_gemm_f32_bt(void* aRef, void* bRef, void* cRef, int M, int K, int N) {
    if (!g_ps_gemm4f_bt) { mtl_fused_gemm_bt(aRef, bRef, cRef, M, K, N); return; }
    uint32_t um = M, uk = K, un = N;
    id<MTLBuffer> mBuf = [g_device newBufferWithBytes:&um length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> kBuf = [g_device newBufferWithBytes:&uk length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> nBuf = [g_device newBufferWithBytes:&un length:4 options:MTLResourceStorageModeShared];
    [g_fused_enc[g_active_fused_slot] setComputePipelineState:g_ps_gemm4f_bt];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)aRef offset:0 atIndex:0];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)bRef offset:0 atIndex:1];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)cRef offset:0 atIndex:2];
    [g_fused_enc[g_active_fused_slot] setBuffer:mBuf offset:0 atIndex:3];
    [g_fused_enc[g_active_fused_slot] setBuffer:kBuf offset:0 atIndex:4];
    [g_fused_enc[g_active_fused_slot] setBuffer:nBuf offset:0 atIndex:5];
    NSUInteger simdWidth = g_ps_gemm4f_bt.threadExecutionWidth;
    [g_fused_enc[g_active_fused_slot] dispatchThreadgroups:MTLSizeMake((N+31)/32, (M+63)/64, 1)
                threadsPerThreadgroup:MTLSizeMake(simdWidth * 4, 1, 1)];
}
void mtl_fused_gemm_f32_nn(void* aRef, void* bRef, void* cRef, int M, int K, int N) {
    if (!g_ps_gemm4f_nn) { mtl_fused_gemm_nn(aRef, bRef, cRef, M, K, N); return; }
    uint32_t um = M, uk = K, un = N;
    id<MTLBuffer> mBuf = [g_device newBufferWithBytes:&um length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> kBuf = [g_device newBufferWithBytes:&uk length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> nBuf = [g_device newBufferWithBytes:&un length:4 options:MTLResourceStorageModeShared];
    [g_fused_enc[g_active_fused_slot] setComputePipelineState:g_ps_gemm4f_nn];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)aRef offset:0 atIndex:0];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)bRef offset:0 atIndex:1];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)cRef offset:0 atIndex:2];
    [g_fused_enc[g_active_fused_slot] setBuffer:mBuf offset:0 atIndex:3];
    [g_fused_enc[g_active_fused_slot] setBuffer:kBuf offset:0 atIndex:4];
    [g_fused_enc[g_active_fused_slot] setBuffer:nBuf offset:0 atIndex:5];
    NSUInteger simdWidth = g_ps_gemm4f_nn.threadExecutionWidth;
    [g_fused_enc[g_active_fused_slot] dispatchThreadgroups:MTLSizeMake((N+31)/32, (M+63)/64, 1)
                threadsPerThreadgroup:MTLSizeMake(simdWidth * 4, 1, 1)];
}
void mtl_fused_gemm_f32_tn(void* aRef, void* bRef, void* cRef, int M, int K, int N) {
    if (!g_ps_gemm4f_tn) { mtl_fused_gemm_tn(aRef, bRef, cRef, M, K, N); return; }
    uint32_t um = M, uk = K, un = N;
    id<MTLBuffer> mBuf = [g_device newBufferWithBytes:&um length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> kBuf = [g_device newBufferWithBytes:&uk length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> nBuf = [g_device newBufferWithBytes:&un length:4 options:MTLResourceStorageModeShared];
    [g_fused_enc[g_active_fused_slot] setComputePipelineState:g_ps_gemm4f_tn];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)aRef offset:0 atIndex:0];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)bRef offset:0 atIndex:1];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)cRef offset:0 atIndex:2];
    [g_fused_enc[g_active_fused_slot] setBuffer:mBuf offset:0 atIndex:3];
    [g_fused_enc[g_active_fused_slot] setBuffer:kBuf offset:0 atIndex:4];
    [g_fused_enc[g_active_fused_slot] setBuffer:nBuf offset:0 atIndex:5];
    NSUInteger simdWidth = g_ps_gemm4f_tn.threadExecutionWidth;
    [g_fused_enc[g_active_fused_slot] dispatchThreadgroups:MTLSizeMake((N+31)/32, (M+63)/64, 1)
                threadsPerThreadgroup:MTLSizeMake(simdWidth * 4, 1, 1)];
}

// Fused RMSNorm backward.
void mtl_fused_rmsnorm_bwd(void* dOutRef, void* xInRef, void* wRef, void* scaleRef, void* dxRef, int seqLen, int dim) {
    uint32_t udim = (uint32_t)dim;
    id<MTLBuffer> dimBuf = [g_device newBufferWithBytes:&udim length:4 options:MTLResourceStorageModeShared];
    [g_fused_enc[g_active_fused_slot] setComputePipelineState:g_ps_rmsnorm_bwd];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)dOutRef  offset:0 atIndex:0];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)xInRef   offset:0 atIndex:1];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)wRef     offset:0 atIndex:2];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)scaleRef offset:0 atIndex:3];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)dxRef    offset:0 atIndex:4];
    [g_fused_enc[g_active_fused_slot] setBuffer:dimBuf offset:0 atIndex:5];
    NSUInteger tpg = g_ps_rmsnorm_bwd.maxTotalThreadsPerThreadgroup;
    if (tpg > (NSUInteger)dim) tpg = (NSUInteger)dim;
    tpg = (tpg / 32) * 32;
    if (tpg == 0) tpg = 32;
    [g_fused_enc[g_active_fused_slot] dispatchThreadgroups:MTLSizeMake(seqLen, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
}

// Non-fused RMSNorm backward.
void mtl_rmsnorm_bwd_q(void* dOutRef, void* xInRef, void* wRef, void* scaleRef, void* dxRef, int seqLen, int dim, int queueIdx) {
    @autoreleasepool {
    id<MTLCommandQueue> q = (queueIdx == 0) ? g_queue : g_queue2;
    uint32_t udim = (uint32_t)dim;
    id<MTLBuffer> dimBuf = [g_device newBufferWithBytes:&udim length:4 options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_ps_rmsnorm_bwd];
    [enc setBuffer:(__bridge id<MTLBuffer>)dOutRef  offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)xInRef   offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)wRef     offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)scaleRef offset:0 atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)dxRef    offset:0 atIndex:4];
    [enc setBuffer:dimBuf offset:0 atIndex:5];
    NSUInteger tpg = g_ps_rmsnorm_bwd.maxTotalThreadsPerThreadgroup;
    if (tpg > (NSUInteger)dim) tpg = (NSUInteger)dim;
    tpg = (tpg / 32) * 32;
    if (tpg == 0) tpg = 32;
    [enc dispatchThreadgroups:MTLSizeMake(seqLen, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    [enc endEncoding];
    [cmd commit]; [cmd waitUntilCompleted];
    } // @autoreleasepool
}

// Fused dequant INT8 → FP16 for Metal 4 GEMM.
void mtl_fused_dequant_fp16(void* srcRef, void* scalesRef, void* dstRef, int n, int cols) {
    uint32_t ucols = (uint32_t)cols;
    id<MTLBuffer> colsBuf = [g_device newBufferWithBytes:&ucols length:4 options:MTLResourceStorageModeShared];
    [g_fused_enc[g_active_fused_slot] setComputePipelineState:g_ps_dequant_fp16];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)srcRef    offset:0 atIndex:0];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)scalesRef offset:0 atIndex:1];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)dstRef    offset:0 atIndex:2];
    [g_fused_enc[g_active_fused_slot] setBuffer:colsBuf offset:0 atIndex:3];
    NSUInteger tpg = g_ps_dequant_fp16.maxTotalThreadsPerThreadgroup;
    if (tpg > 1024) tpg = 1024;
    [g_fused_enc[g_active_fused_slot] dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
}

// Fused FP32 → FP16 narrowing.
void mtl_fused_fp32_to_fp16(void* srcRef, void* dstRef, int n) {
    [g_fused_enc[g_active_fused_slot] setComputePipelineState:g_ps_fp32_to_fp16];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)srcRef offset:0 atIndex:0];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)dstRef offset:0 atIndex:1];
    NSUInteger tpg = g_ps_fp32_to_fp16.maxTotalThreadsPerThreadgroup;
    if (tpg > 1024) tpg = 1024;
    [g_fused_enc[g_active_fused_slot] dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
}

// Fused FP16 → FP32 widening.
void mtl_fused_fp16_to_fp32(void* srcRef, void* dstRef, int n) {
    [g_fused_enc[g_active_fused_slot] setComputePipelineState:g_ps_fp16_to_fp32];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)srcRef offset:0 atIndex:0];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)dstRef offset:0 atIndex:1];
    NSUInteger tpg = g_ps_fp16_to_fp32.maxTotalThreadsPerThreadgroup;
    if (tpg > 1024) tpg = 1024;
    [g_fused_enc[g_active_fused_slot] dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
}

void mtl_fused_rmsnorm(void* xRef, void* wRef, void* scaleRef, int seqLen, int dim) {
    uint32_t udim = (uint32_t)dim;
    float eps = 1e-6f;
    id<MTLBuffer> dimBuf = [g_device newBufferWithBytes:&udim length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> epsBuf = [g_device newBufferWithBytes:&eps length:4 options:MTLResourceStorageModeShared];
    [g_fused_enc[g_active_fused_slot] setComputePipelineState:g_ps_rmsnorm_save];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)xRef     offset:0 atIndex:0];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)wRef     offset:0 atIndex:1];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)scaleRef offset:0 atIndex:2];
    [g_fused_enc[g_active_fused_slot] setBuffer:dimBuf offset:0 atIndex:3];
    [g_fused_enc[g_active_fused_slot] setBuffer:epsBuf offset:0 atIndex:4];
    NSUInteger tpg = g_ps_rmsnorm_save.maxTotalThreadsPerThreadgroup;
    if (tpg > (NSUInteger)dim) tpg = (NSUInteger)dim;
    tpg = (tpg / 32) * 32;
    if (tpg == 0) tpg = 32;
    [g_fused_enc[g_active_fused_slot] dispatchThreadgroups:MTLSizeMake(seqLen, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
}

void mtl_fused_add_inplace(void* aRef, void* bRef, int n) {
    [g_fused_enc[g_active_fused_slot] setComputePipelineState:g_ps_add_inplace];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)aRef offset:0 atIndex:0];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)bRef offset:0 atIndex:1];
    NSUInteger tpg = g_ps_add_inplace.maxTotalThreadsPerThreadgroup;
    if (tpg > 1024) tpg = 1024;
    [g_fused_enc[g_active_fused_slot] dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
}

void mtl_fused_silu_gate_mul(void* gateRef, void* upRef, void* outRef, int n) {
    [g_fused_enc[g_active_fused_slot] setComputePipelineState:g_ps_silu_gate_mul];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)gateRef offset:0 atIndex:0];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)upRef   offset:0 atIndex:1];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)outRef  offset:0 atIndex:2];
    NSUInteger tpg = g_ps_silu_gate_mul.maxTotalThreadsPerThreadgroup;
    if (tpg > 1024) tpg = 1024;
    [g_fused_enc[g_active_fused_slot] dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
}

void mtl_fused_attn(void* qRef, void* kRef, void* vRef, void* outRef, void* scoresRef,
                    int dim, int kvDim, int headDim, int nHeads, int nKVHeads, int seqLen) {
    uint32_t ud = dim, ukv = kvDim, uhd = headDim, unh = nHeads, unkv = nKVHeads, un = seqLen;
    id<MTLBuffer> dBuf   = [g_device newBufferWithBytes:&ud   length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> kvBuf  = [g_device newBufferWithBytes:&ukv  length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> hdBuf  = [g_device newBufferWithBytes:&uhd  length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> nhBuf  = [g_device newBufferWithBytes:&unh  length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> nkvBuf = [g_device newBufferWithBytes:&unkv length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> nBuf   = [g_device newBufferWithBytes:&un   length:4 options:MTLResourceStorageModeShared];
    [g_fused_enc[g_active_fused_slot] setComputePipelineState:g_ps_fused_attn];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)qRef      offset:0 atIndex:0];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)kRef      offset:0 atIndex:1];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)vRef      offset:0 atIndex:2];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)outRef    offset:0 atIndex:3];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)scoresRef offset:0 atIndex:4];
    [g_fused_enc[g_active_fused_slot] setBuffer:dBuf   offset:0 atIndex:5];
    [g_fused_enc[g_active_fused_slot] setBuffer:kvBuf  offset:0 atIndex:6];
    [g_fused_enc[g_active_fused_slot] setBuffer:hdBuf  offset:0 atIndex:7];
    [g_fused_enc[g_active_fused_slot] setBuffer:nhBuf  offset:0 atIndex:8];
    [g_fused_enc[g_active_fused_slot] setBuffer:nkvBuf offset:0 atIndex:9];
    [g_fused_enc[g_active_fused_slot] setBuffer:nBuf   offset:0 atIndex:10];
    NSUInteger tpg = (NSUInteger)seqLen;
    if (tpg > g_ps_fused_attn.maxTotalThreadsPerThreadgroup)
        tpg = g_ps_fused_attn.maxTotalThreadsPerThreadgroup;
    [g_fused_enc[g_active_fused_slot] dispatchThreadgroups:MTLSizeMake(nHeads, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
}

void mtl_fused_rope(void* xRef, int headDim, int nHeads, float theta, int stride, int seqLen) {
    uint32_t uhd = headDim, unh = nHeads, ust = stride;
    float absTheta = theta < 0 ? -theta : theta;
    id<MTLBuffer> hdBuf = [g_device newBufferWithBytes:&uhd      length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> nhBuf = [g_device newBufferWithBytes:&unh      length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> thBuf = [g_device newBufferWithBytes:&absTheta length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> stBuf = [g_device newBufferWithBytes:&ust      length:4 options:MTLResourceStorageModeShared];
    id<MTLComputePipelineState> ps = (theta < 0) ? g_ps_rope_bwd : g_ps_rope_fwd;
    [g_fused_enc[g_active_fused_slot] setComputePipelineState:ps];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)xRef offset:0 atIndex:0];
    [g_fused_enc[g_active_fused_slot] setBuffer:hdBuf offset:0 atIndex:1];
    [g_fused_enc[g_active_fused_slot] setBuffer:nhBuf offset:0 atIndex:2];
    [g_fused_enc[g_active_fused_slot] setBuffer:thBuf offset:0 atIndex:3];
    [g_fused_enc[g_active_fused_slot] setBuffer:stBuf offset:0 atIndex:4];
    int nPairs = nHeads * (headDim / 2);
    [g_fused_enc[g_active_fused_slot] dispatchThreads:MTLSizeMake(nPairs, seqLen, 1)
           threadsPerThreadgroup:MTLSizeMake(nPairs < 256 ? nPairs : 256, 1, 1)];
}

void mtl_fused_copy(void* dstRef, void* srcRef, int n) {
    // GPU memcpy: dst[i] = src[i] using add_inplace trick — zero dst first then add
    // Actually simpler: just memcpy on the encoder using blit
    id<MTLBuffer> dst = (__bridge id<MTLBuffer>)dstRef;
    id<MTLBuffer> src = (__bridge id<MTLBuffer>)srcRef;
    [g_fused_enc[g_active_fused_slot] endEncoding];
    id<MTLBlitCommandEncoder> blit = [g_fused_cmd[g_active_fused_slot] blitCommandEncoder];
    [blit copyFromBuffer:src sourceOffset:0 toBuffer:dst destinationOffset:0 size:n*sizeof(float)];
    [blit endEncoding];
    g_fused_enc[g_active_fused_slot] = [g_fused_cmd[g_active_fused_slot] computeCommandEncoder];
}

void mtl_fused_bias_add(void* xRef, void* bRef, int n, int cols) {
    uint32_t ucols = (uint32_t)cols;
    id<MTLBuffer> colsBuf = [g_device newBufferWithBytes:&ucols length:4 options:MTLResourceStorageModeShared];
    [g_fused_enc[g_active_fused_slot] setComputePipelineState:g_ps_bias_add];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)xRef offset:0 atIndex:0];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)bRef offset:0 atIndex:1];
    [g_fused_enc[g_active_fused_slot] setBuffer:colsBuf offset:0 atIndex:2];
    NSUInteger tpg = g_ps_bias_add.maxTotalThreadsPerThreadgroup;
    if (tpg > 1024) tpg = 1024;
    [g_fused_enc[g_active_fused_slot] dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
}

// === Coil compute dispatches — all queue-aware ===

// Tiled GEMM: C = A @ B^T. Threadgroup=(TILE,TILE), grid=(ceil(N/TILE), ceil(M/TILE)).
void mtl_gemm_bt_q(void* aRef, void* bRef, void* cRef, int M, int K, int N, int queueIdx) {
    @autoreleasepool {
    id<MTLCommandQueue> q = (queueIdx == 0) ? g_queue : g_queue2;
    uint32_t um = M, uk = K, un = N;
    id<MTLBuffer> mBuf = [g_device newBufferWithBytes:&um length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> kBuf = [g_device newBufferWithBytes:&uk length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> nBuf = [g_device newBufferWithBytes:&un length:4 options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setBuffer:(__bridge id<MTLBuffer>)aRef offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)bRef offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)cRef offset:0 atIndex:2];
    [enc setBuffer:mBuf offset:0 atIndex:3];
    [enc setBuffer:kBuf offset:0 atIndex:4];
    [enc setBuffer:nBuf offset:0 atIndex:5];
    if (g_use_metal4_gemm) {
        [enc setComputePipelineState:g_ps_gemm4_bt];
        NSUInteger simdWidth = g_ps_gemm4_bt.threadExecutionWidth;
        [enc dispatchThreadgroups:MTLSizeMake((N+31)/32, (M+63)/64, 1) threadsPerThreadgroup:MTLSizeMake(simdWidth * 4, 1, 1)];
    } else {
        [enc setComputePipelineState:g_ps_gemm_bt];
        [enc dispatchThreadgroups:MTLSizeMake((N+31)/32, (M+31)/32, 1) threadsPerThreadgroup:MTLSizeMake(32, 32, 1)];
    }
    [enc endEncoding];
    [cmd commit]; [cmd waitUntilCompleted];
    } // @autoreleasepool
}

// FP32 GEMM: C = A @ B^T using Metal 4 FP32 kernel or MPS fallback. Queue-aware.
void mtl_gemm_f32_bt_q(void* aRef, void* bRef, void* cRef, int M, int K, int N, int queueIdx) {
    @autoreleasepool {
    id<MTLCommandQueue> q = g_queues[queueIdx < MTL_MAX_QUEUES ? queueIdx : 0];
    if (g_ps_gemm4f_bt) {
        uint32_t um = M, uk = K, un = N;
        id<MTLBuffer> mBuf = [g_device newBufferWithBytes:&um length:4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> kBuf = [g_device newBufferWithBytes:&uk length:4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> nBuf = [g_device newBufferWithBytes:&un length:4 options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cmd = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_ps_gemm4f_bt];
        [enc setBuffer:(__bridge id<MTLBuffer>)aRef offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)bRef offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)cRef offset:0 atIndex:2];
        [enc setBuffer:mBuf offset:0 atIndex:3];
        [enc setBuffer:kBuf offset:0 atIndex:4];
        [enc setBuffer:nBuf offset:0 atIndex:5];
        NSUInteger simdWidth = g_ps_gemm4f_bt.threadExecutionWidth;
        [enc dispatchThreadgroups:MTLSizeMake((N+31)/32, (M+63)/64, 1) threadsPerThreadgroup:MTLSizeMake(simdWidth * 4, 1, 1)];
        [enc endEncoding];
        [cmd commit]; [cmd waitUntilCompleted];
    } else {
        // MPS fallback
        mtl_matmul_raw(aRef, bRef, cRef, M, K, N, queueIdx);
    }
    } // @autoreleasepool
}

// Tiled GEMM accumulate: C += A @ B^T.
void mtl_gemm_bt_acc_q(void* aRef, void* bRef, void* cRef, int M, int K, int N, int queueIdx) {
    @autoreleasepool {
    id<MTLCommandQueue> q = (queueIdx == 0) ? g_queue : g_queue2;
    uint32_t um = M, uk = K, un = N;
    id<MTLBuffer> mBuf = [g_device newBufferWithBytes:&um length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> kBuf = [g_device newBufferWithBytes:&uk length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> nBuf = [g_device newBufferWithBytes:&un length:4 options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_ps_gemm_bt_acc];
    [enc setBuffer:(__bridge id<MTLBuffer>)aRef offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)bRef offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)cRef offset:0 atIndex:2];
    [enc setBuffer:mBuf offset:0 atIndex:3];
    [enc setBuffer:kBuf offset:0 atIndex:4];
    [enc setBuffer:nBuf offset:0 atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake((N+31)/32, (M+31)/32, 1) threadsPerThreadgroup:MTLSizeMake(32, 32, 1)];
    [enc endEncoding];
    [cmd commit]; [cmd waitUntilCompleted];
    } // @autoreleasepool
}

// GEMM: C = A^T @ B (for weight gradients).
void mtl_gemm_tn_q(void* aRef, void* bRef, void* cRef, int M, int K, int N, int queueIdx) {
    @autoreleasepool {
    id<MTLCommandQueue> q = (queueIdx == 0) ? g_queue : g_queue2;
    uint32_t um = M, uk = K, un = N;
    id<MTLBuffer> mBuf = [g_device newBufferWithBytes:&um length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> kBuf = [g_device newBufferWithBytes:&uk length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> nBuf = [g_device newBufferWithBytes:&un length:4 options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setBuffer:(__bridge id<MTLBuffer>)aRef offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)bRef offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)cRef offset:0 atIndex:2];
    [enc setBuffer:mBuf offset:0 atIndex:3];
    [enc setBuffer:kBuf offset:0 atIndex:4];
    [enc setBuffer:nBuf offset:0 atIndex:5];
    if (g_use_metal4_gemm) {
        [enc setComputePipelineState:g_ps_gemm4_tn];
        NSUInteger simdWidth = g_ps_gemm4_tn.threadExecutionWidth;
        [enc dispatchThreadgroups:MTLSizeMake((N+31)/32, (M+63)/64, 1) threadsPerThreadgroup:MTLSizeMake(simdWidth * 4, 1, 1)];
    } else {
        [enc setComputePipelineState:g_ps_gemm_tn];
        [enc dispatchThreadgroups:MTLSizeMake((N+31)/32, (M+31)/32, 1) threadsPerThreadgroup:MTLSizeMake(32, 32, 1)];
    }
    [enc endEncoding];
    [cmd commit]; [cmd waitUntilCompleted];
    } // @autoreleasepool
}

// GEMM: C = A @ B (no transpose).
void mtl_gemm_nn_q(void* aRef, void* bRef, void* cRef, int M, int K, int N, int queueIdx) {
    @autoreleasepool {
    id<MTLCommandQueue> q = (queueIdx == 0) ? g_queue : g_queue2;
    uint32_t um = M, uk = K, un = N;
    id<MTLBuffer> mBuf = [g_device newBufferWithBytes:&um length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> kBuf = [g_device newBufferWithBytes:&uk length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> nBuf = [g_device newBufferWithBytes:&un length:4 options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setBuffer:(__bridge id<MTLBuffer>)aRef offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)bRef offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)cRef offset:0 atIndex:2];
    [enc setBuffer:mBuf offset:0 atIndex:3];
    [enc setBuffer:kBuf offset:0 atIndex:4];
    [enc setBuffer:nBuf offset:0 atIndex:5];
    if (g_use_metal4_gemm) {
        [enc setComputePipelineState:g_ps_gemm4_nn];
        NSUInteger simdWidth = g_ps_gemm4_nn.threadExecutionWidth;
        [enc dispatchThreadgroups:MTLSizeMake((N+31)/32, (M+63)/64, 1) threadsPerThreadgroup:MTLSizeMake(simdWidth * 4, 1, 1)];
    } else {
        [enc setComputePipelineState:g_ps_gemm_nn];
        [enc dispatchThreadgroups:MTLSizeMake((N+31)/32, (M+31)/32, 1) threadsPerThreadgroup:MTLSizeMake(32, 32, 1)];
    }
    [enc endEncoding];
    [cmd commit]; [cmd waitUntilCompleted];
    } // @autoreleasepool
}

// RMSNorm with scale save. In-place normalization, saves 1/rms for backward.
void mtl_rmsnorm_save_q2(void* xRef, void* wRef, void* scaleRef, int seqLen, int dim, int queueIdx) {
    @autoreleasepool {
    id<MTLCommandQueue> q = (queueIdx == 0) ? g_queue : g_queue2;
    uint32_t udim = dim;
    float eps = 1e-6f;
    id<MTLBuffer> dimBuf = [g_device newBufferWithBytes:&udim length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> epsBuf = [g_device newBufferWithBytes:&eps length:4 options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_ps_rmsnorm_save];
    [enc setBuffer:(__bridge id<MTLBuffer>)xRef     offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)wRef     offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)scaleRef offset:0 atIndex:2];
    [enc setBuffer:dimBuf offset:0 atIndex:3];
    [enc setBuffer:epsBuf offset:0 atIndex:4];
    NSUInteger tpg = g_ps_rmsnorm_save.maxTotalThreadsPerThreadgroup;
    if (tpg > (NSUInteger)dim) tpg = (NSUInteger)dim;
    tpg = (tpg / 32) * 32;
    if (tpg == 0) tpg = 32;
    [enc dispatchThreadgroups:MTLSizeMake(seqLen, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    [enc endEncoding];
    [cmd commit]; [cmd waitUntilCompleted];
    } // @autoreleasepool
}

// Fused causal attention: dispatch nHeads threadgroups, seqLen threads each.
void mtl_fused_attention_q(void* qRef, void* kRef, void* vRef, void* outRef, void* scoresRef,
                           int dim, int kvDim, int headDim, int nHeads, int nKVHeads, int seqLen, int queueIdx) {
    @autoreleasepool {
    id<MTLCommandQueue> q = (queueIdx == 0) ? g_queue : g_queue2;
    uint32_t ud = dim, ukv = kvDim, uhd = headDim, unh = nHeads, unkv = nKVHeads, un = seqLen;
    id<MTLBuffer> dBuf   = [g_device newBufferWithBytes:&ud   length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> kvBuf  = [g_device newBufferWithBytes:&ukv  length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> hdBuf  = [g_device newBufferWithBytes:&uhd  length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> nhBuf  = [g_device newBufferWithBytes:&unh  length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> nkvBuf = [g_device newBufferWithBytes:&unkv length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> nBuf   = [g_device newBufferWithBytes:&un   length:4 options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_ps_fused_attn];
    [enc setBuffer:(__bridge id<MTLBuffer>)qRef      offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)kRef      offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)vRef      offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)outRef    offset:0 atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)scoresRef offset:0 atIndex:4];
    [enc setBuffer:dBuf   offset:0 atIndex:5];
    [enc setBuffer:kvBuf  offset:0 atIndex:6];
    [enc setBuffer:hdBuf  offset:0 atIndex:7];
    [enc setBuffer:nhBuf  offset:0 atIndex:8];
    [enc setBuffer:nkvBuf offset:0 atIndex:9];
    [enc setBuffer:nBuf   offset:0 atIndex:10];
    // nHeads threadgroups, seqLen threads per threadgroup
    NSUInteger tpg = (NSUInteger)seqLen;
    if (tpg > g_ps_fused_attn.maxTotalThreadsPerThreadgroup)
        tpg = g_ps_fused_attn.maxTotalThreadsPerThreadgroup;
    [enc dispatchThreadgroups:MTLSizeMake(nHeads, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    [enc endEncoding];
    [cmd commit]; [cmd waitUntilCompleted];
    } // @autoreleasepool
}

// Fused causal attention backward.
void mtl_fused_attention_bwd_q(void* dOutRef, void* qRef, void* kRef, void* vRef, void* scoresRef,
                               void* dQRef, void* dKRef, void* dVRef,
                               int dim, int kvDim, int headDim, int nHeads, int nKVHeads, int seqLen, int queueIdx) {
    @autoreleasepool {
    id<MTLCommandQueue> q = (queueIdx == 0) ? g_queue : g_queue2;
    uint32_t ud = dim, ukv = kvDim, uhd = headDim, unh = nHeads, unkv = nKVHeads, un = seqLen;
    id<MTLBuffer> dBuf   = [g_device newBufferWithBytes:&ud   length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> kvBuf  = [g_device newBufferWithBytes:&ukv  length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> hdBuf  = [g_device newBufferWithBytes:&uhd  length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> nhBuf  = [g_device newBufferWithBytes:&unh  length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> nkvBuf = [g_device newBufferWithBytes:&unkv length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> nBuf   = [g_device newBufferWithBytes:&un   length:4 options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_ps_fused_attn_bwd];
    [enc setBuffer:(__bridge id<MTLBuffer>)dOutRef   offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)qRef      offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)kRef      offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)vRef      offset:0 atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)scoresRef offset:0 atIndex:4];
    [enc setBuffer:(__bridge id<MTLBuffer>)dQRef     offset:0 atIndex:5];
    [enc setBuffer:(__bridge id<MTLBuffer>)dKRef     offset:0 atIndex:6];
    [enc setBuffer:(__bridge id<MTLBuffer>)dVRef     offset:0 atIndex:7];
    [enc setBuffer:dBuf   offset:0 atIndex:8];
    [enc setBuffer:kvBuf  offset:0 atIndex:9];
    [enc setBuffer:hdBuf  offset:0 atIndex:10];
    [enc setBuffer:nhBuf  offset:0 atIndex:11];
    [enc setBuffer:nkvBuf offset:0 atIndex:12];
    [enc setBuffer:nBuf   offset:0 atIndex:13];
    NSUInteger tpg = (NSUInteger)seqLen;
    if (tpg > g_ps_fused_attn_bwd.maxTotalThreadsPerThreadgroup)
        tpg = g_ps_fused_attn_bwd.maxTotalThreadsPerThreadgroup;
    [enc dispatchThreadgroups:MTLSizeMake(nHeads, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    [enc endEncoding];
    [cmd commit]; [cmd waitUntilCompleted];
    } // @autoreleasepool
}

// Bias add: x[i] += bias[i % cols].
void mtl_bias_add_q(void* xRef, void* bRef, int n, int cols, int queueIdx) {
    @autoreleasepool {
    id<MTLCommandQueue> q = (queueIdx == 0) ? g_queue : g_queue2;
    uint32_t ucols = (uint32_t)cols;
    id<MTLBuffer> colsBuf = [g_device newBufferWithBytes:&ucols length:4 options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_ps_bias_add];
    [enc setBuffer:(__bridge id<MTLBuffer>)xRef offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)bRef offset:0 atIndex:1];
    [enc setBuffer:colsBuf offset:0 atIndex:2];
    NSUInteger tpg = g_ps_bias_add.maxTotalThreadsPerThreadgroup;
    if (tpg > 1024) tpg = 1024;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    [enc endEncoding];
    [cmd commit]; [cmd waitUntilCompleted];
    } // @autoreleasepool
}

// RoPE forward: rotate Q or K pairs.
void mtl_rope_fwd_q(void* xRef, int headDim, int nHeads, float theta, int stride, int seqLen, int queueIdx) {
    @autoreleasepool {
    id<MTLCommandQueue> q = (queueIdx == 0) ? g_queue : g_queue2;
    uint32_t uhd = headDim, unh = nHeads, ust = stride;
    id<MTLBuffer> hdBuf  = [g_device newBufferWithBytes:&uhd   length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> nhBuf  = [g_device newBufferWithBytes:&unh   length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> thBuf  = [g_device newBufferWithBytes:&theta length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> stBuf  = [g_device newBufferWithBytes:&ust   length:4 options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_ps_rope_fwd];
    [enc setBuffer:(__bridge id<MTLBuffer>)xRef offset:0 atIndex:0];
    [enc setBuffer:hdBuf offset:0 atIndex:1];
    [enc setBuffer:nhBuf offset:0 atIndex:2];
    [enc setBuffer:thBuf offset:0 atIndex:3];
    [enc setBuffer:stBuf offset:0 atIndex:4];
    int nPairs = nHeads * (headDim / 2);
    [enc dispatchThreads:MTLSizeMake(nPairs, seqLen, 1) threadsPerThreadgroup:MTLSizeMake(nPairs < 256 ? nPairs : 256, 1, 1)];
    [enc endEncoding];
    [cmd commit]; [cmd waitUntilCompleted];
    } // @autoreleasepool
}

// RoPE backward.
void mtl_rope_bwd_q(void* dxRef, int headDim, int nHeads, float theta, int stride, int seqLen, int queueIdx) {
    @autoreleasepool {
    id<MTLCommandQueue> q = (queueIdx == 0) ? g_queue : g_queue2;
    uint32_t uhd = headDim, unh = nHeads, ust = stride;
    id<MTLBuffer> hdBuf  = [g_device newBufferWithBytes:&uhd   length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> nhBuf  = [g_device newBufferWithBytes:&unh   length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> thBuf  = [g_device newBufferWithBytes:&theta length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> stBuf  = [g_device newBufferWithBytes:&ust   length:4 options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_ps_rope_bwd];
    [enc setBuffer:(__bridge id<MTLBuffer>)dxRef offset:0 atIndex:0];
    [enc setBuffer:hdBuf offset:0 atIndex:1];
    [enc setBuffer:nhBuf offset:0 atIndex:2];
    [enc setBuffer:thBuf offset:0 atIndex:3];
    [enc setBuffer:stBuf offset:0 atIndex:4];
    int nPairs = nHeads * (headDim / 2);
    [enc dispatchThreads:MTLSizeMake(nPairs, seqLen, 1) threadsPerThreadgroup:MTLSizeMake(nPairs < 256 ? nPairs : 256, 1, 1)];
    [enc endEncoding];
    [cmd commit]; [cmd waitUntilCompleted];
    } // @autoreleasepool
}

// === Queue-aware kernel dispatches for coiled pipeline ===
// These take a queueIdx (0=forward coil, 1=backward coil) and dispatch
// synchronously on the specified queue. Used for per-layer ops outside MPSGraph.

// RMSNorm with scale output: normed[i] = x[i] / rms * w[i], scale[pos] = 1/rms.
// x modified in-place (becomes normed). scale saved for backward.
void mtl_rmsnorm_save_q(void* xRef, void* weightRef, void* scaleRef, int seqLen, int dim, int queueIdx) {
    @autoreleasepool {
    id<MTLCommandQueue> q = (queueIdx == 0) ? g_queue : g_queue2;
    // RMSNorm: reuse existing kernel (in-place), then compute scale on CPU
    // For now: dispatch existing kernel on specified queue
    uint32_t dimVal = (uint32_t)dim;
    float epsVal = 1e-6f;
    id<MTLBuffer> dimBuf = [g_device newBufferWithBytes:&dimVal length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> epsBuf = [g_device newBufferWithBytes:&epsVal length:sizeof(float) options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_ps_rmsnorm];
    [enc setBuffer:(__bridge id<MTLBuffer>)xRef offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)weightRef offset:0 atIndex:1];
    [enc setBuffer:dimBuf offset:0 atIndex:2];
    [enc setBuffer:epsBuf offset:0 atIndex:3];

    NSUInteger tpg = g_ps_rmsnorm.maxTotalThreadsPerThreadgroup;
    if (tpg > (NSUInteger)dim) tpg = (NSUInteger)dim;
    tpg = (tpg / 32) * 32;
    if (tpg == 0) tpg = 32;
    [enc dispatchThreadgroups:MTLSizeMake(seqLen, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    } // @autoreleasepool
}

// Add in-place on specified queue: a[i] += b[i]
void mtl_add_inplace_q(void* aRef, void* bRef, int n, int queueIdx) {
    @autoreleasepool {
    id<MTLCommandQueue> q = (queueIdx == 0) ? g_queue : g_queue2;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_ps_add_inplace];
    [enc setBuffer:(__bridge id<MTLBuffer>)aRef offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)bRef offset:0 atIndex:1];
    NSUInteger tpg = g_ps_add_inplace.maxTotalThreadsPerThreadgroup;
    if (tpg > 1024) tpg = 1024;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    } // @autoreleasepool
}

// SiLU gate mul on specified queue: out[i] = gate[i]*sigmoid(gate[i]) * up[i]
void mtl_silu_gate_mul_q(void* gateRef, void* upRef, void* outRef, int n, int queueIdx) {
    @autoreleasepool {
    id<MTLCommandQueue> q = (queueIdx == 0) ? g_queue : g_queue2;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_ps_silu_gate_mul];
    [enc setBuffer:(__bridge id<MTLBuffer>)gateRef offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)upRef offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
    NSUInteger tpg = g_ps_silu_gate_mul.maxTotalThreadsPerThreadgroup;
    if (tpg > 1024) tpg = 1024;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    } // @autoreleasepool
}

// SiLU gate backward on specified queue
void mtl_silu_gate_backward_q(void* dOutRef, void* gatePreRef, void* upOutRef,
                              void* gateActRef, void* dGatePreRef, void* dUpRef,
                              int n, int queueIdx) {
    @autoreleasepool {
    id<MTLCommandQueue> q = (queueIdx == 0) ? g_queue : g_queue2;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_ps_silu_gate_backward];
    [enc setBuffer:(__bridge id<MTLBuffer>)dOutRef    offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)gatePreRef offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)upOutRef   offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)gateActRef offset:0 atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)dGatePreRef offset:0 atIndex:4];
    [enc setBuffer:(__bridge id<MTLBuffer>)dUpRef     offset:0 atIndex:5];
    NSUInteger tpg = g_ps_silu_gate_backward.maxTotalThreadsPerThreadgroup;
    if (tpg > 1024) tpg = 1024;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    } // @autoreleasepool
}

// Zero on specified queue
void mtl_zero_q(void* ptrRef, int n, int queueIdx) {
    @autoreleasepool {
    id<MTLCommandQueue> q = (queueIdx == 0) ? g_queue : g_queue2;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_ps_zero_mem];
    [enc setBuffer:(__bridge id<MTLBuffer>)ptrRef offset:0 atIndex:0];
    NSUInteger tpg = g_ps_zero_mem.maxTotalThreadsPerThreadgroup;
    if (tpg > 1024) tpg = 1024;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    } // @autoreleasepool
}

// Needle on specified queue (for backward coil)
void mtl_helix_needle_q(
    void* dataRef, void* scalesRef, void* gradRef,
    void* momRef, void* velRef, void* maskRef,
    float lr, float beta1, float beta2, float bc1, float bc2,
    float eps, float wd, int n, int cols, int queueIdx) {
    @autoreleasepool {
    id<MTLCommandQueue> q = (queueIdx == 0) ? g_queue : g_queue2;

    uint32_t ucols = (uint32_t)cols;
    id<MTLBuffer> lrBuf   = [g_device newBufferWithBytes:&lr    length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> b1Buf   = [g_device newBufferWithBytes:&beta1 length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> b2Buf   = [g_device newBufferWithBytes:&beta2 length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bc1Buf  = [g_device newBufferWithBytes:&bc1   length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bc2Buf  = [g_device newBufferWithBytes:&bc2   length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> epsBuf  = [g_device newBufferWithBytes:&eps   length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> wdBuf   = [g_device newBufferWithBytes:&wd    length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> colsBuf = [g_device newBufferWithBytes:&ucols length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_ps_needle];
    [enc setBuffer:(__bridge id<MTLBuffer>)dataRef   offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)scalesRef offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)gradRef   offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)momRef    offset:0 atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)velRef    offset:0 atIndex:4];
    [enc setBuffer:(__bridge id<MTLBuffer>)maskRef   offset:0 atIndex:5];
    [enc setBuffer:lrBuf   offset:0 atIndex:6];
    [enc setBuffer:b1Buf   offset:0 atIndex:7];
    [enc setBuffer:b2Buf   offset:0 atIndex:8];
    [enc setBuffer:bc1Buf  offset:0 atIndex:9];
    [enc setBuffer:bc2Buf  offset:0 atIndex:10];
    [enc setBuffer:epsBuf  offset:0 atIndex:11];
    [enc setBuffer:wdBuf   offset:0 atIndex:12];
    [enc setBuffer:colsBuf offset:0 atIndex:13];

    NSUInteger tpg = g_ps_needle.maxTotalThreadsPerThreadgroup;
    if (tpg > 1024) tpg = 1024;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    } // @autoreleasepool
}

// Dequant on specified queue
void mtl_dequant_int8_q(void* srcRef, void* scalesRef, void* dstRef, int n, int cols, int queueIdx) {
    @autoreleasepool {
    id<MTLCommandQueue> q = (queueIdx == 0) ? g_queue : g_queue2;
    uint32_t ucols = (uint32_t)cols;
    id<MTLBuffer> colsBuf = [g_device newBufferWithBytes:&ucols length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_ps_dequant_int8];
    [enc setBuffer:(__bridge id<MTLBuffer>)srcRef    offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)scalesRef offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)dstRef    offset:0 atIndex:2];
    [enc setBuffer:colsBuf offset:0 atIndex:3];

    NSUInteger tpg = g_ps_dequant_int8.maxTotalThreadsPerThreadgroup;
    if (tpg > 1024) tpg = 1024;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    } // @autoreleasepool
}

// === Fused training dispatch: gradient clipping + needle optimizer ===
// These encode into the active fused compute encoder (g_fused_enc[slot]).

void mtl_fused_zero_scalar(void* bufRef) {
    [g_fused_enc[g_active_fused_slot] setComputePipelineState:g_ps_zero_mem];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)bufRef offset:0 atIndex:0];
    [g_fused_enc[g_active_fused_slot] dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
}

void mtl_fused_barrier_buffers(void) {
    [g_fused_enc[g_active_fused_slot] memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

void mtl_fused_grad_norm_sq(void* gradRef, void* sumSqRef, int n) {
    [g_fused_enc[g_active_fused_slot] setComputePipelineState:g_ps_grad_norm];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)gradRef  offset:0 atIndex:0];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)sumSqRef offset:0 atIndex:1];
    NSUInteger tpg = g_ps_grad_norm.maxTotalThreadsPerThreadgroup;
    if (tpg > 1024) tpg = 1024;
    [g_fused_enc[g_active_fused_slot] dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
}

void mtl_fused_grad_clip_scale(void* gradRef, void* sumSqRef, float maxNorm, int n) {
    [g_fused_enc[g_active_fused_slot] setComputePipelineState:g_ps_grad_clip];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)gradRef  offset:0 atIndex:0];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)sumSqRef offset:0 atIndex:1];
    [g_fused_enc[g_active_fused_slot] setBytes:&maxNorm length:sizeof(float) atIndex:2];
    NSUInteger tpg = g_ps_grad_clip.maxTotalThreadsPerThreadgroup;
    if (tpg > 1024) tpg = 1024;
    [g_fused_enc[g_active_fused_slot] dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
}

// Fused dequant INT8 + delta → FP32 live weight buffer.
void mtl_fused_dequant_delta(void* srcRef, void* scalesRef, void* deltaRef, void* dstRef, int n, int cols) {
    uint32_t ucols = (uint32_t)cols;
    [g_fused_enc[g_active_fused_slot] setComputePipelineState:g_ps_dequant_delta];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)srcRef    offset:0 atIndex:0];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)scalesRef offset:0 atIndex:1];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)deltaRef  offset:0 atIndex:2];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)dstRef    offset:0 atIndex:3];
    [g_fused_enc[g_active_fused_slot] setBytes:&ucols length:sizeof(uint32_t) atIndex:4];
    NSUInteger tpg = g_ps_dequant_delta.maxTotalThreadsPerThreadgroup;
    if (tpg > 1024) tpg = 1024;
    [g_fused_enc[g_active_fused_slot] dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
}

// Sparse dequant: only updates rows where mask[row] != 0.
void mtl_fused_dequant_delta_sparse(void* srcRef, void* scalesRef, void* deltaRef, void* dstRef, void* maskRef, int n, int cols) {
    uint32_t ucols = (uint32_t)cols;
    [g_fused_enc[g_active_fused_slot] setComputePipelineState:g_ps_dequant_delta_sparse];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)srcRef    offset:0 atIndex:0];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)scalesRef offset:0 atIndex:1];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)deltaRef  offset:0 atIndex:2];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)dstRef    offset:0 atIndex:3];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)maskRef   offset:0 atIndex:4];
    [g_fused_enc[g_active_fused_slot] setBytes:&ucols length:sizeof(uint32_t) atIndex:5];
    NSUInteger tpg = g_ps_dequant_delta_sparse.maxTotalThreadsPerThreadgroup;
    if (tpg > 1024) tpg = 1024;
    [g_fused_enc[g_active_fused_slot] dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
}

// Async commit: end encoding and commit but don't wait. Returns immediately.
void mtl_fused_end_async(void) {
    int s = g_active_fused_slot;
    if (!g_fused_enc[s]) return;
    [g_fused_enc[s] endEncoding];
    [g_fused_cmd[s] commit];
    g_fused_enc[s] = nil;
    // Keep g_fused_cmd[s] alive for mtl_fused_wait
}

// Wait for the most recently committed fused command buffer.
void mtl_fused_wait(void) {
    int s = g_active_fused_slot;
    if (g_fused_cmd[s]) {
        [g_fused_cmd[s] waitUntilCompleted];
        g_fused_cmd[s] = nil;
    }
}

// Fused needle: encode helix_needle into the active fused encoder.
void mtl_fused_needle(
    void* dataRef, void* scalesRef, void* gradRef,
    void* momRef, void* velRef, void* maskRef, void* deltaRef,
    float lr, float beta1, float beta2, float bc1, float bc2,
    float eps, float wd, int n, int cols) {
    id<MTLComputeCommandEncoder> enc = g_fused_enc[g_active_fused_slot];
    [enc setComputePipelineState:g_ps_needle];
    [enc setBuffer:(__bridge id<MTLBuffer>)dataRef   offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)scalesRef offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)gradRef   offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)momRef    offset:0 atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)velRef    offset:0 atIndex:4];
    [enc setBuffer:(__bridge id<MTLBuffer>)maskRef   offset:0 atIndex:5];
    [enc setBuffer:(__bridge id<MTLBuffer>)deltaRef  offset:0 atIndex:6];
    [enc setBytes:&lr    length:sizeof(float) atIndex:7];
    [enc setBytes:&beta1 length:sizeof(float) atIndex:8];
    [enc setBytes:&beta2 length:sizeof(float) atIndex:9];
    [enc setBytes:&bc1   length:sizeof(float) atIndex:10];
    [enc setBytes:&bc2   length:sizeof(float) atIndex:11];
    [enc setBytes:&eps   length:sizeof(float) atIndex:12];
    [enc setBytes:&wd    length:sizeof(float) atIndex:13];
    uint32_t ucols = (uint32_t)cols;
    [enc setBytes:&ucols length:sizeof(uint32_t) atIndex:14];
    NSUInteger tpg = g_ps_needle.maxTotalThreadsPerThreadgroup;
    if (tpg > 1024) tpg = 1024;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
}

// Fused needle paired: encode helix_needle_paired into the active fused encoder.
void mtl_fused_needle_paired(
    void* d1Ref, void* d2Ref,
    void* s1Ref, void* s2Ref,
    void* g1Ref, void* g2Ref,
    void* m1Ref, void* m2Ref,
    void* v1Ref, void* v2Ref,
    void* maskRef, void* delta1Ref, void* delta2Ref,
    float lr, float beta1, float beta2, float bc1, float bc2,
    float eps, float wd,
    float backbone1, float glyco1, float hbond1,
    float hbond2, float glyco2, float backbone2,
    float bondStrength, int n, int cols) {
    id<MTLComputeCommandEncoder> enc = g_fused_enc[g_active_fused_slot];
    [enc setComputePipelineState:g_ps_needle_paired];
    [enc setBuffer:(__bridge id<MTLBuffer>)d1Ref     offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)d2Ref     offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)s1Ref     offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)s2Ref     offset:0 atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)g1Ref     offset:0 atIndex:4];
    [enc setBuffer:(__bridge id<MTLBuffer>)g2Ref     offset:0 atIndex:5];
    [enc setBuffer:(__bridge id<MTLBuffer>)m1Ref     offset:0 atIndex:6];
    [enc setBuffer:(__bridge id<MTLBuffer>)m2Ref     offset:0 atIndex:7];
    [enc setBuffer:(__bridge id<MTLBuffer>)v1Ref     offset:0 atIndex:8];
    [enc setBuffer:(__bridge id<MTLBuffer>)v2Ref     offset:0 atIndex:9];
    [enc setBuffer:(__bridge id<MTLBuffer>)maskRef   offset:0 atIndex:10];
    [enc setBuffer:(__bridge id<MTLBuffer>)delta1Ref offset:0 atIndex:11];
    [enc setBuffer:(__bridge id<MTLBuffer>)delta2Ref offset:0 atIndex:12];
    [enc setBytes:&lr          length:sizeof(float) atIndex:13];
    [enc setBytes:&beta1       length:sizeof(float) atIndex:14];
    [enc setBytes:&beta2       length:sizeof(float) atIndex:15];
    [enc setBytes:&bc1         length:sizeof(float) atIndex:16];
    [enc setBytes:&bc2         length:sizeof(float) atIndex:17];
    [enc setBytes:&eps         length:sizeof(float) atIndex:18];
    [enc setBytes:&wd          length:sizeof(float) atIndex:19];
    [enc setBytes:&backbone1   length:sizeof(float) atIndex:20];
    [enc setBytes:&glyco1      length:sizeof(float) atIndex:21];
    [enc setBytes:&hbond1      length:sizeof(float) atIndex:22];
    [enc setBytes:&hbond2      length:sizeof(float) atIndex:23];
    [enc setBytes:&glyco2      length:sizeof(float) atIndex:24];
    [enc setBytes:&backbone2   length:sizeof(float) atIndex:25];
    [enc setBytes:&bondStrength length:sizeof(float) atIndex:26];
    uint32_t ucols = (uint32_t)cols;
    [enc setBytes:&ucols       length:sizeof(uint32_t) atIndex:27];
    NSUInteger tpg = g_ps_needle_paired.maxTotalThreadsPerThreadgroup;
    if (tpg > 1024) tpg = 1024;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
}

// Inference compute-shader stubs — satisfy linker.
// Real implementations will replace these.
int mtl_fused_build(int dim, int kvDim, int headDim,
                    int nHeads, int nKVHeads, int ffnDim,
                    int vocabSize, int nLayers, int maxSeq) { return -1; }
int mtl_fused_num_weights(void) { return 0; }
int mtl_fused_set_weight(int idx, const float* data, int nFloats) { return -1; }
int mtl_fused_step(const float* hiddenIn, const float* cosData, const float* sinData,
                   int pos, float* logitsOut) { return -1; }
