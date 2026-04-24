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
"                      device const uint* cols [[buffer(3)]],\n"
"                      uint2 gid [[thread_position_in_grid]]) {\n"
"    uint i = gid.y;\n"
"    uint j = gid.x;\n"
"    G[i * cols[0] + j] += a[i] * b[j];\n"
"}\n"
"\n"
"// RMSNorm: x = x / rms * weight, one threadgroup per row\n"
"kernel void rmsnorm(device float* x [[buffer(0)]],\n"
"                    device const float* weight [[buffer(1)]],\n"
"                    device const uint* dim [[buffer(2)]],\n"
"                    device const float* eps [[buffer(3)]],\n"
"                    uint row [[threadgroup_position_in_grid]],\n"
"                    uint tid [[thread_index_in_threadgroup]],\n"
"                    uint tpg [[threads_per_threadgroup]]) {\n"
"    uint base = row * dim[0];\n"
"    // Compute sum of squares (strided reduction)\n"
"    float ss = 0.0f;\n"
"    for (uint i = tid; i < dim[0]; i += tpg) {\n"
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
"        final_scale = rsqrt(ss / float(dim[0]) + eps[0]);\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float s = final_scale;\n"
"    for (uint i = tid; i < dim[0]; i += tpg) {\n"
"        x[base + i] = x[base + i] * s * weight[i];\n"
"    }\n"
"}\n"
"\n"
"// AdamW update — one thread per parameter\n"
"kernel void adamw(device float* param [[buffer(0)]],\n"
"                  device const float* grad [[buffer(1)]],\n"
"                  device float* m [[buffer(2)]],\n"
"                  device float* v [[buffer(3)]],\n"
"                  device const float* lr [[buffer(4)]],\n"
"                  device const float* beta1 [[buffer(5)]],\n"
"                  device const float* beta2 [[buffer(6)]],\n"
"                  device const float* bc1 [[buffer(7)]],\n"
"                  device const float* bc2 [[buffer(8)]],\n"
"                  device const float* eps [[buffer(9)]],\n"
"                  device const float* wd [[buffer(10)]],\n"
"                  uint id [[thread_position_in_grid]]) {\n"
"    float g = grad[id];\n"
"    float m_new = beta1[0] * m[id] + (1.0f - beta1[0]) * g;\n"
"    float v_new = beta2[0] * v[id] + (1.0f - beta2[0]) * g * g;\n"
"    m[id] = m_new;\n"
"    v[id] = v_new;\n"
"    float mh = m_new / bc1[0];\n"
"    float vh = v_new / bc2[0];\n"
"    param[id] -= lr[0] * (mh / (sqrt(vh) + eps[0]) + wd[0] * param[id]);\n"
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
"    device const uint* vocabSize [[buffer(3)]],\n"
"    uint pos [[threadgroup_position_in_grid]],\n"
"    uint tid [[thread_index_in_threadgroup]],\n"
"    uint tpg [[threads_per_threadgroup]])\n"
"{\n"
"    uint off = pos * vocabSize[0];\n"
"    int target = targets[pos];\n"
"\n"
"    // Pass 1: find max logit (cooperative)\n"
"    float localMax = -1e30f;\n"
"    for (uint i = tid; i < vocabSize[0]; i += tpg) {\n"
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
"    for (uint i = tid; i < vocabSize[0]; i += tpg) {\n"
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
"    device const uint* dim        [[buffer(4)]],\n"
"    device const uint* vocabSize  [[buffer(5)]],\n"
"    device const uint* nPositions [[buffer(6)]],\n"
"    uint pos [[threadgroup_position_in_grid]],\n"
"    uint tid [[thread_index_in_threadgroup]],\n"
"    uint tpg [[threads_per_threadgroup]])\n"
"{\n"
"    if (pos >= nPositions[0]) return;\n"
"    uint hOff = pos * dim[0];\n"
"\n"
"    // Each thread scans a strided subset of vocab rows\n"
"    float localMax = -1e30f;\n"
"    for (uint v = tid; v < vocabSize[0]; v += tpg) {\n"
"        float dot = 0.0f;\n"
"        for (uint d = 0; d < dim[0]; d++) {\n"
"            dot += hidden[hOff + d] * embed[v * dim[0] + d];\n"
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
"    for (uint v = tid; v < vocabSize[0]; v += tpg) {\n"
"        float dot = 0.0f;\n"
"        for (uint d = 0; d < dim[0]; d++) {\n"
"            dot += hidden[hOff + d] * embed[v * dim[0] + d];\n"
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
"    device const uint* dim          [[buffer(7)]],\n"
"    device const uint* vocabSize    [[buffer(8)]],\n"
"    device const uint* nPositions   [[buffer(9)]],\n"
"    device const float* invN        [[buffer(10)]],  // 1.0 / nPositions\n"
"    uint pos [[threadgroup_position_in_grid]],\n"
"    uint tid [[thread_index_in_threadgroup]],\n"
"    uint tpg [[threads_per_threadgroup]])\n"
"{\n"
"    if (pos >= nPositions[0]) return;\n"
"    uint hOff = pos * dim[0];\n"
"    float mx = maxLogit[pos];\n"
"    float se = sumExp[pos];\n"
"    float invSe = 1.0f / se;\n"
"    int target = targets[pos];\n"
"\n"
"    // Zero dHidden for this position\n"
"    for (uint d = tid; d < dim[0]; d += tpg) {\n"
"        dHidden[hOff + d] = 0.0f;\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"\n"
"    // Loss from target logit\n"
"    if (tid == 0 && target >= 0 && (uint)target < vocabSize[0]) {\n"
"        float targetDot = 0.0f;\n"
"        for (uint d = 0; d < dim[0]; d++) {\n"
"            targetDot += hidden[hOff + d] * embed[target * dim[0] + d];\n"
"        }\n"
"        float posLoss = -(targetDot - mx) + log(se);\n"
"        atomic_fetch_add_explicit(loss, posLoss * invN[0], memory_order_relaxed);\n"
"    }\n"
"\n"
"    // Gradient accumulation: scan all vocab rows, accumulate grad * embed into dHidden.\n"
"    // grad[v] = softmax(v) * invN, grad[target] -= invN\n"
"    // dHidden[pos] += sum_v grad[v] * embed[v]\n"
"    for (uint v = tid; v < vocabSize[0]; v += tpg) {\n"
"        float dot = 0.0f;\n"
"        for (uint d = 0; d < dim[0]; d++) {\n"
"            dot += hidden[hOff + d] * embed[v * dim[0] + d];\n"
"        }\n"
"        float sv = exp(dot - mx) * invSe;  // softmax value\n"
"        float grad = sv * invN[0];\n"
"        if ((int)v == target) grad -= invN[0];\n"
"\n"
"        // Skip near-zero gradients (the sparsity win)\n"
"        if (abs(grad) < 1e-7f) continue;\n"
"\n"
"        // Accumulate grad * embed[v] into dHidden[pos]\n"
"        for (uint d = 0; d < dim[0]; d++) {\n"
"            // Atomic add — multiple threads may write to same dHidden element\n"
"            atomic_fetch_add_explicit(\n"
"                (device atomic_float*)&dHidden[hOff + d],\n"
"                grad * embed[v * dim[0] + d],\n"
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
"    device const uint* dim           [[buffer(5)]],\n"
"    device const uint* nHot          [[buffer(6)]],\n"
"    device const uint* nPositions    [[buffer(7)]],\n"
"    uint pos [[threadgroup_position_in_grid]],\n"
"    uint tid [[thread_index_in_threadgroup]],\n"
"    uint tpg [[threads_per_threadgroup]])\n"
"{\n"
"    if (pos >= nPositions[0]) return;\n"
"    uint hOff = pos * dim[0];\n"
"\n"
"    float localMax = -1e30f;\n"
"    for (uint i = tid; i < nHot[0]; i += tpg) {\n"
"        uint v = (uint)hotIndices[i];\n"
"        float dot = 0.0f;\n"
"        for (uint d = 0; d < dim[0]; d++) {\n"
"            dot += hidden[hOff + d] * embed[v * dim[0] + d];\n"
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
"    for (uint i = tid; i < nHot[0]; i += tpg) {\n"
"        uint v = (uint)hotIndices[i];\n"
"        float dot = 0.0f;\n"
"        for (uint d = 0; d < dim[0]; d++) {\n"
"            dot += hidden[hOff + d] * embed[v * dim[0] + d];\n"
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
"    device const uint* dim           [[buffer(8)]],\n"
"    device const uint* nHot          [[buffer(9)]],\n"
"    device const uint* nPositions    [[buffer(10)]],\n"
"    device const float* invN         [[buffer(11)]],\n"
"    uint pos [[threadgroup_position_in_grid]],\n"
"    uint tid [[thread_index_in_threadgroup]],\n"
"    uint tpg [[threads_per_threadgroup]])\n"
"{\n"
"    if (pos >= nPositions[0]) return;\n"
"    uint hOff = pos * dim[0];\n"
"    float mx = maxLogit[pos];\n"
"    float se = sumExp[pos];\n"
"    float invSe = 1.0f / se;\n"
"    int target = targets[pos];\n"
"\n"
"    for (uint d = tid; d < dim[0]; d += tpg) {\n"
"        dHidden[hOff + d] = 0.0f;\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"\n"
"    if (tid == 0 && target >= 0) {\n"
"        float targetDot = 0.0f;\n"
"        for (uint d = 0; d < dim[0]; d++) {\n"
"            targetDot += hidden[hOff + d] * embed[target * dim[0] + d];\n"
"        }\n"
"        float posLoss = -(targetDot - mx) + log(se);\n"
"        atomic_fetch_add_explicit(loss, posLoss * invN[0], memory_order_relaxed);\n"
"    }\n"
"\n"
"    for (uint i = tid; i < nHot[0]; i += tpg) {\n"
"        uint v = (uint)hotIndices[i];\n"
"        float dot = 0.0f;\n"
"        for (uint d = 0; d < dim[0]; d++) {\n"
"            dot += hidden[hOff + d] * embed[v * dim[0] + d];\n"
"        }\n"
"        float sv = exp(dot - mx) * invSe;\n"
"        float grad = sv * invN[0];\n"
"        if ((int)v == target) grad -= invN[0];\n"
"        if (abs(grad) < 1e-7f) continue;\n"
"        for (uint d = 0; d < dim[0]; d++) {\n"
"            atomic_fetch_add_explicit(\n"
"                (device atomic_float*)&dHidden[hOff + d],\n"
"                grad * embed[v * dim[0] + d],\n"
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
"    device const float* backbone1 [[buffer(8)]],\n"
"    device const float* glyco1    [[buffer(9)]],\n"
"    device const float* hbond1    [[buffer(10)]],\n"
"    device const float* hbond2    [[buffer(11)]],\n"
"    device const float* glyco2    [[buffer(12)]],\n"
"    device const float* backbone2 [[buffer(13)]],\n"
"    device const float* bondStr   [[buffer(14)]],\n"
"    device const float* lr        [[buffer(15)]],\n"
"    device const float* beta1     [[buffer(16)]],\n"
"    device const float* beta2     [[buffer(17)]],\n"
"    device const float* bc1       [[buffer(18)]],\n"
"    device const float* bc2       [[buffer(19)]],\n"
"    device const float* eps       [[buffer(20)]],\n"
"    device const float* wd        [[buffer(21)]],\n"
"    uint id [[thread_position_in_grid]])\n"
"{\n"
"    float ob1 = 1.0f - beta1[0];\n"
"    float ob2 = 1.0f - beta2[0];\n"
"\n"
"    // Read both strands\n"
"    float grad1 = g1[id];\n"
"    float grad2 = g2[id];\n"
"\n"
"    // 6-point rung: gradient flows through the base pair\n"
"    float signal1  = grad1 * glyco1[0];                  // glycosidic bond 1\n"
"    float crossMom = grad2 * hbond1[0] * bondStr[0];        // H-bond: strand2 -> strand1\n"
"    float crossVel = grad1 * hbond2[0] * bondStr[0];        // H-bond: strand1 -> strand2\n"
"    float signal2  = grad2 * glyco2[0];                  // glycosidic bond 2\n"
"\n"
"    // Strand 1: Adam with cross-strand coupling\n"
"    float eff1 = signal1 + crossMom;\n"
"    float mi1 = beta1[0] * m1[id] + ob1 * eff1;\n"
"    float vi1 = beta2[0] * v1[id] + ob2 * eff1 * eff1;\n"
"    m1[id] = mi1;\n"
"    v1[id] = vi1;\n"
"    d1[id] -= lr[0] * (mi1 / bc1[0] / (sqrt(vi1 / bc2[0]) + eps[0]) + wd[0] * backbone1[0] * d1[id]);\n"
"\n"
"    // Strand 2: Adam with cross-strand coupling\n"
"    float eff2 = signal2 + crossVel;\n"
"    float mi2 = beta1[0] * m2[id] + ob1 * eff2;\n"
"    float vi2 = beta2[0] * v2[id] + ob2 * eff2 * eff2;\n"
"    m2[id] = mi2;\n"
"    v2[id] = vi2;\n"
"    d2[id] -= lr[0] * (mi2 / bc1[0] / (sqrt(vi2 / bc2[0]) + eps[0]) + wd[0] * backbone2[0] * d2[id]);\n"
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
"    device const float* signalScale  [[buffer(6)]],  // +1 improving, -ratio worsening\n"
"    device const float* lr           [[buffer(7)]],\n"
"    device const float* beta1        [[buffer(8)]],\n"
"    device const float* wd           [[buffer(9)]],\n"
"    device const uint* cols          [[buffer(10)]],\n"
"    uint i [[thread_position_in_grid]])\n"
"{\n"
"    if (mask[i] == 0) return;\n"
"\n"
"    float mi = float(mom[i]);\n"
"    float d = delta[i];\n"
"\n"
"    // Synthetic gradient: momentum * signalScale\n"
"    float sg = mi * signalScale[0];\n"
"    mi = beta1[0] * mi + (1.0f - beta1[0]) * sg;\n"
"    d -= lr[0] * (mi + wd[0] * d);\n"
"\n"
"    uint row = i / cols[0];\n"
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
"    device const float* maxNorm       [[buffer(2)]],\n"
"    uint id [[thread_position_in_grid]])\n"
"{\n"
"    float norm = sqrt(sumSq[0]);\n"
"    float scale = (norm > maxNorm[0]) ? (maxNorm[0] / norm) : 1.0f;\n"
"    grad[id] *= scale;\n"
"}\n"
"\n"
"// Compute clip scale: out[0] = min(1, maxNorm / sqrt(sumSq[0])).\n"
"kernel void compute_clip_scale(\n"
"    device const float* sumSq   [[buffer(0)]],\n"
"    device float* clipScale     [[buffer(1)]],\n"
"    device const float* maxNorm     [[buffer(2)]],\n"
"    uint id [[thread_position_in_grid]])\n"
"{\n"
"    float norm = sqrt(sumSq[0]);\n"
"    clipScale[0] = (norm > maxNorm[0]) ? (maxNorm[0] / norm) : 1.0f;\n"
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
"    device const uint* dim      [[buffer(5)]],\n"
"    device const uint* kvDim    [[buffer(6)]],\n"
"    device const uint* headDim  [[buffer(7)]],\n"
"    device const uint* nHeads   [[buffer(8)]],\n"
"    device const uint* nKVHeads [[buffer(9)]],\n"
"    device const uint* seqLen   [[buffer(10)]],\n"
"    uint h [[threadgroup_position_in_grid]],\n"
"    uint tid [[thread_index_in_threadgroup]])\n"
"{\n"
"    if (h >= nHeads[0]) return;\n"
"    uint kvH = h / (nHeads[0] / nKVHeads[0]);\n"
"    float scale = 1.0f / sqrt(float(headDim[0]));\n"
"    uint n = seqLen[0];\n"
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
"        for (uint d = 0; d < headDim[0]; d++) {\n"
"            dot += Q[i * dim[0] + h * headDim[0] + d] * K[j * kvDim[0] + kvH * headDim[0] + d];\n"
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
"    for (uint d = 0; d < headDim[0]; d++) {\n"
"        float acc = 0.0f;\n"
"        for (uint j = 0; j <= i; j++) {\n"
"            acc += scores_out[h * n * n + i * n + j] * V[j * kvDim[0] + kvH * headDim[0] + d];\n"
"        }\n"
"        out[i * dim[0] + h * headDim[0] + d] = acc;\n"
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
"    device const uint* dim         [[buffer(8)]],\n"
"    device const uint* kvDim      [[buffer(9)]],\n"
"    device const uint* headDim    [[buffer(10)]],\n"
"    device const uint* nHeads     [[buffer(11)]],\n"
"    device const uint* nKVHeads   [[buffer(12)]],\n"
"    device const uint* seqLen     [[buffer(13)]],\n"
"    uint h [[threadgroup_position_in_grid]],\n"
"    uint tid [[thread_index_in_threadgroup]])\n"
"{\n"
"    if (h >= nHeads[0]) return;\n"
"    uint kvH = h / (nHeads[0] / nKVHeads[0]);\n"
"    float scale = 1.0f / sqrt(float(headDim[0]));\n"
"    uint n = seqLen[0];\n"
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
"        for (uint d = 0; d < headDim[0]; d++) {\n"
"            dv += dOut[i * dim[0] + h * headDim[0] + d] * V[j * kvDim[0] + kvH * headDim[0] + d];\n"
"        }\n"
"        dotSum += scores[h * n * n + i * n + j] * dv;\n"
"    }\n"
"\n"
"    // Softmax backward + accumulate dQ, dK, dV\n"
"    for (uint j = 0; j <= i; j++) {\n"
"        float dv = 0.0f;\n"
"        for (uint d = 0; d < headDim[0]; d++) {\n"
"            dv += dOut[i * dim[0] + h * headDim[0] + d] * V[j * kvDim[0] + kvH * headDim[0] + d];\n"
"        }\n"
"        float ds = scores[h * n * n + i * n + j] * (dv - dotSum) * scale;\n"
"\n"
"        // dQ[i] += ds * K[j]\n"
"        for (uint d = 0; d < headDim[0]; d++) {\n"
"            dQ[i * dim[0] + h * headDim[0] + d] += ds * K[j * kvDim[0] + kvH * headDim[0] + d];\n"
"        }\n"
"        // dK[j] += ds * Q[i] (atomic for GQA — multiple heads share same KV head)\n"
"        for (uint d = 0; d < headDim[0]; d++) {\n"
"            atomic_fetch_add_explicit(\n"
"                (device atomic_float*)&dK[j * kvDim[0] + kvH * headDim[0] + d],\n"
"                ds * Q[i * dim[0] + h * headDim[0] + d],\n"
"                memory_order_relaxed);\n"
"        }\n"
"        // dV[j] += scores[i,j] * dOut[i] (atomic for GQA)\n"
"        float s = scores[h * n * n + i * n + j];\n"
"        for (uint d = 0; d < headDim[0]; d++) {\n"
"            atomic_fetch_add_explicit(\n"
"                (device atomic_float*)&dV[j * kvDim[0] + kvH * headDim[0] + d],\n"
"                s * dOut[i * dim[0] + h * headDim[0] + d],\n"
"                memory_order_relaxed);\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"// Bias add: x[i] += bias[i % cols] for all positions\n"
"kernel void bias_add(\n"
"    device float* x        [[buffer(0)]],\n"
"    device const float* b  [[buffer(1)]],\n"
"    device const uint* cols   [[buffer(2)]],\n"
"    uint i [[thread_position_in_grid]])\n"
"{\n"
"    x[i] += b[i % cols[0]];\n"
"}\n"
"\n"
"// RoPE forward: rotate pairs in Q and K.\n"
"// One thread per (position, pair). dim_pairs = headDim/2 * nHeads.\n"
"kernel void rope_forward(\n"
"    device float* x        [[buffer(0)]],  // [n, dim] (Q or K)\n"
"    device const uint* headDim [[buffer(1)]],\n"
"    device const uint* nHeads  [[buffer(2)]],\n"
"    device const float* theta  [[buffer(3)]],\n"
"    device const uint* stride  [[buffer(4)]],  // dim for Q, kvDim for K\n"
"    uint2 gid [[thread_position_in_grid]])  // (pair_idx, position)\n"
"{\n"
"    uint pos = gid.y;\n"
"    uint pairIdx = gid.x;  // which pair across all heads\n"
"    uint h = pairIdx / (headDim[0] / 2);\n"
"    uint i = (pairIdx % (headDim[0] / 2)) * 2;\n"
"    if (h >= nHeads[0]) return;\n"
"\n"
"    float freq = 1.0f / pow(theta[0], float(i) / float(headDim[0]));\n"
"    float angle = float(pos) * freq;\n"
"    float cosA = cos(angle), sinA = sin(angle);\n"
"\n"
"    uint off = pos * stride[0] + h * headDim[0] + i;\n"
"    float x0 = x[off], x1 = x[off + 1];\n"
"    x[off]     = x0 * cosA - x1 * sinA;\n"
"    x[off + 1] = x0 * sinA + x1 * cosA;\n"
"}\n"
"\n"
"// RoPE backward: same rotation but negate sinA.\n"
"kernel void rope_backward(\n"
"    device float* dx       [[buffer(0)]],\n"
"    device const uint* headDim [[buffer(1)]],\n"
"    device const uint* nHeads  [[buffer(2)]],\n"
"    device const float* theta  [[buffer(3)]],\n"
"    device const uint* stride  [[buffer(4)]],\n"
"    uint2 gid [[thread_position_in_grid]])\n"
"{\n"
"    uint pos = gid.y;\n"
"    uint pairIdx = gid.x;\n"
"    uint h = pairIdx / (headDim[0] / 2);\n"
"    uint i = (pairIdx % (headDim[0] / 2)) * 2;\n"
"    if (h >= nHeads[0]) return;\n"
"\n"
"    float freq = 1.0f / pow(theta[0], float(i) / float(headDim[0]));\n"
"    float angle = float(pos) * freq;\n"
"    float cosA = cos(angle), sinA = -sin(angle);\n"
"\n"
"    uint off = pos * stride[0] + h * headDim[0] + i;\n"
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
"    device const uint* M     [[buffer(3)]],\n"
"    device const uint* K     [[buffer(4)]],\n"
"    device const uint* N     [[buffer(5)]],\n"
"    uint2 tid [[thread_position_in_threadgroup]],\n"
"    uint2 gid [[threadgroup_position_in_grid]])\n"
"{\n"
"    threadgroup float sA[TILE * TILE];\n"
"    threadgroup float sB[TILE * TILE];\n"
"    tiled_gemm_bt(A, B, C, M[0], K[0], N[0], tid, gid, sA, sB);\n"
"}\n"
"\n"
"// GEMM accumulate: C += A @ B^T.\n"
"kernel void gemm_bt_acc(\n"
"    device const float* A [[buffer(0)]],\n"
"    device const float* B [[buffer(1)]],\n"
"    device float* C       [[buffer(2)]],\n"
"    device const uint* M     [[buffer(3)]],\n"
"    device const uint* K     [[buffer(4)]],\n"
"    device const uint* N     [[buffer(5)]],\n"
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
"    for (uint t = 0; t < (K[0] + TILE - 1) / TILE; t++) {\n"
"        uint aCol = t * TILE + tid.x;\n"
"        uint bCol = t * TILE + tid.y;\n"
"        sA[tid.y * TILE + tid.x] = (row < M[0] && aCol < K[0]) ? A[row * K[0] + aCol] : 0.0f;\n"
"        sB[tid.y * TILE + tid.x] = (col < N[0] && bCol < K[0]) ? B[col * K[0] + bCol] : 0.0f;\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        for (uint i = 0; i < TILE; i++) acc += sA[tid.y * TILE + i] * sB[i * TILE + tid.x];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (row < M[0] && col < N[0]) C[row * N[0] + col] += acc;\n"
"}\n"
"\n"
"// GEMM: C = A^T @ B. A[K,M], B[K,N] → C[M,N]. For weight gradients.\n"
"kernel void gemm_tn(\n"
"    device const float* A [[buffer(0)]],\n"
"    device const float* B [[buffer(1)]],\n"
"    device float* C       [[buffer(2)]],\n"
"    device const uint* M     [[buffer(3)]],\n"
"    device const uint* K     [[buffer(4)]],\n"
"    device const uint* N     [[buffer(5)]],\n"
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
"    for (uint t = 0; t < (K[0] + TILE - 1) / TILE; t++) {\n"
"        uint aRow = t * TILE + tid.x;  // K dimension\n"
"        uint bRow = t * TILE + tid.y;  // K dimension\n"
"        // A is [K,M], transposed access: A^T[row, aRow] = A[aRow, row]\n"
"        sA[tid.y * TILE + tid.x] = (row < M[0] && aRow < K[0]) ? A[aRow * M[0] + row] : 0.0f;\n"
"        sB[tid.y * TILE + tid.x] = (col < N[0] && bRow < K[0]) ? B[bRow * N[0] + col] : 0.0f;\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        for (uint i = 0; i < TILE; i++) acc += sA[tid.y * TILE + i] * sB[i * TILE + tid.x];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (row < M[0] && col < N[0]) C[row * N[0] + col] = acc;\n"
"}\n"
"\n"
"// Sparse TN GEMM: C[m,n] = A^T[m,k] @ B[k,n], skip output rows where mask[row]==0.\n"
"// Threadgroup-level early exit: if no row in the tile is hot, skip entirely.\n"
"kernel void gemm_tn_sparse(\n"
"    device const float* A [[buffer(0)]],\n"
"    device const float* B [[buffer(1)]],\n"
"    device float* C       [[buffer(2)]],\n"
"    device const uint* M     [[buffer(3)]],\n"
"    device const uint* K     [[buffer(4)]],\n"
"    device const uint* N     [[buffer(5)]],\n"
"    device const char* mask [[buffer(6)]],\n"
"    uint2 tid [[thread_position_in_threadgroup]],\n"
"    uint2 gid [[threadgroup_position_in_grid]])\n"
"{\n"
"    // Check if ANY row in this tile is hot\n"
"    uint baseRow = gid.y * TILE;\n"
"    threadgroup bool tileHot;\n"
"    if (tid.x == 0 && tid.y == 0) {\n"
"        tileHot = false;\n"
"        for (uint r = 0; r < TILE && baseRow + r < M[0]; r++) {\n"
"            if (mask[baseRow + r] != 0) { tileHot = true; break; }\n"
"        }\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (!tileHot) return;\n"
"\n"
"    threadgroup float sA[TILE * TILE];\n"
"    threadgroup float sB[TILE * TILE];\n"
"\n"
"    uint row = baseRow + tid.y;\n"
"    uint col = gid.x * TILE + tid.x;\n"
"    float acc = 0.0f;\n"
"\n"
"    for (uint t = 0; t < (K[0] + TILE - 1) / TILE; t++) {\n"
"        uint aRow = t * TILE + tid.x;\n"
"        uint bRow = t * TILE + tid.y;\n"
"        sA[tid.y * TILE + tid.x] = (row < M[0] && aRow < K[0]) ? A[aRow * M[0] + row] : 0.0f;\n"
"        sB[tid.y * TILE + tid.x] = (col < N[0] && bRow < K[0]) ? B[bRow * N[0] + col] : 0.0f;\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        for (uint i = 0; i < TILE; i++) acc += sA[tid.y * TILE + i] * sB[i * TILE + tid.x];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (row < M[0] && col < N[0] && mask[row] != 0) C[row * N[0] + col] = acc;\n"
"}\n"
"\n"
"// GEMM: C = A @ B (no transpose). A[M,K], B[K,N] → C[M,N].\n"
"kernel void gemm_nn(\n"
"    device const float* A [[buffer(0)]],\n"
"    device const float* B [[buffer(1)]],\n"
"    device float* C       [[buffer(2)]],\n"
"    device const uint* M     [[buffer(3)]],\n"
"    device const uint* K     [[buffer(4)]],\n"
"    device const uint* N     [[buffer(5)]],\n"
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
"    for (uint t = 0; t < (K[0] + TILE - 1) / TILE; t++) {\n"
"        uint aCol = t * TILE + tid.x;\n"
"        uint bRow = t * TILE + tid.y;\n"
"        sA[tid.y * TILE + tid.x] = (row < M[0] && aCol < K[0]) ? A[row * K[0] + aCol] : 0.0f;\n"
"        sB[tid.y * TILE + tid.x] = (col < N[0] && bRow < K[0]) ? B[bRow * N[0] + col] : 0.0f;\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        for (uint i = 0; i < TILE; i++) acc += sA[tid.y * TILE + i] * sB[i * TILE + tid.x];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (row < M[0] && col < N[0]) C[row * N[0] + col] = acc;\n"
"}\n"
"\n"
"// RMSNorm with scale output for backward. One threadgroup per position.\n"
"// Writes normed values in-place and scale[pos] = 1/rms for backward.\n"
"kernel void rmsnorm_save(\n"
"    device float* x         [[buffer(0)]],   // [n, dim] modified in-place\n"
"    device const float* w   [[buffer(1)]],   // [dim] weights\n"
"    device float* scale     [[buffer(2)]],   // [n] output: 1/rms per position\n"
"    device const uint* dim     [[buffer(3)]],\n"
"    device const float* eps    [[buffer(4)]],\n"
"    uint pos [[threadgroup_position_in_grid]],\n"
"    uint tid [[thread_index_in_threadgroup]],\n"
"    uint tpg [[threads_per_threadgroup]])\n"
"{\n"
"    uint off = pos * dim[0];\n"
"    // Cooperative sum of squares\n"
"    float localSS = 0.0f;\n"
"    for (uint i = tid; i < dim[0]; i += tpg) localSS += x[off + i] * x[off + i];\n"
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
"    float rms = 1.0f / sqrt(finalSS / float(dim[0]) + eps[0]);\n"
"    if (tid == 0) scale[pos] = rms;\n"
"\n"
"    for (uint i = tid; i < dim[0]; i += tpg) {\n"
"        x[off + i] = x[off + i] * rms * w[i];\n"
"    }\n"
"}\n"
"\n"
"// RMSNorm out-of-place: output = rmsnorm(input, w). Input NOT modified.\n"
"kernel void rmsnorm_out(\n"
"    device const float* input  [[buffer(0)]],   // [n, dim] read-only\n"
"    device float* output       [[buffer(1)]],   // [n, dim] written\n"
"    device const float* w      [[buffer(2)]],   // [dim] weights\n"
"    device const uint* dim     [[buffer(3)]],\n"
"    device const float* eps    [[buffer(4)]],\n"
"    uint pos [[threadgroup_position_in_grid]],\n"
"    uint tid [[thread_index_in_threadgroup]],\n"
"    uint tpg [[threads_per_threadgroup]])\n"
"{\n"
"    uint off = pos * dim[0];\n"
"    float localSS = 0.0f;\n"
"    for (uint i = tid; i < dim[0]; i += tpg) localSS += input[off + i] * input[off + i];\n"
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
"    float rms = 1.0f / sqrt(finalSS / float(dim[0]) + eps[0]);\n"
"    for (uint i = tid; i < dim[0]; i += tpg) {\n"
"        output[off + i] = input[off + i] * rms * w[i];\n"
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
"    device const uint* dim        [[buffer(5)]],\n"
"    uint pos [[threadgroup_position_in_grid]],\n"
"    uint tid [[thread_index_in_threadgroup]],\n"
"    uint tpg [[threads_per_threadgroup]])\n"
"{\n"
"    uint off = pos * dim[0];\n"
"    float rms = scale[pos];\n"
"\n"
"    // Pass 1: cooperative dot product sum(dOut * w * xIn)\n"
"    float localDot = 0.0f;\n"
"    for (uint i = tid; i < dim[0]; i += tpg) {\n"
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
"    float coeff = finalDot * rms * rms * rms / float(dim[0]);\n"
"\n"
"    // Pass 2: compute dx\n"
"    for (uint i = tid; i < dim[0]; i += tpg) {\n"
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
"    device const uint* cols        [[buffer(3)]],\n"
"    uint i [[thread_position_in_grid]])\n"
"{\n"
"    uint row = i / cols[0];\n"
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
"    device const uint* cols        [[buffer(5)]],\n"
"    uint i [[thread_position_in_grid]])\n"
"{\n"
"    uint row = i / cols[0];\n"
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
"    device const uint* cols        [[buffer(4)]],\n"
"    uint i [[thread_position_in_grid]])\n"
"{\n"
"    uint row = i / cols[0];\n"
"    float scale = scales[row] / 127.0f;\n"
"    dst[i] = float(src[i]) * scale + delta[i];\n"
"}\n"
"\n"
"// Dequant INT8 → FP16 for Metal 4 matmul2d.\n"
"kernel void dequant_int8_fp16(\n"
"    device const char* src      [[buffer(0)]],\n"
"    device const float* scales  [[buffer(1)]],\n"
"    device half* dst            [[buffer(2)]],\n"
"    device const uint* cols        [[buffer(3)]],\n"
"    uint i [[thread_position_in_grid]])\n"
"{\n"
"    uint row = i / cols[0];\n"
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
"// Fused: grad clip + Adam + requant + FP32 live writeback.\n"
"// clipScale = min(1, maxNorm/sqrt(gradNormSq)) precomputed on CPU.\n"
"kernel void helix_needle(\n"
"    device char* data_int8        [[buffer(0)]],\n"
"    device float* scales          [[buffer(1)]],\n"
"    device const float* grad      [[buffer(2)]],\n"
"    device half* mom              [[buffer(3)]],\n"
"    device half* vel              [[buffer(4)]],\n"
"    device const char* mask       [[buffer(5)]],\n"
"    device float* delta           [[buffer(6)]],\n"
"    device const float* lr           [[buffer(7)]],\n"
"    device const float* beta1        [[buffer(8)]],\n"
"    device const float* beta2        [[buffer(9)]],\n"
"    device const float* bc1          [[buffer(10)]],\n"
"    device const float* bc2          [[buffer(11)]],\n"
"    device const float* eps          [[buffer(12)]],\n"
"    device const float* wd           [[buffer(13)]],\n"
"    device const uint* cols          [[buffer(14)]],\n"
"    device float* live            [[buffer(15)]],\n"
"    device const float* clipBuf  [[buffer(16)]],\n"
"    uint i [[thread_position_in_grid]])\n"
"{\n"
"    uint row = i / cols[0];\n"
"    if (mask[row] == 0) return;\n"
"\n"
"    float scale = scales[row] / 127.0f;\n"
"    float w = float(data_int8[i]) * scale + delta[i];\n"
"    float mi = float(mom[i]);\n"
"    float vi = float(vel[i]);\n"
"\n"
"    float g = grad[i] * clipBuf[0];\n"
"    float ob1 = 1.0f - beta1[0], ob2 = 1.0f - beta2[0];\n"
"    mi = beta1[0] * mi + ob1 * g;\n"
"    vi = beta2[0] * vi + ob2 * g * g;\n"
"    w -= lr[0] * (mi / bc1[0] / (sqrt(vi / bc2[0]) + eps[0]) + wd[0] * w);\n"
"\n"
"    mom[i] = half(mi);\n"
"    vel[i] = half(vi);\n"
"\n"
"    float inv_scale = 127.0f / (scales[row] + 1e-10f);\n"
"    float qi = clamp(w * inv_scale, -127.0f, 127.0f);\n"
"    char q_int8 = char(rint(qi));\n"
"    delta[i] = w - float(q_int8) * scale;\n"
"    data_int8[i] = q_int8;\n"
"    live[i] = float(q_int8) * scale + delta[i];\n"
"}\n"
"\n"
"// Paired fused: clip + DNA rung + requant + FP32 live writeback.\n"
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
"    device float* delta1          [[buffer(11)]],\n"
"    device float* delta2          [[buffer(12)]],\n"
"    device const float* lr           [[buffer(13)]],\n"
"    device const float* beta1        [[buffer(14)]],\n"
"    device const float* beta2        [[buffer(15)]],\n"
"    device const float* bc1          [[buffer(16)]],\n"
"    device const float* bc2          [[buffer(17)]],\n"
"    device const float* eps          [[buffer(18)]],\n"
"    device const float* wd           [[buffer(19)]],\n"
"    device const float* backbone1    [[buffer(20)]],\n"
"    device const float* glyco1       [[buffer(21)]],\n"
"    device const float* hbond1       [[buffer(22)]],\n"
"    device const float* hbond2       [[buffer(23)]],\n"
"    device const float* glyco2       [[buffer(24)]],\n"
"    device const float* backbone2    [[buffer(25)]],\n"
"    device const float* bondStrength [[buffer(26)]],\n"
"    device const uint* cols          [[buffer(27)]],\n"
"    device float* live1           [[buffer(28)]],\n"
"    device float* live2           [[buffer(29)]],\n"
"    device const float* clipBuf  [[buffer(30)]],\n"
"    uint i [[thread_position_in_grid]])\n"
"{\n"
"    uint row = i / cols[0];\n"
"    if (mask[row] == 0) return;\n"
"\n"
"    float cs = clipBuf[0];\n"
"    float scale1 = s1[row] / 127.0f;\n"
"    float scale2 = s2[row] / 127.0f;\n"
"    float w1 = float(d1_int8[i]) * scale1 + delta1[i];\n"
"    float w2 = float(d2_int8[i]) * scale2 + delta2[i];\n"
"    float mi1 = float(m1[i]), vi1 = float(v1[i]);\n"
"    float mi2 = float(m2[i]), vi2 = float(v2[i]);\n"
"    float ob1 = 1.0f - beta1[0], ob2 = 1.0f - beta2[0];\n"
"\n"
"    float signal1 = g1[i] * cs * glyco1[0];\n"
"    float crossMom = g2[i] * cs * hbond1[0] * bondStrength[0];\n"
"    float effGrad1 = signal1 + crossMom;\n"
"    mi1 = beta1[0] * mi1 + ob1 * effGrad1;\n"
"    vi1 = beta2[0] * vi1 + ob2 * effGrad1 * effGrad1;\n"
"    w1 -= lr[0] * (mi1 / bc1[0] / (sqrt(vi1 / bc2[0]) + eps[0]) + wd[0] * backbone1[0] * w1);\n"
"\n"
"    float signal2 = g2[i] * cs * glyco2[0];\n"
"    float crossVel = g1[i] * cs * hbond2[0] * bondStrength[0];\n"
"    float effGrad2 = signal2 + crossVel;\n"
"    mi2 = beta1[0] * mi2 + ob1 * effGrad2;\n"
"    vi2 = beta2[0] * vi2 + ob2 * effGrad2 * effGrad2;\n"
"    w2 -= lr[0] * (mi2 / bc1[0] / (sqrt(vi2 / bc2[0]) + eps[0]) + wd[0] * backbone2[0] * w2);\n"
"\n"
"    m1[i] = half(mi1); v1[i] = half(vi1);\n"
"    m2[i] = half(mi2); v2[i] = half(vi2);\n"
"\n"
"    float inv1 = 127.0f / (s1[row] + 1e-10f);\n"
"    float inv2 = 127.0f / (s2[row] + 1e-10f);\n"
"    char qi1 = char(rint(clamp(w1 * inv1, -127.0f, 127.0f)));\n"
"    char qi2 = char(rint(clamp(w2 * inv2, -127.0f, 127.0f)));\n"
"    delta1[i] = w1 - float(qi1) * scale1;\n"
"    delta2[i] = w2 - float(qi2) * scale2;\n"
"    d1_int8[i] = qi1; d2_int8[i] = qi2;\n"
"    live1[i] = float(qi1) * scale1 + delta1[i];\n"
"    live2[i] = float(qi2) * scale2 + delta2[i];\n"
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
static id<MTLComputePipelineState> g_ps_clip_scale_compute = nil;
// Fused training kernels (from fused_train.metallib)
static id<MTLComputePipelineState> g_ps_fused_pre_attn = nil;
static id<MTLComputePipelineState> g_ps_fused_post_attn = nil;
static bool g_use_fused_train = false;
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
static id<MTLComputePipelineState> g_ps_gemm_tn_sparse = nil;
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
static id<MTLComputePipelineState> g_ps_rmsnorm_out = nil;
static id<MTLComputePipelineState> g_ps_fused_attn = nil;
static id<MTLComputePipelineState> g_ps_fused_attn_bwd = nil;
static id<MTLComputePipelineState> g_ps_bias_add = nil;
static id<MTLComputePipelineState> g_ps_rope_fwd = nil;
static id<MTLComputePipelineState> g_ps_rope_bwd = nil;

static id<MTLComputePipelineState> make_ps(NSString* name) {
    NSError* err = nil;
    id<MTLFunction> fn = [g_compute_lib newFunctionWithName:name];
    if (!fn) { NSLog(@"mongoose: kernel %@ not found", name); return nil; }
    MTLComputePipelineDescriptor* desc = [[MTLComputePipelineDescriptor alloc] init];
    desc.computeFunction = fn;
    desc.supportIndirectCommandBuffers = YES;
    id<MTLComputePipelineState> ps = [g_device newComputePipelineStateWithDescriptor:desc options:0 reflection:nil error:&err];
    if (err) { NSLog(@"mongoose: pipeline %@: %@", name, err); return nil; }
    return ps;
}

// Constant buffer cache — avoid per-dispatch MTLBuffer allocation.
// Maps 4-byte values to pre-allocated shared MTLBuffers.
#define CONST_CACHE_SIZE 64
static struct { uint32_t key; id<MTLBuffer> buf; bool used; } g_const_cache[CONST_CACHE_SIZE];
static int g_const_cache_count = 0;

static id<MTLBuffer> const_buf(const void* val, size_t len) {
    uint32_t key = 0;
    memcpy(&key, val, len < 4 ? len : 4);
    for (int i = 0; i < g_const_cache_count; i++) {
        if (g_const_cache[i].key == key) return g_const_cache[i].buf;
    }
    id<MTLBuffer> buf = [g_device newBufferWithBytes:val length:4 options:MTLResourceStorageModeShared];
    if (g_const_cache_count < CONST_CACHE_SIZE) {
        g_const_cache[g_const_cache_count++] = (typeof(g_const_cache[0])){key, buf, true};
    }
    return buf;
}

// Mutable constant buffer — for per-step values (lr, bc1, bc2, etc.)
// Pre-allocated, CPU writes before dispatch.
#define MUT_CONST_COUNT 16
static id<MTLBuffer> g_mut_const[MUT_CONST_COUNT];
static int g_mut_const_next = 0;

static id<MTLBuffer> mut_const_buf(float val) {
    int idx = g_mut_const_next++ % MUT_CONST_COUNT;
    if (!g_mut_const[idx]) {
        g_mut_const[idx] = [g_device newBufferWithLength:4 options:MTLResourceStorageModeShared];
    }
    *(float*)[g_mut_const[idx] contents] = val;
    return g_mut_const[idx];
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
    g_ps_clip_scale_compute = make_ps(@"compute_clip_scale");
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
    g_ps_gemm_tn_sparse = make_ps(@"gemm_tn_sparse");
    g_ps_rmsnorm_save = make_ps(@"rmsnorm_save");
    g_ps_rmsnorm_out = make_ps(@"rmsnorm_out");
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

    // Load fused training kernels from fused_train.metallib
    g_use_fused_train = false;
    for (NSString* dir in searchPaths) {
        NSString* libPath = [dir stringByAppendingPathComponent:@"fused_train.metallib"];
        if ([[NSFileManager defaultManager] fileExistsAtPath:libPath]) {
            NSURL* libURL = [NSURL fileURLWithPath:libPath];
            NSError* ftErr = nil;
            id<MTLLibrary> ftLib = [g_device newLibraryWithURL:libURL error:&ftErr];
            if (ftLib) {
                NSError* psErr = nil;
                id<MTLFunction> fnPre = [ftLib newFunctionWithName:@"fused_pre_attn"];
                id<MTLFunction> fnPost = [ftLib newFunctionWithName:@"fused_post_attn"];
                if (fnPre && fnPost) {
                    MTLComputePipelineDescriptor* pd = [[MTLComputePipelineDescriptor alloc] init];
                    pd.supportIndirectCommandBuffers = YES;
                    pd.computeFunction = fnPre;
                    g_ps_fused_pre_attn = [g_device newComputePipelineStateWithDescriptor:pd options:0 reflection:nil error:&psErr];
                    pd.computeFunction = fnPost;
                    g_ps_fused_post_attn = [g_device newComputePipelineStateWithDescriptor:pd options:0 reflection:nil error:&psErr];
                    if (g_ps_fused_pre_attn && g_ps_fused_post_attn) {
                        g_use_fused_train = true;
                        NSLog(@"mongoose: fused training kernels loaded from %@", libPath);
                    }
                }
            }
            if (g_use_fused_train) break;
        }
    }
    if (!g_use_fused_train) {
        NSLog(@"mongoose: fused_train.metallib not found — using per-op dispatch");
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
    id<MTLBuffer> lrBuf   = const_buf(&lr, 4);
    id<MTLBuffer> b1Buf   = const_buf(&beta1, 4);
    id<MTLBuffer> b2Buf   = const_buf(&beta2, 4);
    id<MTLBuffer> bc1Buf  = const_buf(&bc1, 4);
    id<MTLBuffer> bc2Buf  = const_buf(&bc2, 4);
    id<MTLBuffer> epsBuf  = const_buf(&eps, 4);
    id<MTLBuffer> wdBuf   = const_buf(&wd, 4);

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
    id<MTLBuffer> bb1Buf  = const_buf(&backbone1, 4);
    id<MTLBuffer> gl1Buf  = const_buf(&glyco1, 4);
    id<MTLBuffer> hb1Buf  = const_buf(&hbond1, 4);
    id<MTLBuffer> hb2Buf  = const_buf(&hbond2, 4);
    id<MTLBuffer> gl2Buf  = const_buf(&glyco2, 4);
    id<MTLBuffer> bb2Buf  = const_buf(&backbone2, 4);
    id<MTLBuffer> bsBuf   = const_buf(&bondStr, 4);
    id<MTLBuffer> lrBuf   = const_buf(&lr, 4);
    id<MTLBuffer> b1Buf   = const_buf(&beta1, 4);
    id<MTLBuffer> b2Buf   = const_buf(&beta2, 4);
    id<MTLBuffer> bc1Buf  = const_buf(&bc1, 4);
    id<MTLBuffer> bc2Buf  = const_buf(&bc2, 4);
    id<MTLBuffer> epsBuf  = const_buf(&eps, 4);
    id<MTLBuffer> wdBuf   = const_buf(&wd, 4);

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

    id<MTLBuffer> bb1Buf  = const_buf(&backbone1, 4);
    id<MTLBuffer> gl1Buf  = const_buf(&glyco1, 4);
    id<MTLBuffer> hb1Buf  = const_buf(&hbond1, 4);
    id<MTLBuffer> hb2Buf  = const_buf(&hbond2, 4);
    id<MTLBuffer> gl2Buf  = const_buf(&glyco2, 4);
    id<MTLBuffer> bb2Buf  = const_buf(&backbone2, 4);
    id<MTLBuffer> bsBuf   = const_buf(&bondStr, 4);
    id<MTLBuffer> lrBuf   = const_buf(&lr, 4);
    id<MTLBuffer> b1Buf   = const_buf(&beta1, 4);
    id<MTLBuffer> b2Buf   = const_buf(&beta2, 4);
    id<MTLBuffer> bc1Buf  = const_buf(&bc1, 4);
    id<MTLBuffer> bc2Buf  = const_buf(&bc2, 4);
    id<MTLBuffer> epsBuf  = const_buf(&eps, 4);
    id<MTLBuffer> wdBuf   = const_buf(&wd, 4);

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

    id<MTLBuffer> lrBuf   = const_buf(&lr, 4);
    id<MTLBuffer> b1Buf   = const_buf(&beta1, 4);
    id<MTLBuffer> b2Buf   = const_buf(&beta2, 4);
    id<MTLBuffer> bc1Buf  = const_buf(&bc1, 4);
    id<MTLBuffer> bc2Buf  = const_buf(&bc2, 4);
    id<MTLBuffer> epsBuf  = const_buf(&eps, 4);
    id<MTLBuffer> wdBuf   = const_buf(&wd, 4);

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

    id<MTLBuffer> dimBuf   = const_buf(&udim, 4);
    id<MTLBuffer> vocabBuf = const_buf(&uvocab, 4);
    id<MTLBuffer> nBuf     = const_buf(&un, 4);
    id<MTLBuffer> invNBuf  = const_buf(&invN, 4);

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

    id<MTLBuffer> dimBuf  = const_buf(&udim, 4);
    id<MTLBuffer> hotBuf  = const_buf(&uhot, 4);
    id<MTLBuffer> nBuf    = const_buf(&un, 4);
    id<MTLBuffer> invNBuf = const_buf(&invN, 4);

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
    id<MTLBuffer> colsBuf = const_buf(&ucols, 4);

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
    id<MTLBuffer> colsBuf = const_buf(&ucols, 4);
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
    id<MTLBuffer> lrBuf   = const_buf(&lr, 4);
    id<MTLBuffer> b1Buf   = const_buf(&beta1, 4);
    id<MTLBuffer> b2Buf   = const_buf(&beta2, 4);
    id<MTLBuffer> bc1Buf  = const_buf(&bc1, 4);
    id<MTLBuffer> bc2Buf  = const_buf(&bc2, 4);
    id<MTLBuffer> epsBuf  = const_buf(&eps, 4);
    id<MTLBuffer> wdBuf   = const_buf(&wd, 4);
    id<MTLBuffer> colsBuf = const_buf(&ucols, 4);

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
    id<MTLBuffer> lrBuf   = const_buf(&lr, 4);
    id<MTLBuffer> b1Buf   = const_buf(&beta1, 4);
    id<MTLBuffer> b2Buf   = const_buf(&beta2, 4);
    id<MTLBuffer> bc1Buf  = const_buf(&bc1, 4);
    id<MTLBuffer> bc2Buf  = const_buf(&bc2, 4);
    id<MTLBuffer> epsBuf  = const_buf(&eps, 4);
    id<MTLBuffer> wdBuf   = const_buf(&wd, 4);
    id<MTLBuffer> bb1Buf  = const_buf(&backbone1, 4);
    id<MTLBuffer> gl1Buf  = const_buf(&glyco1, 4);
    id<MTLBuffer> hb1Buf  = const_buf(&hbond1, 4);
    id<MTLBuffer> hb2Buf  = const_buf(&hbond2, 4);
    id<MTLBuffer> gl2Buf  = const_buf(&glyco2, 4);
    id<MTLBuffer> bb2Buf  = const_buf(&backbone2, 4);
    id<MTLBuffer> bsBuf   = const_buf(&bondStrength, 4);
    id<MTLBuffer> colsBuf = const_buf(&ucols, 4);

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

// Commit slot but keep cmd for wait.
void mtl_fused_commit_slot(int slot) {
    if (!g_fused_enc[slot]) return;
    [g_fused_enc[slot] endEncoding];
    [g_fused_cmd[slot] commit];
    g_fused_enc[slot] = nil;
}

// Wait for a previously committed slot.
void mtl_fused_wait_slot(int slot) {
    if (g_fused_cmd[slot]) {
        [g_fused_cmd[slot] waitUntilCompleted];
        g_fused_cmd[slot] = nil;
    }
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
    id<MTLBuffer> mBuf = const_buf(&um, 4);
    id<MTLBuffer> kBuf = const_buf(&uk, 4);
    id<MTLBuffer> nBuf = const_buf(&un, 4);
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
    id<MTLBuffer> mBuf = const_buf(&um, 4);
    id<MTLBuffer> kBuf = const_buf(&uk, 4);
    id<MTLBuffer> nBuf = const_buf(&un, 4);
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
    id<MTLBuffer> mBuf = const_buf(&um, 4);
    id<MTLBuffer> kBuf = const_buf(&uk, 4);
    id<MTLBuffer> nBuf = const_buf(&un, 4);
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
    id<MTLBuffer> colsBuf = const_buf(&ucols, 4);
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
    id<MTLBuffer> ssBuf   = const_buf(&ss, 4);
    id<MTLBuffer> lrBuf   = const_buf(&lr2, 4);
    id<MTLBuffer> b1Buf   = const_buf(&b1, 4);
    id<MTLBuffer> wdBuf   = const_buf(&wd2, 4);
    id<MTLBuffer> colsBuf = const_buf(&ucols, 4);
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
    id<MTLBuffer> vBuf = const_buf(&uv, 4);
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
    id<MTLBuffer> mBuf = const_buf(&um, 4);
    id<MTLBuffer> kBuf = const_buf(&uk, 4);
    id<MTLBuffer> nBuf = const_buf(&un, 4);
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
    id<MTLBuffer> mBuf = const_buf(&um, 4);
    id<MTLBuffer> kBuf = const_buf(&uk, 4);
    id<MTLBuffer> nBuf = const_buf(&un, 4);
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
    id<MTLBuffer> mBuf = const_buf(&um, 4);
    id<MTLBuffer> kBuf = const_buf(&uk, 4);
    id<MTLBuffer> nBuf = const_buf(&un, 4);
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
    id<MTLBuffer> dimBuf = const_buf(&udim, 4);
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
    id<MTLBuffer> dimBuf = const_buf(&udim, 4);
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
    id<MTLBuffer> colsBuf = const_buf(&ucols, 4);
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
    id<MTLBuffer> dimBuf = const_buf(&udim, 4);
    id<MTLBuffer> epsBuf = const_buf(&eps, 4);
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
    id<MTLBuffer> dBuf   = const_buf(&ud, 4);
    id<MTLBuffer> kvBuf  = const_buf(&ukv, 4);
    id<MTLBuffer> hdBuf  = const_buf(&uhd, 4);
    id<MTLBuffer> nhBuf  = const_buf(&unh, 4);
    id<MTLBuffer> nkvBuf = const_buf(&unkv, 4);
    id<MTLBuffer> nBuf   = const_buf(&un, 4);
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
    id<MTLBuffer> hdBuf = const_buf(&uhd, 4);
    id<MTLBuffer> nhBuf = const_buf(&unh, 4);
    id<MTLBuffer> thBuf = const_buf(&absTheta, 4);
    id<MTLBuffer> stBuf = const_buf(&ust, 4);
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
    id<MTLBuffer> colsBuf = const_buf(&ucols, 4);
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
    id<MTLBuffer> mBuf = const_buf(&um, 4);
    id<MTLBuffer> kBuf = const_buf(&uk, 4);
    id<MTLBuffer> nBuf = const_buf(&un, 4);
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
        id<MTLBuffer> mBuf = const_buf(&um, 4);
        id<MTLBuffer> kBuf = const_buf(&uk, 4);
        id<MTLBuffer> nBuf = const_buf(&un, 4);
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
    id<MTLBuffer> mBuf = const_buf(&um, 4);
    id<MTLBuffer> kBuf = const_buf(&uk, 4);
    id<MTLBuffer> nBuf = const_buf(&un, 4);
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
    id<MTLBuffer> mBuf = const_buf(&um, 4);
    id<MTLBuffer> kBuf = const_buf(&uk, 4);
    id<MTLBuffer> nBuf = const_buf(&un, 4);
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
    id<MTLBuffer> mBuf = const_buf(&um, 4);
    id<MTLBuffer> kBuf = const_buf(&uk, 4);
    id<MTLBuffer> nBuf = const_buf(&un, 4);
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
    id<MTLBuffer> dimBuf = const_buf(&udim, 4);
    id<MTLBuffer> epsBuf = const_buf(&eps, 4);
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
    id<MTLBuffer> dBuf   = const_buf(&ud, 4);
    id<MTLBuffer> kvBuf  = const_buf(&ukv, 4);
    id<MTLBuffer> hdBuf  = const_buf(&uhd, 4);
    id<MTLBuffer> nhBuf  = const_buf(&unh, 4);
    id<MTLBuffer> nkvBuf = const_buf(&unkv, 4);
    id<MTLBuffer> nBuf   = const_buf(&un, 4);
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
    id<MTLBuffer> dBuf   = const_buf(&ud, 4);
    id<MTLBuffer> kvBuf  = const_buf(&ukv, 4);
    id<MTLBuffer> hdBuf  = const_buf(&uhd, 4);
    id<MTLBuffer> nhBuf  = const_buf(&unh, 4);
    id<MTLBuffer> nkvBuf = const_buf(&unkv, 4);
    id<MTLBuffer> nBuf   = const_buf(&un, 4);
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
    id<MTLBuffer> colsBuf = const_buf(&ucols, 4);
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
    id<MTLBuffer> hdBuf  = const_buf(&uhd, 4);
    id<MTLBuffer> nhBuf  = const_buf(&unh, 4);
    id<MTLBuffer> thBuf  = const_buf(&theta, 4);
    id<MTLBuffer> stBuf  = const_buf(&ust, 4);
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
    id<MTLBuffer> hdBuf  = const_buf(&uhd, 4);
    id<MTLBuffer> nhBuf  = const_buf(&unh, 4);
    id<MTLBuffer> thBuf  = const_buf(&theta, 4);
    id<MTLBuffer> stBuf  = const_buf(&ust, 4);
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
    id<MTLBuffer> dimBuf = const_buf(&dimVal, 4);
    id<MTLBuffer> epsBuf = const_buf(&epsVal, 4);

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
    id<MTLBuffer> lrBuf   = const_buf(&lr, 4);
    id<MTLBuffer> b1Buf   = const_buf(&beta1, 4);
    id<MTLBuffer> b2Buf   = const_buf(&beta2, 4);
    id<MTLBuffer> bc1Buf  = const_buf(&bc1, 4);
    id<MTLBuffer> bc2Buf  = const_buf(&bc2, 4);
    id<MTLBuffer> epsBuf  = const_buf(&eps, 4);
    id<MTLBuffer> wdBuf   = const_buf(&wd, 4);
    id<MTLBuffer> colsBuf = const_buf(&ucols, 4);

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
    id<MTLBuffer> colsBuf = const_buf(&ucols, 4);

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

void mtl_fused_compute_clip_scale(void* sumSqRef, void* clipScaleRef, float maxNorm) {
    [g_fused_enc[g_active_fused_slot] setComputePipelineState:g_ps_clip_scale_compute];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)sumSqRef     offset:0 atIndex:0];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)clipScaleRef offset:0 atIndex:1];
    [g_fused_enc[g_active_fused_slot] setBytes:&maxNorm length:sizeof(float) atIndex:2];
    [g_fused_enc[g_active_fused_slot] dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
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

// Fused LM head pass 1: max + sumExp per position. No logits buffer.
void mtl_fused_lm_head_pass1(void* hiddenRef, void* embedRef, void* maxRef, void* sumExpRef,
                              int dim, int vocabSize, int nPositions) {
    uint32_t udim = dim, uvocab = vocabSize, un = nPositions;
    id<MTLBuffer> dimBuf   = const_buf(&udim, 4);
    id<MTLBuffer> vocabBuf = const_buf(&uvocab, 4);
    id<MTLBuffer> nBuf     = const_buf(&un, 4);
    id<MTLComputeCommandEncoder> enc = g_fused_enc[g_active_fused_slot];
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
    [enc dispatchThreadgroups:MTLSizeMake(nPositions, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
}

// Fused LM head pass 2: loss + dHidden from softmax CE gradient. No logits buffer.
void mtl_fused_lm_head_pass2(void* hiddenRef, void* embedRef, void* maxRef, void* sumExpRef,
                              void* targetsRef, void* dHiddenRef, void* lossRef,
                              int dim, int vocabSize, int nPositions) {
    uint32_t udim = dim, uvocab = vocabSize, un = nPositions;
    float invN = 1.0f / (float)nPositions;
    id<MTLBuffer> dimBuf   = const_buf(&udim, 4);
    id<MTLBuffer> vocabBuf = const_buf(&uvocab, 4);
    id<MTLBuffer> nBuf     = const_buf(&un, 4);
    id<MTLBuffer> invNBuf  = const_buf(&invN, 4);
    // Zero loss scalar before atomic adds
    float zero = 0.0f;
    memcpy([(__bridge id<MTLBuffer>)lossRef contents], &zero, sizeof(float));
    id<MTLComputeCommandEncoder> enc = g_fused_enc[g_active_fused_slot];
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
    [enc dispatchThreadgroups:MTLSizeMake(nPositions, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
}

// Fused pre-attention dispatch.
void mtl_fused_pre_attn(void* hidden, void* normW, void* wq, void* wk, void* wv,
                        void* Q, void* K, void* V, void* normedOut, void* rmsScale, void* xIn,
                        int dim, int kvDim, int headDim, int nHeads, int nKVHeads, int ffnDim, int seqLen,
                        float ropeTheta, float eps) {
    if (!g_ps_fused_pre_attn) return;
    // Pack constants struct
    struct { uint dim,kvDim,headDim,nHeads,nKVHeads,ffnDim,seqLen; float ropeTheta,eps; } C;
    C.dim=dim; C.kvDim=kvDim; C.headDim=headDim; C.nHeads=nHeads; C.nKVHeads=nKVHeads;
    C.ffnDim=ffnDim; C.seqLen=seqLen; C.ropeTheta=ropeTheta; C.eps=eps;
    id<MTLBuffer> cBuf = [g_device newBufferWithBytes:&C length:sizeof(C) options:MTLResourceStorageModeShared];
    id<MTLComputeCommandEncoder> enc = g_fused_enc[g_active_fused_slot];
    [enc setComputePipelineState:g_ps_fused_pre_attn];
    [enc setBuffer:(__bridge id<MTLBuffer>)hidden    offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)normW     offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)wq        offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)wk        offset:0 atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)wv        offset:0 atIndex:4];
    [enc setBuffer:(__bridge id<MTLBuffer>)Q         offset:0 atIndex:5];
    [enc setBuffer:(__bridge id<MTLBuffer>)K         offset:0 atIndex:6];
    [enc setBuffer:(__bridge id<MTLBuffer>)V         offset:0 atIndex:7];
    [enc setBuffer:(__bridge id<MTLBuffer>)normedOut offset:0 atIndex:8];
    [enc setBuffer:(__bridge id<MTLBuffer>)rmsScale  offset:0 atIndex:9];
    [enc setBuffer:(__bridge id<MTLBuffer>)xIn       offset:0 atIndex:10];
    [enc setBuffer:cBuf offset:0 atIndex:11];
    NSUInteger tpg = dim;
    if (tpg > g_ps_fused_pre_attn.maxTotalThreadsPerThreadgroup) tpg = g_ps_fused_pre_attn.maxTotalThreadsPerThreadgroup;
    [enc dispatchThreadgroups:MTLSizeMake(seqLen, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
}

// Fused post-attention dispatch.
void mtl_fused_post_attn(void* hidden, void* attnOut, void* wo, void* normW2,
                         void* gate, void* up, void* down,
                         void* xMid, void* normed2, void* rmsScale2,
                         void* gatePre, void* upOut, void* ffnMid,
                         int dim, int kvDim, int headDim, int nHeads, int nKVHeads, int ffnDim, int seqLen,
                         float ropeTheta, float eps) {
    if (!g_ps_fused_post_attn) return;
    struct { uint dim,kvDim,headDim,nHeads,nKVHeads,ffnDim,seqLen; float ropeTheta,eps; } C;
    C.dim=dim; C.kvDim=kvDim; C.headDim=headDim; C.nHeads=nHeads; C.nKVHeads=nKVHeads;
    C.ffnDim=ffnDim; C.seqLen=seqLen; C.ropeTheta=ropeTheta; C.eps=eps;
    id<MTLBuffer> cBuf = [g_device newBufferWithBytes:&C length:sizeof(C) options:MTLResourceStorageModeShared];
    id<MTLComputeCommandEncoder> enc = g_fused_enc[g_active_fused_slot];
    [enc setComputePipelineState:g_ps_fused_post_attn];
    [enc setBuffer:(__bridge id<MTLBuffer>)hidden    offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)attnOut   offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)wo        offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)normW2    offset:0 atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)gate      offset:0 atIndex:4];
    [enc setBuffer:(__bridge id<MTLBuffer>)up        offset:0 atIndex:5];
    [enc setBuffer:(__bridge id<MTLBuffer>)down      offset:0 atIndex:6];
    [enc setBuffer:(__bridge id<MTLBuffer>)xMid      offset:0 atIndex:7];
    [enc setBuffer:(__bridge id<MTLBuffer>)normed2   offset:0 atIndex:8];
    [enc setBuffer:(__bridge id<MTLBuffer>)rmsScale2 offset:0 atIndex:9];
    [enc setBuffer:(__bridge id<MTLBuffer>)gatePre   offset:0 atIndex:10];
    [enc setBuffer:(__bridge id<MTLBuffer>)upOut     offset:0 atIndex:11];
    [enc setBuffer:(__bridge id<MTLBuffer>)ffnMid    offset:0 atIndex:12];
    [enc setBuffer:cBuf offset:0 atIndex:13];
    NSUInteger tpg = ffnDim > (uint)dim ? ffnDim : dim;
    if (tpg > g_ps_fused_post_attn.maxTotalThreadsPerThreadgroup) tpg = g_ps_fused_post_attn.maxTotalThreadsPerThreadgroup;
    [enc dispatchThreadgroups:MTLSizeMake(seqLen, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
}

int mtl_fused_train_available(void) { return g_use_fused_train ? 1 : 0; }

// Sparse TN GEMM: C = A^T @ B, skipping output rows where mask[row]==0.
void mtl_fused_gemm_tn_sparse(void* aRef, void* bRef, void* cRef, void* maskRef, int M, int K, int N) {
    uint32_t um = M, uk = K, un = N;
    id<MTLBuffer> mBuf = const_buf(&um, 4);
    id<MTLBuffer> kBuf = const_buf(&uk, 4);
    id<MTLBuffer> nBuf = const_buf(&un, 4);
    [g_fused_enc[g_active_fused_slot] setComputePipelineState:g_ps_gemm_tn_sparse];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)aRef    offset:0 atIndex:0];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)bRef    offset:0 atIndex:1];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)cRef    offset:0 atIndex:2];
    [g_fused_enc[g_active_fused_slot] setBuffer:mBuf offset:0 atIndex:3];
    [g_fused_enc[g_active_fused_slot] setBuffer:kBuf offset:0 atIndex:4];
    [g_fused_enc[g_active_fused_slot] setBuffer:nBuf offset:0 atIndex:5];
    [g_fused_enc[g_active_fused_slot] setBuffer:(__bridge id<MTLBuffer>)maskRef offset:0 atIndex:6];
    [g_fused_enc[g_active_fused_slot] dispatchThreadgroups:MTLSizeMake((N+31)/32, (M+31)/32, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 32, 1)];
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
    float eps, float wd, int n, int cols,
    void* liveRef, void* clipBufRef) {
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
    [enc setBuffer:(__bridge id<MTLBuffer>)liveRef    offset:0 atIndex:15];
    [enc setBuffer:(__bridge id<MTLBuffer>)clipBufRef offset:0 atIndex:16];
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
    float bondStrength, int n, int cols,
    void* live1Ref, void* live2Ref, void* clipBufRef) {
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
    [enc setBuffer:(__bridge id<MTLBuffer>)live1Ref   offset:0 atIndex:28];
    [enc setBuffer:(__bridge id<MTLBuffer>)live2Ref   offset:0 atIndex:29];
    [enc setBuffer:(__bridge id<MTLBuffer>)clipBufRef offset:0 atIndex:30];
    NSUInteger tpg = g_ps_needle_paired.maxTotalThreadsPerThreadgroup;
    if (tpg > 1024) tpg = 1024;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
}

// ICB build parameter struct.
typedef struct {
    int dim, kvDim, headDim, nHeads, nKVHeads, ffnDim, vocabSize, seqLen, nLayers;
    void* hidden; void* normedFinal; void* finalNorm; void* finalScales;
    void* lmMaxLogit; void* lmSumExp; void* lmLoss; void* targetsGPU;
    void* dHidden; void* dScratch; void* dEmbed;
    void* gradSumSq; void* clipScaleBuf; void* scores;
    void* embed; void* embedData; void* embedScales; void* embedDelta;
    void* embedMom; void* embedVel; void* embedMask; void* embedLive;
    void** norm1; void** norm2;
    void** a_xIn; void** a_normed; void** a_Q; void** a_K; void** a_V; void** a_attnOut;
    void** a_xMid; void** a_normed2; void** a_gatePre; void** a_upOut; void** a_ffnMid;
    void** a_rmsScale1; void** a_rmsScale2; void** a_gateAct;
    void** wq_data; void** wq_scales; void** wq_delta; void** wq_mom; void** wq_vel; void** wq_live; void** wq_mask;
    void** wk_data; void** wk_scales; void** wk_delta; void** wk_mom; void** wk_vel; void** wk_live; void** wk_mask;
    void** wv_data; void** wv_scales; void** wv_delta; void** wv_mom; void** wv_vel; void** wv_live; void** wv_mask;
    void** wo_data; void** wo_scales; void** wo_delta; void** wo_mom; void** wo_vel; void** wo_live; void** wo_mask;
    void** gate_data; void** gate_scales; void** gate_delta; void** gate_mom; void** gate_vel; void** gate_live; void** gate_mask;
    void** up_data; void** up_scales; void** up_delta; void** up_mom; void** up_vel; void** up_live; void** up_mask;
    void** down_data; void** down_scales; void** down_delta; void** down_mom; void** down_vel; void** down_live; void** down_mask;
    void** b_dFfnMid; void** b_dGate; void** b_dUp; void** b_dN2; void** b_dx;
    void** b_dAttnOut; void** b_dQ; void** b_dK; void** b_dV; void** b_dN1;
    void** b_dWDown; void** b_dWGate; void** b_dWUp; void** b_dWO; void** b_dWQ; void** b_dWK; void** b_dWV;
    void* lrBuf; void* bc1Buf; void* bc2Buf; void* maxNormBuf;
    void* bb1Buf; void* gly1Buf; void* hb1Buf; void* hb2Buf; void* gly2Buf; void* bb2Buf; void* bondStrBuf;
} ICBBuildParams;

// === Indirect Command Buffer (ICB) for training ===
// Apple8+ (M2+): pre-encode into ICB, replay per step.
// Apple7 (M1): batch-encode into regular compute encoder per step (one CGo call).

#define ICB_MAX_CMDS 512

static bool g_use_icb = false;
static id<MTLIndirectCommandBuffer> g_train_icb = nil;
static int g_train_icb_fwd_count = 0;
static int g_train_icb_total_count = 0;
static int g_icb_cursor = 0;

// Track all buffers the ICB touches for useResource calls.
#define ICB_MAX_RESOURCES 1024
static id<MTLResource> g_icb_resources[ICB_MAX_RESOURCES];
static int g_icb_resource_count = 0;

static void icb_track_buf(void* ref) {
    if (!ref || g_icb_resource_count >= ICB_MAX_RESOURCES) return;
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)ref;
    // Deduplicate (simple linear scan — only at init)
    for (int i = 0; i < g_icb_resource_count; i++) {
        if (g_icb_resources[i] == buf) return;
    }
    g_icb_resources[g_icb_resource_count++] = buf;
}

// Saved params for batch-encode fallback (Apple7).
static ICBBuildParams* g_batch_params = nil;

// Helper: encode a 1D dispatch into the ICB
static void icb_1d(id<MTLComputePipelineState> ps, int n, void* bufs[], int bufCount) {
    id<MTLIndirectComputeCommand> cmd = [g_train_icb indirectComputeCommandAtIndex:g_icb_cursor];
    [cmd setComputePipelineState:ps];
    for (int i = 0; i < bufCount; i++) {
        [cmd setKernelBuffer:(__bridge id<MTLBuffer>)bufs[i] offset:0 atIndex:i];
    }
    NSUInteger tpg = ps.maxTotalThreadsPerThreadgroup;
    if (tpg > 1024) tpg = 1024;
    [cmd concurrentDispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    g_icb_cursor++;
}

// Helper: encode a threadgroup dispatch (for GEMM, attention, RMSNorm)
static void icb_tg(id<MTLComputePipelineState> ps, int gx, int gy, int tx, int ty, void* bufs[], int bufCount) {
    id<MTLIndirectComputeCommand> cmd = [g_train_icb indirectComputeCommandAtIndex:g_icb_cursor];
    [cmd setComputePipelineState:ps];
    for (int i = 0; i < bufCount; i++) {
        [cmd setKernelBuffer:(__bridge id<MTLBuffer>)bufs[i] offset:0 atIndex:i];
    }
    [cmd concurrentDispatchThreadgroups:MTLSizeMake(gx, gy, 1) threadsPerThreadgroup:MTLSizeMake(tx, ty, 1)];
    g_icb_cursor++;
}

// Helper: encode a GEMM (BT/NN/TN) — picks Metal4 or tiled fallback
static void icb_gemm(id<MTLComputePipelineState> ps, id<MTLComputePipelineState> ps4,
                     void* A, void* B, void* C, id<MTLBuffer> mBuf, id<MTLBuffer> kBuf, id<MTLBuffer> nBuf,
                     int M, int K, int N) {
    id<MTLIndirectComputeCommand> cmd = [g_train_icb indirectComputeCommandAtIndex:g_icb_cursor];
    id<MTLComputePipelineState> use_ps = (g_use_metal4_gemm && ps4) ? ps4 : ps;
    [cmd setComputePipelineState:use_ps];
    [cmd setKernelBuffer:(__bridge id<MTLBuffer>)A offset:0 atIndex:0];
    [cmd setKernelBuffer:(__bridge id<MTLBuffer>)B offset:0 atIndex:1];
    [cmd setKernelBuffer:(__bridge id<MTLBuffer>)C offset:0 atIndex:2];
    [cmd setKernelBuffer:mBuf offset:0 atIndex:3];
    [cmd setKernelBuffer:kBuf offset:0 atIndex:4];
    [cmd setKernelBuffer:nBuf offset:0 atIndex:5];
    if (g_use_metal4_gemm && ps4) {
        NSUInteger sw = ps4.threadExecutionWidth;
        [cmd concurrentDispatchThreadgroups:MTLSizeMake((N+31)/32, (M+63)/64, 1) threadsPerThreadgroup:MTLSizeMake(sw*4, 1, 1)];
    } else {
        [cmd concurrentDispatchThreadgroups:MTLSizeMake((N+31)/32, (M+31)/32, 1) threadsPerThreadgroup:MTLSizeMake(32, 32, 1)];
    }
    g_icb_cursor++;
}

// Helper: encode sparse TN GEMM with mask
static void icb_gemm_tn_sparse(void* A, void* B, void* C, void* mask,
                                id<MTLBuffer> mBuf, id<MTLBuffer> kBuf, id<MTLBuffer> nBuf, int M, int N) {
    id<MTLIndirectComputeCommand> cmd = [g_train_icb indirectComputeCommandAtIndex:g_icb_cursor];
    [cmd setComputePipelineState:g_ps_gemm_tn_sparse];
    [cmd setKernelBuffer:(__bridge id<MTLBuffer>)A offset:0 atIndex:0];
    [cmd setKernelBuffer:(__bridge id<MTLBuffer>)B offset:0 atIndex:1];
    [cmd setKernelBuffer:(__bridge id<MTLBuffer>)C offset:0 atIndex:2];
    [cmd setKernelBuffer:mBuf offset:0 atIndex:3];
    [cmd setKernelBuffer:kBuf offset:0 atIndex:4];
    [cmd setKernelBuffer:nBuf offset:0 atIndex:5];
    [cmd setKernelBuffer:(__bridge id<MTLBuffer>)mask offset:0 atIndex:6];
    [cmd concurrentDispatchThreadgroups:MTLSizeMake((N+31)/32, (M+31)/32, 1) threadsPerThreadgroup:MTLSizeMake(32, 32, 1)];
    g_icb_cursor++;
}

// Helper: encode copy_mem kernel
static void icb_copy(void* dst, void* src, int n) {
    id<MTLIndirectComputeCommand> cmd = [g_train_icb indirectComputeCommandAtIndex:g_icb_cursor];
    [cmd setComputePipelineState:g_ps_copy_mem];
    [cmd setKernelBuffer:(__bridge id<MTLBuffer>)src offset:0 atIndex:0];
    [cmd setKernelBuffer:(__bridge id<MTLBuffer>)dst offset:0 atIndex:1];
    NSUInteger tpg = g_ps_copy_mem.maxTotalThreadsPerThreadgroup;
    if (tpg > 1024) tpg = 1024;
    [cmd concurrentDispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    g_icb_cursor++;
}

// Helper: encode RMSNorm (save variant)
static void icb_rmsnorm(void* x, void* w, void* scale, id<MTLBuffer> dimBuf, id<MTLBuffer> epsBuf, int seqLen, int dim) {
    id<MTLIndirectComputeCommand> cmd = [g_train_icb indirectComputeCommandAtIndex:g_icb_cursor];
    [cmd setComputePipelineState:g_ps_rmsnorm_save];
    [cmd setKernelBuffer:(__bridge id<MTLBuffer>)x offset:0 atIndex:0];
    [cmd setKernelBuffer:(__bridge id<MTLBuffer>)w offset:0 atIndex:1];
    [cmd setKernelBuffer:(__bridge id<MTLBuffer>)scale offset:0 atIndex:2];
    [cmd setKernelBuffer:dimBuf offset:0 atIndex:3];
    [cmd setKernelBuffer:epsBuf offset:0 atIndex:4];
    NSUInteger tpg = g_ps_rmsnorm_save.maxTotalThreadsPerThreadgroup;
    if (tpg > (NSUInteger)dim) tpg = (NSUInteger)dim;
    tpg = (tpg / 32) * 32; if (tpg == 0) tpg = 32;
    [cmd concurrentDispatchThreadgroups:MTLSizeMake(seqLen, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    g_icb_cursor++;
}

// Helper: encode RMSNorm backward
static void icb_rmsnorm_bwd(void* dOut, void* xIn, void* w, void* scale, void* dx,
                             id<MTLBuffer> dimBuf, int seqLen, int dim) {
    id<MTLIndirectComputeCommand> cmd = [g_train_icb indirectComputeCommandAtIndex:g_icb_cursor];
    [cmd setComputePipelineState:g_ps_rmsnorm_bwd];
    [cmd setKernelBuffer:(__bridge id<MTLBuffer>)dOut offset:0 atIndex:0];
    [cmd setKernelBuffer:(__bridge id<MTLBuffer>)xIn offset:0 atIndex:1];
    [cmd setKernelBuffer:(__bridge id<MTLBuffer>)w offset:0 atIndex:2];
    [cmd setKernelBuffer:(__bridge id<MTLBuffer>)scale offset:0 atIndex:3];
    [cmd setKernelBuffer:(__bridge id<MTLBuffer>)dx offset:0 atIndex:4];
    [cmd setKernelBuffer:dimBuf offset:0 atIndex:5];
    NSUInteger tpg = g_ps_rmsnorm_bwd.maxTotalThreadsPerThreadgroup;
    if (tpg > (NSUInteger)dim) tpg = (NSUInteger)dim;
    tpg = (tpg / 32) * 32; if (tpg == 0) tpg = 32;
    [cmd concurrentDispatchThreadgroups:MTLSizeMake(seqLen, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    g_icb_cursor++;
}

// Helper: encode add_inplace
static void icb_add(void* a, void* b, int n) {
    id<MTLIndirectComputeCommand> cmd = [g_train_icb indirectComputeCommandAtIndex:g_icb_cursor];
    [cmd setComputePipelineState:g_ps_add_inplace];
    [cmd setKernelBuffer:(__bridge id<MTLBuffer>)a offset:0 atIndex:0];
    [cmd setKernelBuffer:(__bridge id<MTLBuffer>)b offset:0 atIndex:1];
    NSUInteger tpg = g_ps_add_inplace.maxTotalThreadsPerThreadgroup;
    if (tpg > 1024) tpg = 1024;
    [cmd concurrentDispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    g_icb_cursor++;
}

// Batch-encode helper: encode the full step into the active fused encoder.
// Same dispatch sequence as the ICB, but using the existing mtl_fused_* functions.
static void batch_encode_fwd(ICBBuildParams* p);
static void batch_encode_bwd(ICBBuildParams* p);

int mtl_icb_build_training(ICBBuildParams* p) {
    // Detect GPU family
    g_use_icb = [g_device supportsFamily:MTLGPUFamilyApple8];
    if (g_use_icb) {
        NSLog(@"ICB: Apple8+ detected, using indirect command buffers");
    } else {
        NSLog(@"ICB: Apple7 (M1), using batch-encode fallback");
        // Save params for batch-encode per step
        g_batch_params = (ICBBuildParams*)malloc(sizeof(ICBBuildParams));
        memcpy(g_batch_params, p, sizeof(ICBBuildParams));
        return 0;
    }
    // Unpack for readability
    int dim = p->dim, kvDim = p->kvDim, headDim = p->headDim;
    int nHeads = p->nHeads, nKVHeads = p->nKVHeads, ffnDim = p->ffnDim;
    int vocabSize = p->vocabSize, seqLen = p->seqLen, nLayers = p->nLayers;
    void* hidden = p->hidden; void* normedFinal = p->normedFinal;
    void* finalNorm = p->finalNorm; void* finalScales = p->finalScales;
    void* lmMaxLogit = p->lmMaxLogit; void* lmSumExp = p->lmSumExp;
    void* lmLoss = p->lmLoss; void* targetsGPU = p->targetsGPU;
    void* dHidden = p->dHidden; void* dScratch = p->dScratch; void* dEmbed = p->dEmbed;
    void* gradSumSq = p->gradSumSq; void* clipScaleBuf = p->clipScaleBuf; void* scores = p->scores;
    void* embed = p->embed;
    void* embedData = p->embedData; void* embedScales = p->embedScales; void* embedDelta = p->embedDelta;
    void* embedMom = p->embedMom; void* embedVel = p->embedVel; void* embedMask = p->embedMask; void* embedLive = p->embedLive;
    void** norm1 = p->norm1; void** norm2 = p->norm2;
    void** a_xIn = p->a_xIn; void** a_normed = p->a_normed; void** a_Q = p->a_Q; void** a_K = p->a_K;
    void** a_V = p->a_V; void** a_attnOut = p->a_attnOut; void** a_xMid = p->a_xMid;
    void** a_normed2 = p->a_normed2; void** a_gatePre = p->a_gatePre; void** a_upOut = p->a_upOut;
    void** a_ffnMid = p->a_ffnMid; void** a_rmsScale1 = p->a_rmsScale1; void** a_rmsScale2 = p->a_rmsScale2;
    void** a_gateAct = p->a_gateAct;
    void** wq_data=p->wq_data; void** wq_scales=p->wq_scales; void** wq_delta=p->wq_delta;
    void** wq_mom=p->wq_mom; void** wq_vel=p->wq_vel; void** wq_live=p->wq_live; void** wq_mask=p->wq_mask;
    void** wk_data=p->wk_data; void** wk_scales=p->wk_scales; void** wk_delta=p->wk_delta;
    void** wk_mom=p->wk_mom; void** wk_vel=p->wk_vel; void** wk_live=p->wk_live; void** wk_mask=p->wk_mask;
    void** wv_data=p->wv_data; void** wv_scales=p->wv_scales; void** wv_delta=p->wv_delta;
    void** wv_mom=p->wv_mom; void** wv_vel=p->wv_vel; void** wv_live=p->wv_live; void** wv_mask=p->wv_mask;
    void** wo_data=p->wo_data; void** wo_scales=p->wo_scales; void** wo_delta=p->wo_delta;
    void** wo_mom=p->wo_mom; void** wo_vel=p->wo_vel; void** wo_live=p->wo_live; void** wo_mask=p->wo_mask;
    void** gate_data=p->gate_data; void** gate_scales=p->gate_scales; void** gate_delta=p->gate_delta;
    void** gate_mom=p->gate_mom; void** gate_vel=p->gate_vel; void** gate_live=p->gate_live; void** gate_mask=p->gate_mask;
    void** up_data=p->up_data; void** up_scales=p->up_scales; void** up_delta=p->up_delta;
    void** up_mom=p->up_mom; void** up_vel=p->up_vel; void** up_live=p->up_live; void** up_mask=p->up_mask;
    void** down_data=p->down_data; void** down_scales=p->down_scales; void** down_delta=p->down_delta;
    void** down_mom=p->down_mom; void** down_vel=p->down_vel; void** down_live=p->down_live; void** down_mask=p->down_mask;
    void** b_dFfnMid=p->b_dFfnMid; void** b_dGate=p->b_dGate; void** b_dUp=p->b_dUp;
    void** b_dN2=p->b_dN2; void** b_dx=p->b_dx;
    void** b_dAttnOut=p->b_dAttnOut; void** b_dQ=p->b_dQ; void** b_dK=p->b_dK;
    void** b_dV=p->b_dV; void** b_dN1=p->b_dN1;
    void** b_dWDown=p->b_dWDown; void** b_dWGate=p->b_dWGate; void** b_dWUp=p->b_dWUp;
    void** b_dWO=p->b_dWO; void** b_dWQ=p->b_dWQ; void** b_dWK=p->b_dWK; void** b_dWV=p->b_dWV;
    void* lrBuf=p->lrBuf; void* bc1Buf=p->bc1Buf; void* bc2Buf=p->bc2Buf; void* maxNormBuf=p->maxNormBuf;
    void* bb1Buf=p->bb1Buf; void* gly1Buf=p->gly1Buf; void* hb1Buf=p->hb1Buf;
    void* hb2Buf=p->hb2Buf; void* gly2Buf=p->gly2Buf; void* bb2Buf=p->bb2Buf; void* bondStrBuf=p->bondStrBuf;

    int n = seqLen;

    // Pre-cache all constant buffers
    uint32_t udim = dim, ukvDim = kvDim, uhdim = headDim, unHeads = nHeads;
    uint32_t unKVH = nKVHeads, uffn = ffnDim, uvocab = vocabSize, un = seqLen;
    id<MTLBuffer> dimBuf = const_buf(&udim, 4);
    id<MTLBuffer> kvDimBuf = const_buf(&ukvDim, 4);
    id<MTLBuffer> hdBuf = const_buf(&uhdim, 4);
    id<MTLBuffer> nhBuf = const_buf(&unHeads, 4);
    id<MTLBuffer> nkvBuf = const_buf(&unKVH, 4);
    id<MTLBuffer> ffnBuf = const_buf(&uffn, 4);
    id<MTLBuffer> vocabBuf = const_buf(&uvocab, 4);
    id<MTLBuffer> nBuf = const_buf(&un, 4);
    float epsVal = 1e-6f;
    id<MTLBuffer> epsBuf = const_buf(&epsVal, 4);
    float thetaFwd = 10000.0f, thetaBwd = 10000.0f; // backward uses rope_bwd kernel, same positive theta
    id<MTLBuffer> thetaBuf = const_buf(&thetaFwd, 4);
    float invN = 1.0f / (float)n;
    id<MTLBuffer> invNBuf = const_buf(&invN, 4);

    // GEMM dimension buffers for each unique (M,K,N) shape
    uint32_t un_dim = n, udim_dim = dim, ukvDim_dim = kvDim, uffn_dim = ffnDim, uvocab_dim = vocabSize;
    // Forward GEMMs: [n,dim]@[rows,dim]^T → shapes vary
    id<MTLBuffer> gn = const_buf(&un_dim, 4);     // M=n for forward
    id<MTLBuffer> gdim = const_buf(&udim_dim, 4);  // various K/N
    id<MTLBuffer> gkv = const_buf(&ukvDim_dim, 4);
    id<MTLBuffer> gffn = const_buf(&uffn_dim, 4);
    id<MTLBuffer> gvocab = const_buf(&uvocab_dim, 4);

    // Stride buffers for RoPE
    id<MTLBuffer> strideDim = const_buf(&udim_dim, 4);
    id<MTLBuffer> strideKv = const_buf(&ukvDim_dim, 4);

    // Allocate ICB
    MTLIndirectCommandBufferDescriptor *desc = [[MTLIndirectCommandBufferDescriptor alloc] init];
    desc.commandTypes = MTLIndirectCommandTypeConcurrentDispatchThreads;
    desc.inheritBuffers = NO;
    desc.inheritPipelineState = NO;
    desc.maxKernelBufferBindCount = 31;
    g_train_icb = [g_device newIndirectCommandBufferWithDescriptor:desc maxCommandCount:ICB_MAX_CMDS options:MTLResourceStorageModeShared];
    if (!g_train_icb) {
        NSLog(@"ICB allocation failed! device=%@", g_device);
        return -1;
    }
    NSLog(@"ICB allocated: %lu max commands, ps_copy_mem=%p supportICB=%d",
          (unsigned long)g_train_icb.size, g_ps_copy_mem,
          g_ps_copy_mem ? (int)g_ps_copy_mem.supportIndirectCommandBuffers : -1);
    g_icb_cursor = 0;

    // ============ FORWARD ============
    NSLog(@"ICB: encoding forward, nLayers=%d", nLayers);
    for (int li = 0; li < nLayers; li++) {
        NSLog(@"ICB: layer %d copy, cursor=%d, a_xIn=%p hidden=%p", li, g_icb_cursor, a_xIn[li], hidden);
        icb_copy(a_xIn[li], hidden, n*dim);
        icb_rmsnorm(hidden, norm1[li], a_rmsScale1[li], dimBuf, epsBuf, n, dim);
        icb_copy(a_normed[li], hidden, n*dim);
        icb_copy(hidden, a_xIn[li], n*dim);

        icb_gemm(g_ps_gemm_bt, g_ps_gemm4f_bt, a_normed[li], wq_live[li], a_Q[li], gn, gdim, gdim, n, dim, dim);
        icb_gemm(g_ps_gemm_bt, g_ps_gemm4f_bt, a_normed[li], wk_live[li], a_K[li], gn, gdim, gkv, n, dim, kvDim);
        icb_gemm(g_ps_gemm_bt, g_ps_gemm4f_bt, a_normed[li], wv_live[li], a_V[li], gn, gdim, gkv, n, dim, kvDim);

        // RoPE forward Q
        {
            id<MTLIndirectComputeCommand> cmd = [g_train_icb indirectComputeCommandAtIndex:g_icb_cursor];
            [cmd setComputePipelineState:g_ps_rope_fwd];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)a_Q[li] offset:0 atIndex:0];
            [cmd setKernelBuffer:hdBuf offset:0 atIndex:1];
            [cmd setKernelBuffer:nhBuf offset:0 atIndex:2];
            [cmd setKernelBuffer:thetaBuf offset:0 atIndex:3];
            [cmd setKernelBuffer:strideDim offset:0 atIndex:4];
            int nPairs = nHeads * (headDim / 2);
            [cmd concurrentDispatchThreads:MTLSizeMake(nPairs, n, 1) threadsPerThreadgroup:MTLSizeMake(nPairs < 256 ? nPairs : 256, 1, 1)];
            g_icb_cursor++;
        }
        // RoPE forward K
        {
            id<MTLIndirectComputeCommand> cmd = [g_train_icb indirectComputeCommandAtIndex:g_icb_cursor];
            [cmd setComputePipelineState:g_ps_rope_fwd];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)a_K[li] offset:0 atIndex:0];
            [cmd setKernelBuffer:hdBuf offset:0 atIndex:1];
            [cmd setKernelBuffer:nkvBuf offset:0 atIndex:2];
            [cmd setKernelBuffer:thetaBuf offset:0 atIndex:3];
            [cmd setKernelBuffer:strideKv offset:0 atIndex:4];
            int nPairs = nKVHeads * (headDim / 2);
            [cmd concurrentDispatchThreads:MTLSizeMake(nPairs, n, 1) threadsPerThreadgroup:MTLSizeMake(nPairs < 256 ? nPairs : 256, 1, 1)];
            g_icb_cursor++;
        }

        // Attention
        {
            id<MTLIndirectComputeCommand> cmd = [g_train_icb indirectComputeCommandAtIndex:g_icb_cursor];
            [cmd setComputePipelineState:g_ps_fused_attn];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)a_Q[li] offset:0 atIndex:0];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)a_K[li] offset:0 atIndex:1];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)a_V[li] offset:0 atIndex:2];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)a_attnOut[li] offset:0 atIndex:3];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)scores offset:0 atIndex:4];
            [cmd setKernelBuffer:dimBuf offset:0 atIndex:5];
            [cmd setKernelBuffer:kvDimBuf offset:0 atIndex:6];
            [cmd setKernelBuffer:hdBuf offset:0 atIndex:7];
            [cmd setKernelBuffer:nhBuf offset:0 atIndex:8];
            [cmd setKernelBuffer:nkvBuf offset:0 atIndex:9];
            [cmd setKernelBuffer:nBuf offset:0 atIndex:10];
            NSUInteger tpg = (NSUInteger)n;
            if (tpg > g_ps_fused_attn.maxTotalThreadsPerThreadgroup)
                tpg = g_ps_fused_attn.maxTotalThreadsPerThreadgroup;
            [cmd concurrentDispatchThreadgroups:MTLSizeMake(nHeads, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            g_icb_cursor++;
        }

        icb_gemm(g_ps_gemm_bt, g_ps_gemm4f_bt, a_attnOut[li], wo_live[li], dScratch, gn, gdim, gdim, n, dim, dim);
        icb_add(hidden, dScratch, n*dim);

        icb_copy(a_xMid[li], hidden, n*dim);
        icb_rmsnorm(hidden, norm2[li], a_rmsScale2[li], dimBuf, epsBuf, n, dim);
        icb_copy(a_normed2[li], hidden, n*dim);
        icb_copy(hidden, a_xMid[li], n*dim);

        icb_gemm(g_ps_gemm_bt, g_ps_gemm4f_bt, a_normed2[li], gate_live[li], a_gatePre[li], gn, gdim, gffn, n, dim, ffnDim);
        icb_gemm(g_ps_gemm_bt, g_ps_gemm4f_bt, a_normed2[li], up_live[li], a_upOut[li], gn, gdim, gffn, n, dim, ffnDim);

        // SiLU gate mul
        {
            id<MTLIndirectComputeCommand> cmd = [g_train_icb indirectComputeCommandAtIndex:g_icb_cursor];
            [cmd setComputePipelineState:g_ps_silu_gate_mul];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)a_gatePre[li] offset:0 atIndex:0];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)a_upOut[li] offset:0 atIndex:1];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)a_ffnMid[li] offset:0 atIndex:2];
            NSUInteger tpg = g_ps_silu_gate_mul.maxTotalThreadsPerThreadgroup;
            if (tpg > 1024) tpg = 1024;
            [cmd concurrentDispatchThreads:MTLSizeMake(n*ffnDim, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            g_icb_cursor++;
        }

        icb_gemm(g_ps_gemm_bt, g_ps_gemm4f_bt, a_ffnMid[li], down_live[li], dScratch, gn, gffn, gdim, n, ffnDim, dim);
        icb_add(hidden, dScratch, n*dim);
    }

    // Final RMSNorm + copy + LM head
    icb_rmsnorm(hidden, normedFinal /* wait - this is finalNorm not normedFinal */, finalScales, dimBuf, epsBuf, n, dim);

    // Actually: forward does RMSNorm(hidden, finalNorm, finalScales) then copy(normedFinal, hidden)
    // Let me fix: the rmsnorm operates in-place on hidden, saves scale. Then copy to normedFinal.
    // I already encoded rmsnorm above but passed wrong weight buffer. Let me re-do:
    // The rmsnorm helper encodes: ps=rmsnorm_save, buf0=x(hidden), buf1=w(finalNorm), buf2=scale(finalScales)
    // That's correct if I pass finalNorm as the weight. Let me check the call above...
    // icb_rmsnorm(hidden, normedFinal, finalScales, ...) — WRONG, should be finalNorm not normedFinal.
    // I need to fix this. But I can't edit the ICB command after encoding. Let me redo.

    // Ugh — I encoded the wrong buffer. The ICB cursor already advanced.
    // I need to back up. Let me restructure: encode the final block after the loop correctly.

    // Actually, I made the mistake in the code above. Let me just set the cursor back and re-encode.
    g_icb_cursor--; // back up the bad rmsnorm

    icb_rmsnorm(hidden, finalNorm, finalScales, dimBuf, epsBuf, n, dim);
    icb_copy(normedFinal, hidden, n*dim);

    // LM head pass 1
    {
        id<MTLIndirectComputeCommand> cmd = [g_train_icb indirectComputeCommandAtIndex:g_icb_cursor];
        [cmd setComputePipelineState:g_ps_lm_pass1];
        [cmd setKernelBuffer:(__bridge id<MTLBuffer>)normedFinal offset:0 atIndex:0];
        [cmd setKernelBuffer:(__bridge id<MTLBuffer>)embed offset:0 atIndex:1];
        [cmd setKernelBuffer:(__bridge id<MTLBuffer>)lmMaxLogit offset:0 atIndex:2];
        [cmd setKernelBuffer:(__bridge id<MTLBuffer>)lmSumExp offset:0 atIndex:3];
        [cmd setKernelBuffer:dimBuf offset:0 atIndex:4];
        [cmd setKernelBuffer:vocabBuf offset:0 atIndex:5];
        [cmd setKernelBuffer:nBuf offset:0 atIndex:6];
        NSUInteger tpg = g_ps_lm_pass1.maxTotalThreadsPerThreadgroup;
        tpg = (tpg / 32) * 32; if (tpg == 0) tpg = 32;
        [cmd concurrentDispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        g_icb_cursor++;
    }

    // LM head pass 2
    {
        // Zero loss scalar — write directly to shared memory before execute
        // (CPU does this before mtl_icb_execute_training)
        id<MTLIndirectComputeCommand> cmd = [g_train_icb indirectComputeCommandAtIndex:g_icb_cursor];
        [cmd setComputePipelineState:g_ps_lm_pass2];
        [cmd setKernelBuffer:(__bridge id<MTLBuffer>)normedFinal offset:0 atIndex:0];
        [cmd setKernelBuffer:(__bridge id<MTLBuffer>)embed offset:0 atIndex:1];
        [cmd setKernelBuffer:(__bridge id<MTLBuffer>)lmMaxLogit offset:0 atIndex:2];
        [cmd setKernelBuffer:(__bridge id<MTLBuffer>)lmSumExp offset:0 atIndex:3];
        [cmd setKernelBuffer:(__bridge id<MTLBuffer>)targetsGPU offset:0 atIndex:4];
        [cmd setKernelBuffer:(__bridge id<MTLBuffer>)dHidden offset:0 atIndex:5];
        [cmd setKernelBuffer:(__bridge id<MTLBuffer>)lmLoss offset:0 atIndex:6];
        [cmd setKernelBuffer:dimBuf offset:0 atIndex:7];
        [cmd setKernelBuffer:vocabBuf offset:0 atIndex:8];
        [cmd setKernelBuffer:nBuf offset:0 atIndex:9];
        [cmd setKernelBuffer:invNBuf offset:0 atIndex:10];
        NSUInteger tpg = g_ps_lm_pass2.maxTotalThreadsPerThreadgroup;
        tpg = (tpg / 32) * 32; if (tpg == 0) tpg = 32;
        [cmd concurrentDispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        g_icb_cursor++;
    }

    g_train_icb_fwd_count = g_icb_cursor;

    // ============ BACKWARD ============
    icb_rmsnorm_bwd(dHidden, hidden, finalNorm, finalScales, dScratch, dimBuf, n, dim);
    icb_copy(dHidden, dScratch, n*dim);

    for (int li = nLayers - 1; li >= 0; li--) {
        icb_gemm(g_ps_gemm_nn, g_ps_gemm4f_nn, dHidden, down_live[li], b_dFfnMid[li], gn, gdim, gffn, n, dim, ffnDim);
        icb_gemm_tn_sparse(dHidden, a_ffnMid[li], b_dWDown[li], down_mask[li], gdim, gn, gffn, dim, ffnDim);

        // SiLU gate backward
        {
            id<MTLIndirectComputeCommand> cmd = [g_train_icb indirectComputeCommandAtIndex:g_icb_cursor];
            [cmd setComputePipelineState:g_ps_silu_gate_backward];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)b_dFfnMid[li] offset:0 atIndex:0];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)a_gatePre[li] offset:0 atIndex:1];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)a_upOut[li] offset:0 atIndex:2];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)a_gateAct[li] offset:0 atIndex:3];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)b_dGate[li] offset:0 atIndex:4];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)b_dUp[li] offset:0 atIndex:5];
            NSUInteger tpg = g_ps_silu_gate_backward.maxTotalThreadsPerThreadgroup;
            if (tpg > 1024) tpg = 1024;
            [cmd concurrentDispatchThreads:MTLSizeMake(n*ffnDim, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            g_icb_cursor++;
        }

        icb_gemm(g_ps_gemm_nn, g_ps_gemm4f_nn, b_dGate[li], gate_live[li], b_dN2[li], gn, gffn, gdim, n, ffnDim, dim);
        icb_gemm(g_ps_gemm_nn, g_ps_gemm4f_nn, b_dUp[li], up_live[li], b_dx[li], gn, gffn, gdim, n, ffnDim, dim);
        icb_add(b_dN2[li], b_dx[li], n*dim);

        icb_gemm_tn_sparse(b_dGate[li], a_normed2[li], b_dWGate[li], gate_mask[li], gffn, gn, gdim, ffnDim, dim);
        icb_gemm_tn_sparse(b_dUp[li], a_normed2[li], b_dWUp[li], up_mask[li], gffn, gn, gdim, ffnDim, dim);

        icb_rmsnorm_bwd(b_dN2[li], a_xMid[li], norm2[li], a_rmsScale2[li], b_dx[li], dimBuf, n, dim);
        icb_add(dHidden, b_dx[li], n*dim);

        icb_gemm(g_ps_gemm_nn, g_ps_gemm4f_nn, dHidden, wo_live[li], b_dAttnOut[li], gn, gdim, gdim, n, dim, dim);
        icb_gemm_tn_sparse(dHidden, a_attnOut[li], b_dWO[li], wo_mask[li], gdim, gn, gdim, dim, dim);

        // Attention backward
        {
            id<MTLIndirectComputeCommand> cmd = [g_train_icb indirectComputeCommandAtIndex:g_icb_cursor];
            [cmd setComputePipelineState:g_ps_fused_attn_bwd];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)b_dAttnOut[li] offset:0 atIndex:0];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)a_Q[li] offset:0 atIndex:1];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)a_K[li] offset:0 atIndex:2];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)a_V[li] offset:0 atIndex:3];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)scores offset:0 atIndex:4];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)b_dQ[li] offset:0 atIndex:5];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)b_dK[li] offset:0 atIndex:6];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)b_dV[li] offset:0 atIndex:7];
            [cmd setKernelBuffer:dimBuf offset:0 atIndex:8];
            [cmd setKernelBuffer:kvDimBuf offset:0 atIndex:9];
            [cmd setKernelBuffer:hdBuf offset:0 atIndex:10];
            [cmd setKernelBuffer:nhBuf offset:0 atIndex:11];
            [cmd setKernelBuffer:nkvBuf offset:0 atIndex:12];
            [cmd setKernelBuffer:nBuf offset:0 atIndex:13];
            NSUInteger tpg = (NSUInteger)n;
            if (tpg > g_ps_fused_attn_bwd.maxTotalThreadsPerThreadgroup)
                tpg = g_ps_fused_attn_bwd.maxTotalThreadsPerThreadgroup;
            [cmd concurrentDispatchThreadgroups:MTLSizeMake(nHeads, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            g_icb_cursor++;
        }

        // RoPE backward Q
        {
            id<MTLIndirectComputeCommand> cmd = [g_train_icb indirectComputeCommandAtIndex:g_icb_cursor];
            [cmd setComputePipelineState:g_ps_rope_bwd];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)b_dQ[li] offset:0 atIndex:0];
            [cmd setKernelBuffer:hdBuf offset:0 atIndex:1];
            [cmd setKernelBuffer:nhBuf offset:0 atIndex:2];
            [cmd setKernelBuffer:thetaBuf offset:0 atIndex:3];
            [cmd setKernelBuffer:strideDim offset:0 atIndex:4];
            int nPairs = nHeads * (headDim / 2);
            [cmd concurrentDispatchThreads:MTLSizeMake(nPairs, n, 1) threadsPerThreadgroup:MTLSizeMake(nPairs < 256 ? nPairs : 256, 1, 1)];
            g_icb_cursor++;
        }
        // RoPE backward K
        {
            id<MTLIndirectComputeCommand> cmd = [g_train_icb indirectComputeCommandAtIndex:g_icb_cursor];
            [cmd setComputePipelineState:g_ps_rope_bwd];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)b_dK[li] offset:0 atIndex:0];
            [cmd setKernelBuffer:hdBuf offset:0 atIndex:1];
            [cmd setKernelBuffer:nkvBuf offset:0 atIndex:2];
            [cmd setKernelBuffer:thetaBuf offset:0 atIndex:3];
            [cmd setKernelBuffer:strideKv offset:0 atIndex:4];
            int nPairs = nKVHeads * (headDim / 2);
            [cmd concurrentDispatchThreads:MTLSizeMake(nPairs, n, 1) threadsPerThreadgroup:MTLSizeMake(nPairs < 256 ? nPairs : 256, 1, 1)];
            g_icb_cursor++;
        }

        icb_gemm(g_ps_gemm_nn, g_ps_gemm4f_nn, b_dQ[li], wq_live[li], b_dN1[li], gn, gdim, gdim, n, dim, dim);
        icb_gemm(g_ps_gemm_nn, g_ps_gemm4f_nn, b_dK[li], wk_live[li], b_dx[li], gn, gkv, gdim, n, kvDim, dim);
        icb_gemm(g_ps_gemm_nn, g_ps_gemm4f_nn, b_dV[li], wv_live[li], b_dN2[li], gn, gkv, gdim, n, kvDim, dim);
        icb_add(b_dN1[li], b_dx[li], n*dim);
        icb_add(b_dN1[li], b_dN2[li], n*dim);

        icb_gemm_tn_sparse(b_dQ[li], a_normed[li], b_dWQ[li], wq_mask[li], gdim, gn, gdim, dim, dim);
        icb_gemm_tn_sparse(b_dK[li], a_normed[li], b_dWK[li], wk_mask[li], gkv, gn, gdim, kvDim, dim);
        icb_gemm_tn_sparse(b_dV[li], a_normed[li], b_dWV[li], wv_mask[li], gkv, gn, gdim, kvDim, dim);

        icb_rmsnorm_bwd(b_dN1[li], a_xIn[li], norm1[li], a_rmsScale1[li], b_dx[li], dimBuf, n, dim);
        icb_add(dHidden, b_dx[li], n*dim);
    }

    // ============ GRAD NORM ============
    // Zero gradSumSq
    {
        id<MTLIndirectComputeCommand> cmd = [g_train_icb indirectComputeCommandAtIndex:g_icb_cursor];
        [cmd setComputePipelineState:g_ps_zero_mem];
        [cmd setKernelBuffer:(__bridge id<MTLBuffer>)gradSumSq offset:0 atIndex:0];
        [cmd concurrentDispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        g_icb_cursor++;
    }

    // Accumulate grad norm for all dW tensors
    for (int li = 0; li < nLayers; li++) {
        void* grads[] = {b_dWQ[li], b_dWK[li], b_dWV[li], b_dWO[li], b_dWGate[li], b_dWUp[li], b_dWDown[li]};
        int sizes[] = {dim*dim, kvDim*dim, kvDim*dim, dim*dim, ffnDim*dim, ffnDim*dim, dim*ffnDim};
        for (int g = 0; g < 7; g++) {
            void* bufs2[] = {grads[g], gradSumSq};
            icb_1d(g_ps_grad_norm, sizes[g], bufs2, 2);
        }
    }

    // Compute clip scale
    {
        id<MTLIndirectComputeCommand> cmd = [g_train_icb indirectComputeCommandAtIndex:g_icb_cursor];
        [cmd setComputePipelineState:g_ps_clip_scale_compute];
        [cmd setKernelBuffer:(__bridge id<MTLBuffer>)gradSumSq offset:0 atIndex:0];
        [cmd setKernelBuffer:(__bridge id<MTLBuffer>)clipScaleBuf offset:0 atIndex:1];
        [cmd setKernelBuffer:(__bridge id<MTLBuffer>)maxNormBuf offset:0 atIndex:2];
        [cmd concurrentDispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        g_icb_cursor++;
    }

    // ============ NEEDLE (per-layer) ============
    // Needle constants: lr, beta1, beta2, bc1, bc2, eps, wd at buffer indices 7-13
    // cols at 14, live at 15, clipBuf at 16
    float beta1Val = 0.9f, beta2Val = 0.95f, epsNeedle = 1e-8f, wdVal = 0.1f;
    id<MTLBuffer> beta1Buf = const_buf(&beta1Val, 4);
    id<MTLBuffer> beta2Buf = const_buf(&beta2Val, 4);
    id<MTLBuffer> epsNBuf = const_buf(&epsNeedle, 4);
    id<MTLBuffer> wdBuf = const_buf(&wdVal, 4);

    for (int li = 0; li < nLayers; li++) {
        // Gate/Up: GC paired
        {
            uint32_t uc = dim;
            id<MTLBuffer> colsBuf = const_buf(&uc, 4);
            float bondGC = 3.0f/5.0f;
            id<MTLBuffer> bondGCBuf = const_buf(&bondGC, 4);
            id<MTLIndirectComputeCommand> cmd = [g_train_icb indirectComputeCommandAtIndex:g_icb_cursor];
            [cmd setComputePipelineState:g_ps_needle_paired];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)gate_data[li] offset:0 atIndex:0];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)up_data[li] offset:0 atIndex:1];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)gate_scales[li] offset:0 atIndex:2];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)up_scales[li] offset:0 atIndex:3];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)b_dWGate[li] offset:0 atIndex:4];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)b_dWUp[li] offset:0 atIndex:5];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)gate_mom[li] offset:0 atIndex:6];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)up_mom[li] offset:0 atIndex:7];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)gate_vel[li] offset:0 atIndex:8];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)up_vel[li] offset:0 atIndex:9];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)gate_mask[li] offset:0 atIndex:10];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)gate_delta[li] offset:0 atIndex:11];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)up_delta[li] offset:0 atIndex:12];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)lrBuf offset:0 atIndex:13];
            [cmd setKernelBuffer:beta1Buf offset:0 atIndex:14];
            [cmd setKernelBuffer:beta2Buf offset:0 atIndex:15];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)bc1Buf offset:0 atIndex:16];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)bc2Buf offset:0 atIndex:17];
            [cmd setKernelBuffer:epsNBuf offset:0 atIndex:18];
            [cmd setKernelBuffer:wdBuf offset:0 atIndex:19];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)bb1Buf offset:0 atIndex:20];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)gly1Buf offset:0 atIndex:21];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)hb1Buf offset:0 atIndex:22];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)hb2Buf offset:0 atIndex:23];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)gly2Buf offset:0 atIndex:24];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)bb2Buf offset:0 atIndex:25];
            [cmd setKernelBuffer:bondGCBuf offset:0 atIndex:26];
            [cmd setKernelBuffer:colsBuf offset:0 atIndex:27];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)gate_live[li] offset:0 atIndex:28];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)up_live[li] offset:0 atIndex:29];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)clipScaleBuf offset:0 atIndex:30];
            NSUInteger tpg = g_ps_needle_paired.maxTotalThreadsPerThreadgroup;
            if (tpg > 1024) tpg = 1024;
            [cmd concurrentDispatchThreads:MTLSizeMake(ffnDim*dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            g_icb_cursor++;
        }

        // Singles: wq, wk, wv, wo, down
        void* single_data[] = {wq_data[li], wk_data[li], wv_data[li], wo_data[li], down_data[li]};
        void* single_scales[] = {wq_scales[li], wk_scales[li], wv_scales[li], wo_scales[li], down_scales[li]};
        void* single_delta[] = {wq_delta[li], wk_delta[li], wv_delta[li], wo_delta[li], down_delta[li]};
        void* single_mom[] = {wq_mom[li], wk_mom[li], wv_mom[li], wo_mom[li], down_mom[li]};
        void* single_vel[] = {wq_vel[li], wk_vel[li], wv_vel[li], wo_vel[li], down_vel[li]};
        void* single_live[] = {wq_live[li], wk_live[li], wv_live[li], wo_live[li], down_live[li]};
        void* single_mask[] = {wq_mask[li], wk_mask[li], wv_mask[li], wo_mask[li], down_mask[li]};
        void* single_grad[] = {b_dWQ[li], b_dWK[li], b_dWV[li], b_dWO[li], b_dWDown[li]};
        int single_n[] = {dim*dim, kvDim*dim, kvDim*dim, dim*dim, dim*ffnDim};
        int single_cols[] = {dim, dim, dim, dim, ffnDim};

        for (int s = 0; s < 5; s++) {
            uint32_t uc = single_cols[s];
            id<MTLBuffer> colsBuf = const_buf(&uc, 4);
            id<MTLIndirectComputeCommand> cmd = [g_train_icb indirectComputeCommandAtIndex:g_icb_cursor];
            [cmd setComputePipelineState:g_ps_needle];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)single_data[s] offset:0 atIndex:0];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)single_scales[s] offset:0 atIndex:1];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)single_grad[s] offset:0 atIndex:2];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)single_mom[s] offset:0 atIndex:3];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)single_vel[s] offset:0 atIndex:4];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)single_mask[s] offset:0 atIndex:5];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)single_delta[s] offset:0 atIndex:6];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)lrBuf offset:0 atIndex:7];
            [cmd setKernelBuffer:beta1Buf offset:0 atIndex:8];
            [cmd setKernelBuffer:beta2Buf offset:0 atIndex:9];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)bc1Buf offset:0 atIndex:10];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)bc2Buf offset:0 atIndex:11];
            [cmd setKernelBuffer:epsNBuf offset:0 atIndex:12];
            [cmd setKernelBuffer:wdBuf offset:0 atIndex:13];
            [cmd setKernelBuffer:colsBuf offset:0 atIndex:14];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)single_live[s] offset:0 atIndex:15];
            [cmd setKernelBuffer:(__bridge id<MTLBuffer>)clipScaleBuf offset:0 atIndex:16];
            NSUInteger tpg = g_ps_needle.maxTotalThreadsPerThreadgroup;
            if (tpg > 1024) tpg = 1024;
            [cmd concurrentDispatchThreads:MTLSizeMake(single_n[s], 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            g_icb_cursor++;
        }
    }

    // Embed needle
    {
        uint32_t uc = dim;
        id<MTLBuffer> colsBuf = const_buf(&uc, 4);
        id<MTLIndirectComputeCommand> cmd = [g_train_icb indirectComputeCommandAtIndex:g_icb_cursor];
        [cmd setComputePipelineState:g_ps_needle];
        [cmd setKernelBuffer:(__bridge id<MTLBuffer>)embedData offset:0 atIndex:0];
        [cmd setKernelBuffer:(__bridge id<MTLBuffer>)embedScales offset:0 atIndex:1];
        [cmd setKernelBuffer:(__bridge id<MTLBuffer>)dEmbed offset:0 atIndex:2];
        [cmd setKernelBuffer:(__bridge id<MTLBuffer>)embedMom offset:0 atIndex:3];
        [cmd setKernelBuffer:(__bridge id<MTLBuffer>)embedVel offset:0 atIndex:4];
        [cmd setKernelBuffer:(__bridge id<MTLBuffer>)embedMask offset:0 atIndex:5];
        [cmd setKernelBuffer:(__bridge id<MTLBuffer>)embedDelta offset:0 atIndex:6];
        [cmd setKernelBuffer:(__bridge id<MTLBuffer>)lrBuf offset:0 atIndex:7];
        [cmd setKernelBuffer:beta1Buf offset:0 atIndex:8];
        [cmd setKernelBuffer:beta2Buf offset:0 atIndex:9];
        [cmd setKernelBuffer:(__bridge id<MTLBuffer>)bc1Buf offset:0 atIndex:10];
        [cmd setKernelBuffer:(__bridge id<MTLBuffer>)bc2Buf offset:0 atIndex:11];
        [cmd setKernelBuffer:epsNBuf offset:0 atIndex:12];
        [cmd setKernelBuffer:wdBuf offset:0 atIndex:13];
        [cmd setKernelBuffer:colsBuf offset:0 atIndex:14];
        [cmd setKernelBuffer:(__bridge id<MTLBuffer>)embedLive offset:0 atIndex:15];
        [cmd setKernelBuffer:(__bridge id<MTLBuffer>)clipScaleBuf offset:0 atIndex:16];
        NSUInteger tpg = g_ps_needle.maxTotalThreadsPerThreadgroup;
        if (tpg > 1024) tpg = 1024;
        [cmd concurrentDispatchThreads:MTLSizeMake(vocabSize*dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        g_icb_cursor++;
    }

    g_train_icb_total_count = g_icb_cursor;

    // Track all buffers for useResource calls
    g_icb_resource_count = 0;
    icb_track_buf(hidden); icb_track_buf(normedFinal); icb_track_buf(finalNorm); icb_track_buf(finalScales);
    icb_track_buf(lmMaxLogit); icb_track_buf(lmSumExp); icb_track_buf(lmLoss); icb_track_buf(targetsGPU);
    icb_track_buf(dHidden); icb_track_buf(dScratch); icb_track_buf(dEmbed);
    icb_track_buf(gradSumSq); icb_track_buf(clipScaleBuf); icb_track_buf(scores);
    icb_track_buf(embed); icb_track_buf(embedData); icb_track_buf(embedScales); icb_track_buf(embedDelta);
    icb_track_buf(embedMom); icb_track_buf(embedVel); icb_track_buf(embedMask); icb_track_buf(embedLive);
    icb_track_buf(lrBuf); icb_track_buf(bc1Buf); icb_track_buf(bc2Buf); icb_track_buf(maxNormBuf);
    icb_track_buf(bb1Buf); icb_track_buf(gly1Buf); icb_track_buf(hb1Buf);
    icb_track_buf(hb2Buf); icb_track_buf(gly2Buf); icb_track_buf(bb2Buf); icb_track_buf(bondStrBuf);
    for (int li = 0; li < nLayers; li++) {
        icb_track_buf(norm1[li]); icb_track_buf(norm2[li]);
        icb_track_buf(a_xIn[li]); icb_track_buf(a_normed[li]); icb_track_buf(a_Q[li]); icb_track_buf(a_K[li]);
        icb_track_buf(a_V[li]); icb_track_buf(a_attnOut[li]); icb_track_buf(a_xMid[li]); icb_track_buf(a_normed2[li]);
        icb_track_buf(a_gatePre[li]); icb_track_buf(a_upOut[li]); icb_track_buf(a_ffnMid[li]);
        icb_track_buf(a_rmsScale1[li]); icb_track_buf(a_rmsScale2[li]); icb_track_buf(a_gateAct[li]);
        icb_track_buf(wq_data[li]); icb_track_buf(wq_scales[li]); icb_track_buf(wq_delta[li]);
        icb_track_buf(wq_mom[li]); icb_track_buf(wq_vel[li]); icb_track_buf(wq_live[li]); icb_track_buf(wq_mask[li]);
        icb_track_buf(wk_data[li]); icb_track_buf(wk_scales[li]); icb_track_buf(wk_delta[li]);
        icb_track_buf(wk_mom[li]); icb_track_buf(wk_vel[li]); icb_track_buf(wk_live[li]); icb_track_buf(wk_mask[li]);
        icb_track_buf(wv_data[li]); icb_track_buf(wv_scales[li]); icb_track_buf(wv_delta[li]);
        icb_track_buf(wv_mom[li]); icb_track_buf(wv_vel[li]); icb_track_buf(wv_live[li]); icb_track_buf(wv_mask[li]);
        icb_track_buf(wo_data[li]); icb_track_buf(wo_scales[li]); icb_track_buf(wo_delta[li]);
        icb_track_buf(wo_mom[li]); icb_track_buf(wo_vel[li]); icb_track_buf(wo_live[li]); icb_track_buf(wo_mask[li]);
        icb_track_buf(gate_data[li]); icb_track_buf(gate_scales[li]); icb_track_buf(gate_delta[li]);
        icb_track_buf(gate_mom[li]); icb_track_buf(gate_vel[li]); icb_track_buf(gate_live[li]); icb_track_buf(gate_mask[li]);
        icb_track_buf(up_data[li]); icb_track_buf(up_scales[li]); icb_track_buf(up_delta[li]);
        icb_track_buf(up_mom[li]); icb_track_buf(up_vel[li]); icb_track_buf(up_live[li]); icb_track_buf(up_mask[li]);
        icb_track_buf(down_data[li]); icb_track_buf(down_scales[li]); icb_track_buf(down_delta[li]);
        icb_track_buf(down_mom[li]); icb_track_buf(down_vel[li]); icb_track_buf(down_live[li]); icb_track_buf(down_mask[li]);
        icb_track_buf(b_dFfnMid[li]); icb_track_buf(b_dGate[li]); icb_track_buf(b_dUp[li]);
        icb_track_buf(b_dN2[li]); icb_track_buf(b_dx[li]); icb_track_buf(b_dAttnOut[li]);
        icb_track_buf(b_dQ[li]); icb_track_buf(b_dK[li]); icb_track_buf(b_dV[li]); icb_track_buf(b_dN1[li]);
        icb_track_buf(b_dWDown[li]); icb_track_buf(b_dWGate[li]); icb_track_buf(b_dWUp[li]);
        icb_track_buf(b_dWO[li]); icb_track_buf(b_dWQ[li]); icb_track_buf(b_dWK[li]); icb_track_buf(b_dWV[li]);
    }
    // Also track all cached constant buffers
    for (int i = 0; i < g_const_cache_count; i++) {
        if (g_icb_resource_count < ICB_MAX_RESOURCES)
            g_icb_resources[g_icb_resource_count++] = g_const_cache[i].buf;
    }
    NSLog(@"ICB training: %d commands (%d fwd, %d bwd+opt), %d resources", g_train_icb_total_count, g_train_icb_fwd_count, g_train_icb_total_count - g_train_icb_fwd_count, g_icb_resource_count);
    return g_train_icb_fwd_count;
}

// Execute the forward-only portion (step 1 noop).
void mtl_icb_execute_fwd(void) {
    if (g_use_icb) {
        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        for (int i = 0; i < g_icb_resource_count; i++)
            [enc useResource:g_icb_resources[i] usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        [enc executeCommandsInBuffer:g_train_icb withRange:NSMakeRange(0, g_train_icb_fwd_count)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    } else {
        mtl_fused_begin();
        batch_encode_fwd(g_batch_params);
        mtl_fused_end();
    }
}

// Execute the full training step (forward + backward + needle).
void mtl_icb_execute_full(void) {
    if (g_use_icb) {
        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        for (int i = 0; i < g_icb_resource_count; i++)
            [enc useResource:g_icb_resources[i] usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        [enc executeCommandsInBuffer:g_train_icb withRange:NSMakeRange(0, g_train_icb_total_count)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    } else {
        mtl_fused_begin();
        batch_encode_fwd(g_batch_params);
        batch_encode_bwd(g_batch_params);
        mtl_fused_end();
    }
}

// Batch-encode forward pass into active fused encoder.
static void batch_encode_fwd(ICBBuildParams* p) {
    int dim=p->dim, kvDim=p->kvDim, headDim=p->headDim, nHeads=p->nHeads;
    int nKVHeads=p->nKVHeads, ffnDim=p->ffnDim, vocabSize=p->vocabSize, n=p->seqLen, nLayers=p->nLayers;

    // Fused kernels win at dim<512 (dispatch-bound). Tiled GEMMs win at dim>=512 (compute-bound).
    bool use_fused_fwd = g_use_fused_train && (dim < 512);

    for (int li = 0; li < nLayers; li++) {
        if (use_fused_fwd) {
            // 3 dispatches per layer instead of ~22
            mtl_fused_pre_attn(p->hidden, p->norm1[li], p->wq_live[li], p->wk_live[li], p->wv_live[li],
                              p->a_Q[li], p->a_K[li], p->a_V[li], p->a_normed[li], p->a_rmsScale1[li], p->a_xIn[li],
                              dim, kvDim, headDim, nHeads, nKVHeads, ffnDim, n, 10000.0f, 1e-6f);

            mtl_fused_attn(p->a_Q[li], p->a_K[li], p->a_V[li], p->a_attnOut[li], p->scores,
                          dim, kvDim, headDim, nHeads, nKVHeads, n);

            mtl_fused_post_attn(p->hidden, p->a_attnOut[li], p->wo_live[li], p->norm2[li],
                               p->gate_live[li], p->up_live[li], p->down_live[li],
                               p->a_xMid[li], p->a_normed2[li], p->a_rmsScale2[li],
                               p->a_gatePre[li], p->a_upOut[li], p->a_ffnMid[li],
                               dim, kvDim, headDim, nHeads, nKVHeads, ffnDim, n, 10000.0f, 1e-6f);
        } else {
            // Fallback: per-op dispatch (~22 per layer)
            mtl_fused_copy(p->a_xIn[li], p->hidden, n*dim);
            mtl_fused_rmsnorm(p->hidden, p->norm1[li], p->a_rmsScale1[li], n, dim);
            mtl_fused_copy(p->a_normed[li], p->hidden, n*dim);
            mtl_fused_copy(p->hidden, p->a_xIn[li], n*dim);

            mtl_fused_gemm_f32_bt(p->a_normed[li], p->wq_live[li], p->a_Q[li], n, dim, dim);
            mtl_fused_gemm_f32_bt(p->a_normed[li], p->wk_live[li], p->a_K[li], n, dim, kvDim);
            mtl_fused_gemm_f32_bt(p->a_normed[li], p->wv_live[li], p->a_V[li], n, dim, kvDim);

            mtl_fused_rope(p->a_Q[li], headDim, nHeads, 10000.0f, dim, n);
            mtl_fused_rope(p->a_K[li], headDim, nKVHeads, 10000.0f, kvDim, n);

            mtl_fused_attn(p->a_Q[li], p->a_K[li], p->a_V[li], p->a_attnOut[li], p->scores,
                          dim, kvDim, headDim, nHeads, nKVHeads, n);

            mtl_fused_gemm_f32_bt(p->a_attnOut[li], p->wo_live[li], p->dScratch, n, dim, dim);
            mtl_fused_add_inplace(p->hidden, p->dScratch, n*dim);

            mtl_fused_copy(p->a_xMid[li], p->hidden, n*dim);
            mtl_fused_rmsnorm(p->hidden, p->norm2[li], p->a_rmsScale2[li], n, dim);
            mtl_fused_copy(p->a_normed2[li], p->hidden, n*dim);
            mtl_fused_copy(p->hidden, p->a_xMid[li], n*dim);

            mtl_fused_gemm_f32_bt(p->a_normed2[li], p->gate_live[li], p->a_gatePre[li], n, dim, ffnDim);
            mtl_fused_gemm_f32_bt(p->a_normed2[li], p->up_live[li], p->a_upOut[li], n, dim, ffnDim);

            mtl_fused_silu_gate_mul(p->a_gatePre[li], p->a_upOut[li], p->a_ffnMid[li], n*ffnDim);

            mtl_fused_gemm_f32_bt(p->a_ffnMid[li], p->down_live[li], p->dScratch, n, ffnDim, dim);
            mtl_fused_add_inplace(p->hidden, p->dScratch, n*dim);
        }
    }

    mtl_fused_rmsnorm(p->hidden, p->finalNorm, p->finalScales, n, dim);
    mtl_fused_copy(p->normedFinal, p->hidden, n*dim);

    mtl_fused_lm_head_pass1(p->normedFinal, p->embed, p->lmMaxLogit, p->lmSumExp, dim, vocabSize, n);
    mtl_fused_barrier_buffers();
    mtl_fused_lm_head_pass2(p->normedFinal, p->embed, p->lmMaxLogit, p->lmSumExp,
                            p->targetsGPU, p->dHidden, p->lmLoss, dim, vocabSize, n);
    mtl_fused_barrier_buffers();
}

// Batch-encode backward + clip + needle into active fused encoder.
static void batch_encode_bwd(ICBBuildParams* p) {
    int dim=p->dim, kvDim=p->kvDim, headDim=p->headDim, nHeads=p->nHeads;
    int nKVHeads=p->nKVHeads, ffnDim=p->ffnDim, n=p->seqLen, nLayers=p->nLayers;

    mtl_fused_rmsnorm_bwd(p->dHidden, p->hidden, p->finalNorm, p->finalScales, p->dScratch, n, dim);
    mtl_fused_copy(p->dHidden, p->dScratch, n*dim);

    for (int li = nLayers - 1; li >= 0; li--) {
        mtl_fused_gemm_f32_nn(p->dHidden, p->down_live[li], p->b_dFfnMid[li], n, dim, ffnDim);
        mtl_fused_gemm_tn_sparse(p->dHidden, p->a_ffnMid[li], p->b_dWDown[li], p->down_mask[li], dim, n, ffnDim);

        mtl_silu_gate_backward_gpu(p->b_dFfnMid[li], p->a_gatePre[li], p->a_upOut[li],
                                    p->a_gateAct[li], p->b_dGate[li], p->b_dUp[li], n*ffnDim);

        mtl_fused_gemm_f32_nn(p->b_dGate[li], p->gate_live[li], p->b_dN2[li], n, ffnDim, dim);
        mtl_fused_gemm_f32_nn(p->b_dUp[li], p->up_live[li], p->b_dx[li], n, ffnDim, dim);
        mtl_fused_add_inplace(p->b_dN2[li], p->b_dx[li], n*dim);

        mtl_fused_gemm_tn_sparse(p->b_dGate[li], p->a_normed2[li], p->b_dWGate[li], p->gate_mask[li], ffnDim, n, dim);
        mtl_fused_gemm_tn_sparse(p->b_dUp[li], p->a_normed2[li], p->b_dWUp[li], p->up_mask[li], ffnDim, n, dim);

        mtl_fused_rmsnorm_bwd(p->b_dN2[li], p->a_xMid[li], p->norm2[li], p->a_rmsScale2[li], p->b_dx[li], n, dim);
        mtl_fused_add_inplace(p->dHidden, p->b_dx[li], n*dim);

        mtl_fused_gemm_f32_nn(p->dHidden, p->wo_live[li], p->b_dAttnOut[li], n, dim, dim);
        mtl_fused_gemm_tn_sparse(p->dHidden, p->a_attnOut[li], p->b_dWO[li], p->wo_mask[li], dim, n, dim);

        mtl_fused_attention_bwd_q(p->b_dAttnOut[li], p->a_Q[li], p->a_K[li], p->a_V[li], p->scores,
                                  p->b_dQ[li], p->b_dK[li], p->b_dV[li], dim, kvDim, headDim, nHeads, nKVHeads, n, n);

        mtl_fused_rope(p->b_dQ[li], headDim, nHeads, -10000.0f, dim, n);
        mtl_fused_rope(p->b_dK[li], headDim, nKVHeads, -10000.0f, kvDim, n);

        mtl_fused_gemm_f32_nn(p->b_dQ[li], p->wq_live[li], p->b_dN1[li], n, dim, dim);
        mtl_fused_gemm_f32_nn(p->b_dK[li], p->wk_live[li], p->b_dx[li], n, kvDim, dim);
        mtl_fused_gemm_f32_nn(p->b_dV[li], p->wv_live[li], p->b_dN2[li], n, kvDim, dim);
        mtl_fused_add_inplace(p->b_dN1[li], p->b_dx[li], n*dim);
        mtl_fused_add_inplace(p->b_dN1[li], p->b_dN2[li], n*dim);

        mtl_fused_gemm_tn_sparse(p->b_dQ[li], p->a_normed[li], p->b_dWQ[li], p->wq_mask[li], dim, n, dim);
        mtl_fused_gemm_tn_sparse(p->b_dK[li], p->a_normed[li], p->b_dWK[li], p->wk_mask[li], kvDim, n, dim);
        mtl_fused_gemm_tn_sparse(p->b_dV[li], p->a_normed[li], p->b_dWV[li], p->wv_mask[li], kvDim, n, dim);

        mtl_fused_rmsnorm_bwd(p->b_dN1[li], p->a_xIn[li], p->norm1[li], p->a_rmsScale1[li], p->b_dx[li], n, dim);
        mtl_fused_add_inplace(p->dHidden, p->b_dx[li], n*dim);
    }

    // Grad norm
    mtl_fused_zero_scalar(p->gradSumSq);
    mtl_fused_barrier_buffers();
    for (int li = 0; li < nLayers; li++) {
        mtl_fused_grad_norm_sq(p->b_dWQ[li], p->gradSumSq, dim*dim);
        mtl_fused_grad_norm_sq(p->b_dWK[li], p->gradSumSq, kvDim*dim);
        mtl_fused_grad_norm_sq(p->b_dWV[li], p->gradSumSq, kvDim*dim);
        mtl_fused_grad_norm_sq(p->b_dWO[li], p->gradSumSq, dim*dim);
        mtl_fused_grad_norm_sq(p->b_dWGate[li], p->gradSumSq, ffnDim*dim);
        mtl_fused_grad_norm_sq(p->b_dWUp[li], p->gradSumSq, ffnDim*dim);
        mtl_fused_grad_norm_sq(p->b_dWDown[li], p->gradSumSq, dim*ffnDim);
    }
    mtl_fused_barrier_buffers();
    mtl_fused_compute_clip_scale(p->gradSumSq, p->clipScaleBuf, 1.0f);
    mtl_fused_barrier_buffers();

    // Needle — encode directly into fused encoder using buffer pointers for constants.
    // The mutable buffers (lr, bc1, bc2, rung) were written by CPU before this call.
    float beta1=0.9f, beta2=0.95f, eps=1e-8f, wd=0.1f;
    id<MTLComputeCommandEncoder> enc = g_fused_enc[g_active_fused_slot];

    for (int li = 0; li < nLayers; li++) {
        void* sd[] = {p->wq_data[li], p->wk_data[li], p->wv_data[li], p->wo_data[li], p->gate_data[li], p->up_data[li], p->down_data[li]};
        void* ss[] = {p->wq_scales[li], p->wk_scales[li], p->wv_scales[li], p->wo_scales[li], p->gate_scales[li], p->up_scales[li], p->down_scales[li]};
        void* sg[] = {p->b_dWQ[li], p->b_dWK[li], p->b_dWV[li], p->b_dWO[li], p->b_dWGate[li], p->b_dWUp[li], p->b_dWDown[li]};
        void* sm[] = {p->wq_mom[li], p->wk_mom[li], p->wv_mom[li], p->wo_mom[li], p->gate_mom[li], p->up_mom[li], p->down_mom[li]};
        void* sv[] = {p->wq_vel[li], p->wk_vel[li], p->wv_vel[li], p->wo_vel[li], p->gate_vel[li], p->up_vel[li], p->down_vel[li]};
        void* sk[] = {p->wq_mask[li], p->wk_mask[li], p->wv_mask[li], p->wo_mask[li], p->gate_mask[li], p->up_mask[li], p->down_mask[li]};
        void* sdl[] = {p->wq_delta[li], p->wk_delta[li], p->wv_delta[li], p->wo_delta[li], p->gate_delta[li], p->up_delta[li], p->down_delta[li]};
        void* sl[] = {p->wq_live[li], p->wk_live[li], p->wv_live[li], p->wo_live[li], p->gate_live[li], p->up_live[li], p->down_live[li]};
        int sn[] = {dim*dim, kvDim*dim, kvDim*dim, dim*dim, ffnDim*dim, ffnDim*dim, dim*ffnDim};
        int sc[] = {dim, dim, dim, dim, dim, dim, ffnDim};

        for (int s = 0; s < 7; s++) {
            uint32_t ucols = sc[s];
            [enc setComputePipelineState:g_ps_needle];
            [enc setBuffer:(__bridge id<MTLBuffer>)sd[s] offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)ss[s] offset:0 atIndex:1];
            [enc setBuffer:(__bridge id<MTLBuffer>)sg[s] offset:0 atIndex:2];
            [enc setBuffer:(__bridge id<MTLBuffer>)sm[s] offset:0 atIndex:3];
            [enc setBuffer:(__bridge id<MTLBuffer>)sv[s] offset:0 atIndex:4];
            [enc setBuffer:(__bridge id<MTLBuffer>)sk[s] offset:0 atIndex:5];
            [enc setBuffer:(__bridge id<MTLBuffer>)sdl[s] offset:0 atIndex:6];
            [enc setBuffer:(__bridge id<MTLBuffer>)p->lrBuf offset:0 atIndex:7];
            [enc setBuffer:const_buf(&beta1, 4) offset:0 atIndex:8];
            [enc setBuffer:const_buf(&beta2, 4) offset:0 atIndex:9];
            [enc setBuffer:(__bridge id<MTLBuffer>)p->bc1Buf offset:0 atIndex:10];
            [enc setBuffer:(__bridge id<MTLBuffer>)p->bc2Buf offset:0 atIndex:11];
            [enc setBuffer:const_buf(&eps, 4) offset:0 atIndex:12];
            [enc setBuffer:const_buf(&wd, 4) offset:0 atIndex:13];
            [enc setBuffer:const_buf(&ucols, 4) offset:0 atIndex:14];
            [enc setBuffer:(__bridge id<MTLBuffer>)sl[s] offset:0 atIndex:15];
            [enc setBuffer:(__bridge id<MTLBuffer>)p->clipScaleBuf offset:0 atIndex:16];
            NSUInteger tpg = g_ps_needle.maxTotalThreadsPerThreadgroup;
            if (tpg > 1024) tpg = 1024;
            [enc dispatchThreads:MTLSizeMake(sn[s], 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        }
    }

    // Embed needle
    {
        uint32_t ucols = dim;
        [enc setComputePipelineState:g_ps_needle];
        [enc setBuffer:(__bridge id<MTLBuffer>)p->embedData offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)p->embedScales offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)p->dEmbed offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)p->embedMom offset:0 atIndex:3];
        [enc setBuffer:(__bridge id<MTLBuffer>)p->embedVel offset:0 atIndex:4];
        [enc setBuffer:(__bridge id<MTLBuffer>)p->embedMask offset:0 atIndex:5];
        [enc setBuffer:(__bridge id<MTLBuffer>)p->embedDelta offset:0 atIndex:6];
        [enc setBuffer:(__bridge id<MTLBuffer>)p->lrBuf offset:0 atIndex:7];
        [enc setBuffer:const_buf(&beta1, 4) offset:0 atIndex:8];
        [enc setBuffer:const_buf(&beta2, 4) offset:0 atIndex:9];
        [enc setBuffer:(__bridge id<MTLBuffer>)p->bc1Buf offset:0 atIndex:10];
        [enc setBuffer:(__bridge id<MTLBuffer>)p->bc2Buf offset:0 atIndex:11];
        [enc setBuffer:const_buf(&eps, 4) offset:0 atIndex:12];
        [enc setBuffer:const_buf(&wd, 4) offset:0 atIndex:13];
        [enc setBuffer:const_buf(&ucols, 4) offset:0 atIndex:14];
        [enc setBuffer:(__bridge id<MTLBuffer>)p->embedLive offset:0 atIndex:15];
        [enc setBuffer:(__bridge id<MTLBuffer>)p->clipScaleBuf offset:0 atIndex:16];
        NSUInteger tpg = g_ps_needle.maxTotalThreadsPerThreadgroup;
        if (tpg > 1024) tpg = 1024;
        [enc dispatchThreads:MTLSizeMake(p->vocabSize * dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    }
}

// ============================================================
// Inference engine — single command buffer, single compute encoder per token.
// Pattern: llama.cpp ggml-metal. Barriers between dependent dispatches.
// ============================================================

// Inline kernels for inference (not needed by training).
static NSString* g_infer_kernel_src =
@"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"// Q8 matvec: 4 rows per threadgroup, 256 threads (8 simdgroups).\n"
"// Q8 matvec: 4 rows per threadgroup, 2 simdgroups per row.\n"
"// Each pair of simdgroups processes 2 adjacent rows sharing activation loads.\n"
"// Q8 matvec: 4 rows per threadgroup, 2 simdgroups per row.\n"
"// Per-row scale, contiguous int8 weights. Vectorized char4 loads.\n"
"kernel void q8_matvec(\n"
"    device const float* act    [[buffer(0)]],  // [K]\n"
"    device const char4* weight [[buffer(1)]],  // [N, K] int8, read as char4\n"
"    device const float* scales [[buffer(2)]],  // [N]\n"
"    device float* out          [[buffer(3)]],  // [N]\n"
"    device const uint* p_K     [[buffer(4)]],\n"
"    device const uint* p_N     [[buffer(5)]],\n"
"    uint tgid [[threadgroup_position_in_grid]],\n"
"    ushort tiisg [[thread_index_in_simdgroup]],\n"
"    ushort sgitg [[simdgroup_index_in_threadgroup]])\n"
"{\n"
"    uint K = p_K[0], N = p_N[0];\n"
"    uint row = tgid * 4 + sgitg / 2;\n"
"    if (row >= N) return;\n"
"    ushort half_sg = sgitg % 2;\n"
"    uint K4 = K / 4;\n"
"    device const char4* wRow = weight + row * K4;\n"
"    device const float4* act4 = (device const float4*)act;\n"
"    float sum = 0.0f;\n"
"    uint tid_in_row = half_sg * 32 + tiisg;\n"
"    for (uint k4 = tid_in_row; k4 < K4; k4 += 64) {\n"
"        float4 a = act4[k4];\n"
"        char4 w = wRow[k4];\n"
"        sum += a.x * w.x + a.y * w.y + a.z * w.z + a.w * w.w;\n"
"    }\n"
"    sum *= scales[row];\n"
"    sum = simd_sum(sum);\n"
"    threadgroup float shmem[8];\n"
"    if (tiisg == 0) shmem[sgitg] = sum;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (sgitg % 2 == 0 && tiisg == 0) {\n"
"        out[row] = shmem[sgitg] + shmem[sgitg + 1];\n"
"    }\n"
"}\n"
"\n"
"// Q4_0 matvec: block_q4_0 format. 4 rows/tg, 2 simdgroups per row.\n"
"// block_q4_0: { half d; uint8_t qs[16]; } = 18 bytes per 32 elements.\n"
"// qs[j] = elem[2j] | (elem[2j+1] << 4). Sequential nibble pairs.\n"
"struct block_q4_0 { half d; uchar qs[16]; };\n"
"kernel void q4_matvec(\n"
"    device const float* act     [[buffer(0)]],\n"
"    device const uchar* weight  [[buffer(1)]],  // block_q4_0 packed\n"
"    device float* out           [[buffer(2)]],\n"
"    device const uint* p_K      [[buffer(3)]],\n"
"    device const uint* p_N      [[buffer(4)]],\n"
"    uint tgid [[threadgroup_position_in_grid]],\n"
"    ushort tiisg [[thread_index_in_simdgroup]],\n"
"    ushort sgitg [[simdgroup_index_in_threadgroup]])\n"
"{\n"
"    uint K = p_K[0], N = p_N[0];\n"
"    uint nb = K / 32;\n"
"    uint row = tgid * 4 + sgitg / 2;\n"
"    if (row >= N) return;\n"
"    ushort half_sg = sgitg % 2;\n"
"    float sum = 0.0f;\n"
"    uint tid_in_row = half_sg * 32 + tiisg;  // 0..63\n"
"    device const block_q4_0* wr = (device const block_q4_0*)(weight + row * nb * 18);\n"
"    for (uint b = tid_in_row; b < nb; b += 64) {\n"
"        float d = float(wr[b].d);\n"
"        float s = 0.0f;\n"
"        uint aOff = b * 32;\n"
"        for (ushort j = 0; j < 16; j += 2) {\n"
"            float4 a = *((device const float4*)(act + aOff + j*2));\n"
"            uchar q0 = wr[b].qs[j];\n"
"            uchar q1 = wr[b].qs[j+1];\n"
"            s += a.x * ((q0 & 0xF) - 8) + a.y * ((q0 >> 4) - 8)\n"
"               + a.z * ((q1 & 0xF) - 8) + a.w * ((q1 >> 4) - 8);\n"
"        }\n"
"        sum += s * d;\n"
"    }\n"
"    sum = simd_sum(sum);\n"
"    threadgroup float shmem[8];\n"
"    if (tiisg == 0) shmem[sgitg] = sum;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (sgitg % 2 == 0 && tiisg == 0) {\n"
"        out[row] = shmem[sgitg] + shmem[sgitg + 1];\n"
"    }\n"
"}\n"
"\n"
"// Rotate-half RoPE at explicit position (for pretrained HF models).\n"
"// x is [1, nHeads*headDim]. Pairs: (x[j], x[j+halfDim]) per head.\n"
"kernel void rope_rotate_half(\n"
"    device float* x          [[buffer(0)]],\n"
"    device const uint* p_hd  [[buffer(1)]],  // headDim\n"
"    device const uint* p_nh  [[buffer(2)]],  // nHeads\n"
"    device const uint* p_pos [[buffer(3)]],  // position\n"
"    device const float* p_th [[buffer(4)]],  // theta\n"
"    uint pid [[thread_position_in_grid]])\n"
"{\n"
"    uint headDim = p_hd[0], nHeads = p_nh[0], pos = p_pos[0];\n"
"    float theta = p_th[0];\n"
"    uint halfDim = headDim / 2;\n"
"    uint h = pid / halfDim;\n"
"    uint j = pid % halfDim;\n"
"    if (h >= nHeads) return;\n"
"    float freq = 1.0f / pow(theta, float(2*j) / float(headDim));\n"
"    float angle = float(pos) * freq;\n"
"    float cosA = cos(angle), sinA = sin(angle);\n"
"    uint base = h * headDim;\n"
"    float x0 = x[base + j], x1 = x[base + j + halfDim];\n"
"    x[base + j]           = x0 * cosA - x1 * sinA;\n"
"    x[base + j + halfDim] = x0 * sinA + x1 * cosA;\n"
"}\n"
"\n"
"// Decode attention: single Q over KV cache.\n"
"// One threadgroup per head, headDim threads.\n"
"kernel void decode_attn(\n"
"    device const float* Q      [[buffer(0)]],  // [1, dim]\n"
"    device const float* kCache [[buffer(1)]],  // [maxSeq, kvDim]\n"
"    device const float* vCache [[buffer(2)]],  // [maxSeq, kvDim]\n"
"    device float* out          [[buffer(3)]],  // [1, dim]\n"
"    device const uint* p_kvDim   [[buffer(4)]],\n"
"    device const uint* p_headDim [[buffer(5)]],\n"
"    device const uint* p_nHeads  [[buffer(6)]],\n"
"    device const uint* p_nKVH   [[buffer(7)]],\n"
"    device const uint* p_seqLen [[buffer(8)]],\n"
"    uint h [[threadgroup_position_in_grid]],\n"
"    uint tid [[thread_index_in_threadgroup]])\n"
"{\n"
"    uint kvDim = p_kvDim[0], headDim = p_headDim[0];\n"
"    uint nHeads = p_nHeads[0], nKVH = p_nKVH[0], seqLen = p_seqLen[0];\n"
"    if (h >= nHeads || tid >= headDim) return;\n"
"    uint kvMul = nHeads / nKVH;\n"
"    uint kvH = h / kvMul;\n"
"    float scale = 1.0f / sqrt(float(headDim));\n"
"    threadgroup float scores[4096];\n"
"    threadgroup float smax[1];\n"
"    threadgroup float ssum[1];\n"
"    uint work = (seqLen + headDim - 1) / headDim;\n"
"    for (uint w = 0; w < work; w++) {\n"
"        uint t = tid + w * headDim;\n"
"        if (t < seqLen) {\n"
"            float dot = 0.0f;\n"
"            for (uint d = 0; d < headDim; d++)\n"
"                dot += Q[h * headDim + d] * kCache[t * kvDim + kvH * headDim + d];\n"
"            scores[t] = dot * scale;\n"
"        }\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (tid == 0) { float mx = -1e30f; for (uint t = 0; t < seqLen; t++) mx = max(mx, scores[t]); smax[0] = mx; }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float mx = smax[0];\n"
"    for (uint w = 0; w < work; w++) { uint t = tid + w * headDim; if (t < seqLen) scores[t] = exp(scores[t] - mx); }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (tid == 0) { float s = 0.0f; for (uint t = 0; t < seqLen; t++) s += scores[t]; ssum[0] = s; }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float invSum = 1.0f / ssum[0];\n"
"    float acc = 0.0f;\n"
"    for (uint t = 0; t < seqLen; t++)\n"
"        acc += scores[t] * invSum * vCache[t * kvDim + kvH * headDim + tid];\n"
"    out[h * headDim + tid] = acc;\n"
"}\n";

static id<MTLComputePipelineState> g_ps_rope_rh = nil;  // rotate-half RoPE
static id<MTLComputePipelineState> g_ps_dec_attn = nil;  // decode attention
static id<MTLComputePipelineState> g_ps_q8mv = nil;      // Q8 fused dequant matvec
static id<MTLComputePipelineState> g_ps_q4mv = nil;      // Q4 fused dequant matvec
static id<MTLComputePipelineState> g_ps_q4mv_simd = nil; // Q4 simd-optimized matvec

// Weight storage: quantized (Q8/Q4) or FP32 (Metal 4 hardware matmul)
typedef struct {
    void* data;   // Q8: [rows*cols] int8. Q4: [rows*cols/2] uint8. FP32: [rows*cols] float.
    void* scales; // [rows] float32 (NULL for FP32 mode)
    int rows, cols;
    bool q4;      // true = 4-bit packed
    bool fp32;    // true = FP32 (Metal 4 path)
} inf_weight_t;

// Per-slot state: KV cache + scratch buffers. Slots run on separate command queues.
#define INF_NUM_SLOTS 2

typedef struct {
    void** kCache; void** vCache;
    void* hidden; void* normed; void* Q; void* K; void* V; void* attnOut;
    void* rmsScale; void* normed2; void* rmsScale2;
    void* gatePre; void* upOut; void* ffnMid; void* proj; void* logits;
} inf_slot_t;

// Inference state
static struct {
    int dim, kvDim, headDim, nHeads, nKVHeads, ffnDim, vocabSize, nLayers, maxSeq;
    bool built, metal4;
    // Per-layer weights (shared across slots — read-only)
    void** norm1; void** norm2;
    inf_weight_t* wq; inf_weight_t* wk; inf_weight_t* wv; inf_weight_t* wo;
    inf_weight_t* wgate; inf_weight_t* wup; inf_weight_t* wdown;
    void** bq; void** bk; void** bv;
    // Final (shared)
    void* finalNorm;
    inf_weight_t lmHead;
    // Per-slot state
    inf_slot_t slots[INF_NUM_SLOTS];
    // Legacy aliases (slot 0) for backward compat
    void** kCache; void** vCache;
    void* hidden; void* normed; void* Q; void* K; void* V; void* attnOut;
    void* rmsScale; void* normed2; void* rmsScale2;
    void* gatePre; void* upOut; void* ffnMid; void* proj; void* logits;
    // Cached constants
    id<MTLBuffer> cb_dim; id<MTLBuffer> cb_kvDim; id<MTLBuffer> cb_headDim;
    id<MTLBuffer> cb_nHeads; id<MTLBuffer> cb_nKVHeads; id<MTLBuffer> cb_ffnDim;
    id<MTLBuffer> cb_eps; id<MTLBuffer> cb_theta;
    id<MTLBuffer> cb_M1; id<MTLBuffer> cb_Kdim; id<MTLBuffer> cb_Nkvdim;
    id<MTLBuffer> cb_Nffn; id<MTLBuffer> cb_Nvocab;
    id<MTLBuffer> cb_pos; id<MTLBuffer> cb_seq; // pre-allocated, written per token
} g_inf = {0};

static bool g_inf_use_q4 = false;

static inf_weight_t inf_weight_alloc(int rows, int cols) {
    inf_weight_t w;
    w.rows = rows; w.cols = cols; w.q4 = g_inf_use_q4; w.fp32 = g_inf.metal4;
    int dataBytes;
    if (w.fp32)       dataBytes = rows * cols * sizeof(float);
    else if (w.q4)    dataBytes = rows * (cols / 32) * 18; // block_q4_0: 18 bytes per 32 elements
    else              dataBytes = rows * cols; // Q8 per-row
    w.data = (__bridge_retained void*)[g_device newBufferWithLength:dataBytes options:MTLResourceStorageModeShared];
    w.scales = (w.fp32 || w.q4) ? NULL : (__bridge_retained void*)[g_device newBufferWithLength:rows * sizeof(float) options:MTLResourceStorageModeShared];
    return w;
}

static void inf_weight_upload(inf_weight_t* w, const float* fp32, int rows, int cols) {
    if (w->fp32) {
        memcpy(((__bridge id<MTLBuffer>)w->data).contents, fp32, rows * cols * sizeof(float));
        return;
    }
    float* scales = (float*)((__bridge id<MTLBuffer>)w->scales).contents;
    if (w->q4) {
        // block_q4_0: per-32-element blocks. {half d, uint8_t qs[16]}
        uint8_t* dst = (uint8_t*)((__bridge id<MTLBuffer>)w->data).contents;
        int nGroups = cols / 32;
        for (int r = 0; r < rows; r++) {
            for (int g = 0; g < nGroups; g++) {
                const float* src = fp32 + r * cols + g * 32;
                float absMax = 0;
                for (int j = 0; j < 32; j++) {
                    float v = src[j]; if (v < 0) v = -v;
                    if (v > absMax) absMax = v;
                }
                uint8_t* block = dst + (r * nGroups + g) * 18;
                __fp16 d = (__fp16)(absMax / 8.0f); // Q4 range: -8..7, scale = absMax/8
                memcpy(block, &d, 2);
                float invScale = absMax > 0 ? 8.0f / absMax : 0;
                uint8_t* qs = block + 2;
                for (int j = 0; j < 16; j++) {
                    float v0 = src[j * 2] * invScale;
                    float v1 = src[j * 2 + 1] * invScale;
                    int q0 = (int)(v0 + (v0 > 0 ? 0.5f : -0.5f)) + 8;
                    int q1 = (int)(v1 + (v1 > 0 ? 0.5f : -0.5f)) + 8;
                    if (q0 < 0) q0 = 0; if (q0 > 15) q0 = 15;
                    if (q1 < 0) q1 = 0; if (q1 > 15) q1 = 15;
                    qs[j] = (uint8_t)(q0 | (q1 << 4));
                }
            }
        }
    } else {
        // Q8 per-row: contiguous int8 + per-row scale (pre-divided by 127)
        int8_t* dst = (int8_t*)((__bridge id<MTLBuffer>)w->data).contents;
        for (int r = 0; r < rows; r++) {
            float absMax = 0;
            for (int c = 0; c < cols; c++) {
                float v = fp32[r * cols + c]; if (v < 0) v = -v;
                if (v > absMax) absMax = v;
            }
            scales[r] = absMax / 127.0f;
            float invScale = absMax > 0 ? 127.0f / absMax : 0;
            for (int c = 0; c < cols; c++) {
                float v = fp32[r * cols + c] * invScale;
                if (v > 127) v = 127; if (v < -127) v = -127;
                dst[r * cols + c] = (int8_t)(v + (v > 0 ? 0.5f : -0.5f));
            }
        }
    }
}

static void* inf_buf(int nFloats) {
    return (__bridge_retained void*)[g_device newBufferWithLength:nFloats * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
}

static id<MTLBuffer> inf_const(uint32_t val) {
    return [g_device newBufferWithBytes:&val length:4 options:MTLResourceStorageModeShared];
}

static id<MTLBuffer> inf_constf(float val) {
    return [g_device newBufferWithBytes:&val length:4 options:MTLResourceStorageModeShared];
}

int mtl_fused_build(int dim, int kvDim, int headDim,
                    int nHeads, int nKVHeads, int ffnDim,
                    int vocabSize, int nLayers, int maxSeq,
                    float ropeTheta, float rmsEps) {
    if (g_inf.built) return 0;
    if (!g_device || !g_ps_rmsnorm_save || !g_ps_gemm_bt || !g_ps_copy_mem) return -1;

    // Load inference kernels from pre-compiled metallib
    NSError* err = nil;
    id<MTLLibrary> lib = nil;
    NSArray* searchPaths = @[
        [[[NSBundle mainBundle] executablePath] stringByDeletingLastPathComponent],
        [[[[NSBundle mainBundle] executablePath] stringByDeletingLastPathComponent] stringByAppendingPathComponent:@"kernels"],
        @".", @"kernels",
        [[NSFileManager defaultManager] currentDirectoryPath]
    ];
    for (NSString* dir in searchPaths) {
        NSString* path = [dir stringByAppendingPathComponent:@"infer.metallib"];
        if ([[NSFileManager defaultManager] fileExistsAtPath:path]) {
            lib = [g_device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
            if (lib) { NSLog(@"mongoose: inference kernels loaded from %@", path); break; }
        }
    }
    if (!lib) {
        // Fallback: compile from source at runtime
        lib = [g_device newLibraryWithSource:g_infer_kernel_src options:nil error:&err];
        if (lib) NSLog(@"mongoose: inference kernels compiled from source");
    }
    if (!lib) { NSLog(@"infer kernel load failed: %@", err); return -1; }
    g_ps_rope_rh = [g_device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rope_rotate_half"] error:&err];
    g_ps_dec_attn = [g_device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"decode_attn"] error:&err];
    g_ps_q8mv = [g_device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q8_matvec"] error:&err];
    g_ps_q4mv = [g_device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q4_matvec"] error:&err];
    if (!g_ps_rope_rh || !g_ps_dec_attn || !g_ps_q8mv || !g_ps_q4mv) { NSLog(@"infer pipeline: %@", err); return -1; }

    // Load Q4 simd-optimized kernel from separate metallib
    for (NSString* dir in searchPaths) {
        NSString* path = [dir stringByAppendingPathComponent:@"q4_simd.metallib"];
        if ([[NSFileManager defaultManager] fileExistsAtPath:path]) {
            id<MTLLibrary> q4lib = [g_device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
            if (q4lib) {
                g_ps_q4mv_simd = [g_device newComputePipelineStateWithFunction:[q4lib newFunctionWithName:@"q4_matvec_simd"] error:&err];
                if (g_ps_q4mv_simd) {
                    NSLog(@"mongoose: Q4 simd kernel loaded from %@", path);
                    break;
                }
            }
        }
    }

    if (maxSeq > 4096) maxSeq = 4096; // decode_attn threadgroup memory limit

    // Metal 4: use FP32 weights + hardware matmul2d TensorOp.
    // Otherwise: Q8 (<4B) or Q4 (>4B) with fused dequant matvec kernel.
    int64_t nParams = (int64_t)vocabSize * dim;
    for (int l = 0; l < nLayers; l++) nParams += (int64_t)dim*dim*2 + (int64_t)kvDim*dim*2 + (int64_t)ffnDim*dim*3;
    g_inf.metal4 = false; // Q8 beats Metal 4 FP32 for decode (M=1) — 4x less memory traffic
    g_inf_use_q4 = (!g_inf.metal4 && nParams > 4000000000LL);
    if (g_inf.metal4)
        NSLog(@"mongoose: inference FP32 Metal 4 matmul2d (%.1fB params)", nParams / 1e9);
    else
        NSLog(@"mongoose: inference Q%d (%.1fB params)", g_inf_use_q4 ? 4 : 8, nParams / 1e9);

    g_inf.dim=dim; g_inf.kvDim=kvDim; g_inf.headDim=headDim;
    g_inf.nHeads=nHeads; g_inf.nKVHeads=nKVHeads; g_inf.ffnDim=ffnDim;
    g_inf.vocabSize=vocabSize; g_inf.nLayers=nLayers; g_inf.maxSeq=maxSeq;

    // Per-layer weights: norms + biases as FP32, matmul weights as Q8
    g_inf.norm1=(void**)calloc(nLayers,8); g_inf.norm2=(void**)calloc(nLayers,8);
    g_inf.bq=(void**)calloc(nLayers,8); g_inf.bk=(void**)calloc(nLayers,8); g_inf.bv=(void**)calloc(nLayers,8);
    g_inf.wq=(inf_weight_t*)calloc(nLayers,sizeof(inf_weight_t));
    g_inf.wk=(inf_weight_t*)calloc(nLayers,sizeof(inf_weight_t));
    g_inf.wv=(inf_weight_t*)calloc(nLayers,sizeof(inf_weight_t));
    g_inf.wo=(inf_weight_t*)calloc(nLayers,sizeof(inf_weight_t));
    g_inf.wgate=(inf_weight_t*)calloc(nLayers,sizeof(inf_weight_t));
    g_inf.wup=(inf_weight_t*)calloc(nLayers,sizeof(inf_weight_t));
    g_inf.wdown=(inf_weight_t*)calloc(nLayers,sizeof(inf_weight_t));
    for (int l = 0; l < nLayers; l++) {
        g_inf.norm1[l]=inf_buf(dim); g_inf.norm2[l]=inf_buf(dim);
        g_inf.bq[l]=inf_buf(dim); g_inf.bk[l]=inf_buf(kvDim); g_inf.bv[l]=inf_buf(kvDim);
        g_inf.wq[l]=inf_weight_alloc(dim, dim);
        g_inf.wk[l]=inf_weight_alloc(kvDim, dim);
        g_inf.wv[l]=inf_weight_alloc(kvDim, dim);
        g_inf.wo[l]=inf_weight_alloc(dim, dim);
        g_inf.wgate[l]=inf_weight_alloc(ffnDim, dim);
        g_inf.wup[l]=inf_weight_alloc(ffnDim, dim);
        g_inf.wdown[l]=inf_weight_alloc(dim, ffnDim);
    }
    g_inf.finalNorm=inf_buf(dim);
    g_inf.lmHead=inf_weight_alloc(vocabSize, dim);

    // Per-slot: KV cache + scratch buffers
    for (int s = 0; s < INF_NUM_SLOTS; s++) {
        inf_slot_t* sl = &g_inf.slots[s];
        sl->kCache=(void**)calloc(nLayers,8); sl->vCache=(void**)calloc(nLayers,8);
        for (int l = 0; l < nLayers; l++) {
            sl->kCache[l]=inf_buf(maxSeq*kvDim); sl->vCache[l]=inf_buf(maxSeq*kvDim);
        }
        sl->hidden=inf_buf(dim); sl->normed=inf_buf(dim);
        sl->Q=inf_buf(dim); sl->K=inf_buf(kvDim); sl->V=inf_buf(kvDim);
        sl->attnOut=inf_buf(dim); sl->rmsScale=inf_buf(1); sl->rmsScale2=inf_buf(1);
        sl->normed2=inf_buf(dim); sl->gatePre=inf_buf(ffnDim);
        sl->upOut=inf_buf(ffnDim); sl->ffnMid=inf_buf(ffnDim);
        sl->proj=inf_buf(dim); sl->logits=inf_buf(vocabSize);
    }

    // Legacy aliases → slot 0 (backward compat with mtl_fused_step)
    g_inf.kCache=g_inf.slots[0].kCache; g_inf.vCache=g_inf.slots[0].vCache;
    g_inf.hidden=g_inf.slots[0].hidden; g_inf.normed=g_inf.slots[0].normed;
    g_inf.Q=g_inf.slots[0].Q; g_inf.K=g_inf.slots[0].K; g_inf.V=g_inf.slots[0].V;
    g_inf.attnOut=g_inf.slots[0].attnOut; g_inf.rmsScale=g_inf.slots[0].rmsScale;
    g_inf.normed2=g_inf.slots[0].normed2; g_inf.rmsScale2=g_inf.slots[0].rmsScale2;
    g_inf.gatePre=g_inf.slots[0].gatePre; g_inf.upOut=g_inf.slots[0].upOut;
    g_inf.ffnMid=g_inf.slots[0].ffnMid; g_inf.proj=g_inf.slots[0].proj;
    g_inf.logits=g_inf.slots[0].logits;

    // Cached constants (allocated once, never change)
    g_inf.cb_dim=inf_const(dim); g_inf.cb_kvDim=inf_const(kvDim);
    g_inf.cb_headDim=inf_const(headDim); g_inf.cb_nHeads=inf_const(nHeads);
    g_inf.cb_nKVHeads=inf_const(nKVHeads); g_inf.cb_ffnDim=inf_const(ffnDim);
    g_inf.cb_eps=inf_constf(rmsEps); g_inf.cb_theta=inf_constf(ropeTheta);
    g_inf.cb_M1=inf_const(1); g_inf.cb_Kdim=inf_const(dim); g_inf.cb_Nkvdim=inf_const(kvDim);
    g_inf.cb_Nffn=inf_const(ffnDim); g_inf.cb_Nvocab=inf_const(vocabSize);
    g_inf.cb_pos=inf_const(0); g_inf.cb_seq=inf_const(0);

    g_inf.built = true;
    return 0;
}

int mtl_fused_num_weights(void) {
    return g_inf.built ? g_inf.nLayers * 12 + 2 : 0;
}

int mtl_fused_set_weight(int idx, const float* data, int nFloats) {
    if (!g_inf.built) return -1;
    int nL = g_inf.nLayers, dim = g_inf.dim, kvDim = g_inf.kvDim, ffnDim = g_inf.ffnDim;
    if (idx < nL * 12) {
        int layer = idx / 12, w = idx % 12;
        switch (w) {
            case 0: // norm1 — FP32
                memcpy(((__bridge id<MTLBuffer>)g_inf.norm1[layer]).contents, data, nFloats*4); break;
            case 1: inf_weight_upload(&g_inf.wq[layer], data, dim, dim); break;
            case 2: inf_weight_upload(&g_inf.wk[layer], data, kvDim, dim); break;
            case 3: inf_weight_upload(&g_inf.wv[layer], data, kvDim, dim); break;
            case 4: memcpy(((__bridge id<MTLBuffer>)g_inf.bq[layer]).contents, data, nFloats*4); break;
            case 5: memcpy(((__bridge id<MTLBuffer>)g_inf.bk[layer]).contents, data, nFloats*4); break;
            case 6: memcpy(((__bridge id<MTLBuffer>)g_inf.bv[layer]).contents, data, nFloats*4); break;
            case 7: inf_weight_upload(&g_inf.wo[layer], data, dim, dim); break;
            case 8: memcpy(((__bridge id<MTLBuffer>)g_inf.norm2[layer]).contents, data, nFloats*4); break;
            case 9: inf_weight_upload(&g_inf.wgate[layer], data, ffnDim, dim); break;
            case 10: inf_weight_upload(&g_inf.wup[layer], data, ffnDim, dim); break;
            case 11: inf_weight_upload(&g_inf.wdown[layer], data, dim, ffnDim); break;
        }
    } else if (idx == nL*12) {
        memcpy(((__bridge id<MTLBuffer>)g_inf.finalNorm).contents, data, nFloats*4);
    } else if (idx == nL*12+1) {
        inf_weight_upload(&g_inf.lmHead, data, g_inf.vocabSize, dim);
    }
    return 0;
}

// Macros for clean dispatch — all use the same encoder `enc`.
#define PS(ps) [enc setComputePipelineState:ps]
#define BUF(ref, idx) [enc setBuffer:(__bridge id<MTLBuffer>)(ref) offset:0 atIndex:idx]
#define BUFO(ref, off, idx) [enc setBuffer:(__bridge id<MTLBuffer>)(ref) offset:(off) atIndex:idx]
#define CB(cb, idx) [enc setBuffer:cb offset:0 atIndex:idx]
#define DT(x,y,z, tx,ty,tz) [enc dispatchThreads:MTLSizeMake(x,y,z) threadsPerThreadgroup:MTLSizeMake(tx,ty,tz)]
#define DTG(gx,gy,gz, tx,ty,tz) [enc dispatchThreadgroups:MTLSizeMake(gx,gy,gz) threadsPerThreadgroup:MTLSizeMake(tx,ty,tz)]
#define BARRIER() [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]
// Matvec dispatch: Metal 4 FP32, Q8 group (8 rows/tg), or Q4.
#define MV(act, w, out, cb_K, cb_N) do { \
    if ((w).fp32) { \
        PS(g_ps_gemm4f_bt); BUF(act, 0); BUF((w).data, 1); BUF(out, 2); \
        CB(g_inf.cb_M1, 3); CB(cb_K, 4); CB(cb_N, 5); \
        NSUInteger sw = g_ps_gemm4f_bt.threadExecutionWidth; \
        DTG(((w).rows+31)/32, 1, 1, sw*4, 1, 1); \
    } else if ((w).q4) { \
        PS(g_ps_q4mv); BUF(act, 0); BUF((w).data, 1); BUF(out, 2); CB(cb_K, 3); CB(cb_N, 4); \
        DTG(((w).rows + 3) / 4, 1, 1, 256, 1, 1); \
    } else { \
        PS(g_ps_q8mv); BUF(act, 0); BUF((w).data, 1); BUF((w).scales, 2); BUF(out, 3); CB(cb_K, 4); CB(cb_N, 5); \
        DTG(((w).rows + 3) / 4, 1, 1, 256, 1, 1); \
    } \
} while(0)

int mtl_fused_step(const float* hiddenIn, const float* cosData, const float* sinData,
                   int pos, float* logitsOut) {
    if (!g_inf.built) return -1;
    int dim=g_inf.dim, kvDim=g_inf.kvDim, headDim=g_inf.headDim;
    int nHeads=g_inf.nHeads, nKVHeads=g_inf.nKVHeads, ffnDim=g_inf.ffnDim;
    int nLayers=g_inf.nLayers, vocabSize=g_inf.vocabSize;
    int seqLen = pos + 1;

    // Upload hidden
    memcpy(((__bridge id<MTLBuffer>)g_inf.hidden).contents, hiddenIn, dim * sizeof(float));

    // Write per-token constants into pre-allocated buffers (no allocation)
    ((uint32_t*)g_inf.cb_pos.contents)[0] = (uint32_t)pos;
    ((uint32_t*)g_inf.cb_seq.contents)[0] = (uint32_t)seqLen;
    id<MTLBuffer> cb_pos = g_inf.cb_pos;
    id<MTLBuffer> cb_seq = g_inf.cb_seq;

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    NSUInteger tpg_norm = (dim / 32) * 32; if (tpg_norm == 0) tpg_norm = 32;
    if (tpg_norm > g_ps_rmsnorm_save.maxTotalThreadsPerThreadgroup) tpg_norm = g_ps_rmsnorm_save.maxTotalThreadsPerThreadgroup;

    for (int l = 0; l < nLayers; l++) {
        // --- Pre-attention: RMSNorm → QKV → bias → RoPE ---

        // RMSNorm out-of-place: normed = rmsnorm(hidden, norm1)
        PS(g_ps_rmsnorm_out); BUF(g_inf.hidden, 0); BUF(g_inf.normed, 1); BUF(g_inf.norm1[l], 2);
        CB(g_inf.cb_dim, 3); CB(g_inf.cb_eps, 4); DTG(1,1,1, tpg_norm,1,1); BARRIER();

        // QKV matvecs (Q8 fused dequant)
        MV(g_inf.normed, g_inf.wq[l], g_inf.Q, g_inf.cb_Kdim, g_inf.cb_dim);
        MV(g_inf.normed, g_inf.wk[l], g_inf.K, g_inf.cb_Kdim, g_inf.cb_Nkvdim);
        MV(g_inf.normed, g_inf.wv[l], g_inf.V, g_inf.cb_Kdim, g_inf.cb_Nkvdim);
        BARRIER();

        // Bias add
        PS(g_ps_bias_add);
        BUF(g_inf.Q, 0); BUF(g_inf.bq[l], 1); CB(g_inf.cb_dim, 2); DT(dim,1,1, 256,1,1);
        BUF(g_inf.K, 0); BUF(g_inf.bk[l], 1); CB(g_inf.cb_kvDim, 2); DT(kvDim,1,1, 256,1,1);
        BUF(g_inf.V, 0); BUF(g_inf.bv[l], 1); CB(g_inf.cb_kvDim, 2); DT(kvDim,1,1, 256,1,1);
        BARRIER();

        // RoPE rotate-half
        int nPQ = nHeads * (headDim / 2), nPK = nKVHeads * (headDim / 2);
        PS(g_ps_rope_rh);
        BUF(g_inf.Q, 0); CB(g_inf.cb_headDim, 1); CB(g_inf.cb_nHeads, 2); CB(cb_pos, 3); CB(g_inf.cb_theta, 4);
        DT(nPQ,1,1, MIN(256,(int)g_ps_rope_rh.maxTotalThreadsPerThreadgroup),1,1);
        BUF(g_inf.K, 0); CB(g_inf.cb_headDim, 1); CB(g_inf.cb_nKVHeads, 2); CB(cb_pos, 3); CB(g_inf.cb_theta, 4);
        DT(nPK,1,1, MIN(256,(int)g_ps_rope_rh.maxTotalThreadsPerThreadgroup),1,1);
        BARRIER();

        // K/V → cache at pos
        int cOff = pos * kvDim * (int)sizeof(float);
        PS(g_ps_copy_mem);
        BUF(g_inf.K, 0); BUFO(g_inf.kCache[l], cOff, 1); DT(kvDim,1,1, 256,1,1);
        BUF(g_inf.V, 0); BUFO(g_inf.vCache[l], cOff, 1); DT(kvDim,1,1, 256,1,1);
        BARRIER();

        // Decode attention
        PS(g_ps_dec_attn);
        BUF(g_inf.Q, 0); BUF(g_inf.kCache[l], 1); BUF(g_inf.vCache[l], 2); BUF(g_inf.attnOut, 3);
        CB(g_inf.cb_kvDim, 4); CB(g_inf.cb_headDim, 5); CB(g_inf.cb_nHeads, 6);
        CB(g_inf.cb_nKVHeads, 7); CB(cb_seq, 8);
        DTG(nHeads,1,1, headDim,1,1); BARRIER();

        // O-proj + residual
        MV(g_inf.attnOut, g_inf.wo[l], g_inf.proj, g_inf.cb_Kdim, g_inf.cb_dim); BARRIER();
        PS(g_ps_add_inplace); BUF(g_inf.hidden, 0); BUF(g_inf.proj, 1);
        DT(dim,1,1, 256,1,1); BARRIER();

        // Post-attn norm: normed2 = rmsnorm(hidden, norm2)
        PS(g_ps_rmsnorm_out); BUF(g_inf.hidden, 0); BUF(g_inf.normed2, 1); BUF(g_inf.norm2[l], 2);
        CB(g_inf.cb_dim, 3); CB(g_inf.cb_eps, 4); DTG(1,1,1, tpg_norm,1,1); BARRIER();

        // FFN
        MV(g_inf.normed2, g_inf.wgate[l], g_inf.gatePre, g_inf.cb_Kdim, g_inf.cb_Nffn);
        MV(g_inf.normed2, g_inf.wup[l], g_inf.upOut, g_inf.cb_Kdim, g_inf.cb_Nffn);
        BARRIER();
        PS(g_ps_silu_gate_mul); BUF(g_inf.gatePre, 0); BUF(g_inf.upOut, 1); BUF(g_inf.ffnMid, 2);
        DT(ffnDim,1,1, 256,1,1); BARRIER();
        MV(g_inf.ffnMid, g_inf.wdown[l], g_inf.proj, g_inf.cb_Nffn, g_inf.cb_dim); BARRIER();
        PS(g_ps_add_inplace); BUF(g_inf.hidden, 0); BUF(g_inf.proj, 1);
        DT(dim,1,1, 256,1,1); BARRIER();
    }

    // Final: RMSNorm hidden in-place, then lm_head (Q8)
    PS(g_ps_rmsnorm_save); BUF(g_inf.hidden, 0); BUF(g_inf.finalNorm, 1); BUF(g_inf.rmsScale, 2);
    CB(g_inf.cb_dim, 3); CB(g_inf.cb_eps, 4); DTG(1,1,1, tpg_norm,1,1); BARRIER();
    MV(g_inf.hidden, g_inf.lmHead, g_inf.logits, g_inf.cb_Kdim, g_inf.cb_Nvocab);

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    memcpy(logitsOut, ((__bridge id<MTLBuffer>)g_inf.logits).contents, vocabSize * sizeof(float));
    return 0;
}

// Reset KV cache — zero all layers in slot 0. Call between conversations.
void mtl_fused_reset_kv(void) {
    if (!g_inf.built) return;
    int kvDim = g_inf.kvDim, maxSeq = g_inf.maxSeq;
    for (int l = 0; l < g_inf.nLayers; l++) {
        memset(((__bridge id<MTLBuffer>)g_inf.slots[0].kCache[l]).contents, 0, maxSeq * kvDim * sizeof(float));
        memset(((__bridge id<MTLBuffer>)g_inf.slots[0].vCache[l]).contents, 0, maxSeq * kvDim * sizeof(float));
    }
}

void mtl_fused_reset_kv_slot(int slot) {
    if (!g_inf.built || slot < 0 || slot >= INF_NUM_SLOTS) return;
    inf_slot_t* sl = &g_inf.slots[slot];
    int kvDim = g_inf.kvDim, maxSeq = g_inf.maxSeq;
    for (int l = 0; l < g_inf.nLayers; l++) {
        memset(((__bridge id<MTLBuffer>)sl->kCache[l]).contents, 0, maxSeq * kvDim * sizeof(float));
        memset(((__bridge id<MTLBuffer>)sl->vCache[l]).contents, 0, maxSeq * kvDim * sizeof(float));
    }
}

int mtl_fused_num_slots(void) { return INF_NUM_SLOTS; }

// Partial step on a specific slot: run layers [layerStart, layerEnd) using slot's
// independent KV cache and scratch buffers. Slot 0 uses g_queue, slot 1 uses g_queue2.
int mtl_fused_partial_step_slot(int slot, const float* hiddenIn, int pos,
                                int layerStart, int layerEnd,
                                float* hiddenOut, float* logitsOut) {
    if (!g_inf.built || slot < 0 || slot >= INF_NUM_SLOTS) return -1;
    inf_slot_t* sl = &g_inf.slots[slot];
    id<MTLCommandQueue> queue = (slot == 0) ? g_queue : g_queue2;

    int dim=g_inf.dim, kvDim=g_inf.kvDim, headDim=g_inf.headDim;
    int nHeads=g_inf.nHeads, nKVHeads=g_inf.nKVHeads, ffnDim=g_inf.ffnDim;
    int nLayers=g_inf.nLayers, vocabSize=g_inf.vocabSize;
    int seqLen = pos + 1;
    if (layerEnd > nLayers) layerEnd = nLayers;

    memcpy(((__bridge id<MTLBuffer>)sl->hidden).contents, hiddenIn, dim * sizeof(float));

    ((uint32_t*)g_inf.cb_pos.contents)[0] = (uint32_t)pos;
    ((uint32_t*)g_inf.cb_seq.contents)[0] = (uint32_t)seqLen;
    id<MTLBuffer> cb_pos = g_inf.cb_pos;
    id<MTLBuffer> cb_seq = g_inf.cb_seq;

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    NSUInteger tpg_norm = (dim / 32) * 32; if (tpg_norm == 0) tpg_norm = 32;
    if (tpg_norm > g_ps_rmsnorm_save.maxTotalThreadsPerThreadgroup) tpg_norm = g_ps_rmsnorm_save.maxTotalThreadsPerThreadgroup;

    for (int l = layerStart; l < layerEnd; l++) {
        PS(g_ps_copy_mem); BUF(sl->hidden, 0); BUF(sl->normed, 1);
        DT(dim,1,1, 256,1,1); BARRIER();

        PS(g_ps_rmsnorm_save); BUF(sl->normed, 0); BUF(g_inf.norm1[l], 1); BUF(sl->rmsScale, 2);
        CB(g_inf.cb_dim, 3); CB(g_inf.cb_eps, 4); DTG(1,1,1, tpg_norm,1,1); BARRIER();

        MV(sl->normed, g_inf.wq[l], sl->Q, g_inf.cb_Kdim, g_inf.cb_dim);
        MV(sl->normed, g_inf.wk[l], sl->K, g_inf.cb_Kdim, g_inf.cb_Nkvdim);
        MV(sl->normed, g_inf.wv[l], sl->V, g_inf.cb_Kdim, g_inf.cb_Nkvdim);
        BARRIER();

        PS(g_ps_bias_add);
        BUF(sl->Q, 0); BUF(g_inf.bq[l], 1); CB(g_inf.cb_dim, 2); DT(dim,1,1, 256,1,1);
        BUF(sl->K, 0); BUF(g_inf.bk[l], 1); CB(g_inf.cb_kvDim, 2); DT(kvDim,1,1, 256,1,1);
        BUF(sl->V, 0); BUF(g_inf.bv[l], 1); CB(g_inf.cb_kvDim, 2); DT(kvDim,1,1, 256,1,1);
        BARRIER();

        int nPQ = nHeads * (headDim / 2), nPK = nKVHeads * (headDim / 2);
        PS(g_ps_rope_rh);
        BUF(sl->Q, 0); CB(g_inf.cb_headDim, 1); CB(g_inf.cb_nHeads, 2); CB(cb_pos, 3); CB(g_inf.cb_theta, 4);
        DT(nPQ,1,1, MIN(256,(int)g_ps_rope_rh.maxTotalThreadsPerThreadgroup),1,1);
        BUF(sl->K, 0); CB(g_inf.cb_headDim, 1); CB(g_inf.cb_nKVHeads, 2); CB(cb_pos, 3); CB(g_inf.cb_theta, 4);
        DT(nPK,1,1, MIN(256,(int)g_ps_rope_rh.maxTotalThreadsPerThreadgroup),1,1);
        BARRIER();

        int cOff = pos * kvDim * (int)sizeof(float);
        PS(g_ps_copy_mem);
        BUF(sl->K, 0); BUFO(sl->kCache[l], cOff, 1); DT(kvDim,1,1, 256,1,1);
        BUF(sl->V, 0); BUFO(sl->vCache[l], cOff, 1); DT(kvDim,1,1, 256,1,1);
        BARRIER();

        PS(g_ps_dec_attn);
        BUF(sl->Q, 0); BUF(sl->kCache[l], 1); BUF(sl->vCache[l], 2); BUF(sl->attnOut, 3);
        CB(g_inf.cb_kvDim, 4); CB(g_inf.cb_headDim, 5); CB(g_inf.cb_nHeads, 6);
        CB(g_inf.cb_nKVHeads, 7); CB(cb_seq, 8);
        DTG(nHeads,1,1, headDim,1,1); BARRIER();

        MV(sl->attnOut, g_inf.wo[l], sl->proj, g_inf.cb_Kdim, g_inf.cb_dim); BARRIER();
        PS(g_ps_add_inplace); BUF(sl->hidden, 0); BUF(sl->proj, 1);
        DT(dim,1,1, 256,1,1); BARRIER();

        PS(g_ps_copy_mem); BUF(sl->hidden, 0); BUF(sl->normed2, 1);
        DT(dim,1,1, 256,1,1); BARRIER();
        PS(g_ps_rmsnorm_save); BUF(sl->normed2, 0); BUF(g_inf.norm2[l], 1); BUF(sl->rmsScale2, 2);
        CB(g_inf.cb_dim, 3); CB(g_inf.cb_eps, 4); DTG(1,1,1, tpg_norm,1,1); BARRIER();

        MV(sl->normed2, g_inf.wgate[l], sl->gatePre, g_inf.cb_Kdim, g_inf.cb_Nffn);
        MV(sl->normed2, g_inf.wup[l], sl->upOut, g_inf.cb_Kdim, g_inf.cb_Nffn);
        BARRIER();
        PS(g_ps_silu_gate_mul); BUF(sl->gatePre, 0); BUF(sl->upOut, 1); BUF(sl->ffnMid, 2);
        DT(ffnDim,1,1, 256,1,1); BARRIER();
        MV(sl->ffnMid, g_inf.wdown[l], sl->proj, g_inf.cb_Nffn, g_inf.cb_dim); BARRIER();
        PS(g_ps_add_inplace); BUF(sl->hidden, 0); BUF(sl->proj, 1);
        DT(dim,1,1, 256,1,1); BARRIER();
    }

    if (layerEnd >= nLayers && logitsOut) {
        PS(g_ps_rmsnorm_save); BUF(sl->hidden, 0); BUF(g_inf.finalNorm, 1); BUF(sl->rmsScale, 2);
        CB(g_inf.cb_dim, 3); CB(g_inf.cb_eps, 4); DTG(1,1,1, tpg_norm,1,1); BARRIER();
        MV(sl->hidden, g_inf.lmHead, sl->logits, g_inf.cb_Kdim, g_inf.cb_Nvocab);
    }

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    if (hiddenOut && layerEnd < nLayers) {
        memcpy(hiddenOut, ((__bridge id<MTLBuffer>)sl->hidden).contents, dim * sizeof(float));
    }
    if (logitsOut && layerEnd >= nLayers) {
        memcpy(logitsOut, ((__bridge id<MTLBuffer>)sl->logits).contents, vocabSize * sizeof(float));
    }
    return 0;
}

// Convenience: partial step on slot 0.
int mtl_fused_partial_step(const float* hiddenIn, int pos,
                           int layerStart, int layerEnd,
                           float* hiddenOut, float* logitsOut) {
    return mtl_fused_partial_step_slot(0, hiddenIn, pos, layerStart, layerEnd, hiddenOut, logitsOut);
}

// Streaming inference state — ping-pong weight buffers for 19x less VRAM.
// Only 2 layers' worth of weights in GPU memory at any time.
static struct {
    bool built;
    inf_weight_t wA[12]; // weight set A (norm1, wq, wk, wv, bq, bk, bv, wo, norm2, gate, up, down)
    inf_weight_t wB[12]; // weight set B (ping-pong)
    void* normA; void* normB;         // norm buffers (FP32, set A/B)
    void* norm2A; void* norm2B;
    void* bqA; void* bkA; void* bvA;
    void* bqB; void* bkB; void* bvB;
} g_stream = {0};

int mtl_stream_build(void) {
    if (!g_inf.built) return -1;
    if (g_stream.built) return 0;
    int dim=g_inf.dim, kvDim=g_inf.kvDim, ffnDim=g_inf.ffnDim;
    // Allocate two sets of weight buffers
    g_stream.wA[0]=inf_weight_alloc(dim, dim);   g_stream.wB[0]=inf_weight_alloc(dim, dim);   // wq
    g_stream.wA[1]=inf_weight_alloc(kvDim, dim); g_stream.wB[1]=inf_weight_alloc(kvDim, dim); // wk
    g_stream.wA[2]=inf_weight_alloc(kvDim, dim); g_stream.wB[2]=inf_weight_alloc(kvDim, dim); // wv
    g_stream.wA[3]=inf_weight_alloc(dim, dim);   g_stream.wB[3]=inf_weight_alloc(dim, dim);   // wo
    g_stream.wA[4]=inf_weight_alloc(ffnDim, dim); g_stream.wB[4]=inf_weight_alloc(ffnDim, dim); // gate
    g_stream.wA[5]=inf_weight_alloc(ffnDim, dim); g_stream.wB[5]=inf_weight_alloc(ffnDim, dim); // up
    g_stream.wA[6]=inf_weight_alloc(dim, ffnDim); g_stream.wB[6]=inf_weight_alloc(dim, ffnDim); // down
    g_stream.normA=inf_buf(dim); g_stream.normB=inf_buf(dim);
    g_stream.norm2A=inf_buf(dim); g_stream.norm2B=inf_buf(dim);
    g_stream.bqA=inf_buf(dim); g_stream.bqB=inf_buf(dim);
    g_stream.bkA=inf_buf(kvDim); g_stream.bkB=inf_buf(kvDim);
    g_stream.bvA=inf_buf(kvDim); g_stream.bvB=inf_buf(kvDim);
    g_stream.built = true;
    return 0;
}

// Upload one layer's weights into the specified ping-pong set (0=A, 1=B).
void mtl_stream_upload_layer(int set, int layer,
    const float* norm1, const float* wq, const float* wk, const float* wv,
    const float* bq, const float* bk, const float* bv, const float* wo,
    const float* norm2, const float* gate, const float* up, const float* down) {
    int dim=g_inf.dim, kvDim=g_inf.kvDim, ffnDim=g_inf.ffnDim;
    inf_weight_t* w = (set == 0) ? g_stream.wA : g_stream.wB;
    void* norm = (set == 0) ? g_stream.normA : g_stream.normB;
    void* n2 = (set == 0) ? g_stream.norm2A : g_stream.norm2B;
    void* bqBuf = (set == 0) ? g_stream.bqA : g_stream.bqB;
    void* bkBuf = (set == 0) ? g_stream.bkA : g_stream.bkB;
    void* bvBuf = (set == 0) ? g_stream.bvA : g_stream.bvB;

    memcpy(((__bridge id<MTLBuffer>)norm).contents, norm1, dim * sizeof(float));
    inf_weight_upload(&w[0], wq, dim, dim);
    inf_weight_upload(&w[1], wk, kvDim, dim);
    inf_weight_upload(&w[2], wv, kvDim, dim);
    if (bq) memcpy(((__bridge id<MTLBuffer>)bqBuf).contents, bq, dim * sizeof(float));
    else    memset(((__bridge id<MTLBuffer>)bqBuf).contents, 0, dim * sizeof(float));
    if (bk) memcpy(((__bridge id<MTLBuffer>)bkBuf).contents, bk, kvDim * sizeof(float));
    else    memset(((__bridge id<MTLBuffer>)bkBuf).contents, 0, kvDim * sizeof(float));
    if (bv) memcpy(((__bridge id<MTLBuffer>)bvBuf).contents, bv, kvDim * sizeof(float));
    else    memset(((__bridge id<MTLBuffer>)bvBuf).contents, 0, kvDim * sizeof(float));
    inf_weight_upload(&w[3], wo, dim, dim);
    memcpy(((__bridge id<MTLBuffer>)n2).contents, norm2, dim * sizeof(float));
    inf_weight_upload(&w[4], gate, ffnDim, dim);
    inf_weight_upload(&w[5], up, ffnDim, dim);
    inf_weight_upload(&w[6], down, dim, ffnDim);
}

// Run one layer using the specified ping-pong set's weights.
// Per-layer command buffer — caller manages the ping-pong timing.
int mtl_stream_step_layer(int set, int layer, int pos) {
    if (!g_inf.built || !g_stream.built) return -1;
    inf_slot_t* sl = &g_inf.slots[0];
    inf_weight_t* w = (set == 0) ? g_stream.wA : g_stream.wB;
    void* norm = (set == 0) ? g_stream.normA : g_stream.normB;
    void* n2 = (set == 0) ? g_stream.norm2A : g_stream.norm2B;
    void* bqBuf = (set == 0) ? g_stream.bqA : g_stream.bqB;
    void* bkBuf = (set == 0) ? g_stream.bkA : g_stream.bkB;
    void* bvBuf = (set == 0) ? g_stream.bvA : g_stream.bvB;

    int dim=g_inf.dim, kvDim=g_inf.kvDim, headDim=g_inf.headDim;
    int nHeads=g_inf.nHeads, nKVHeads=g_inf.nKVHeads, ffnDim=g_inf.ffnDim;
    int seqLen = pos + 1;

    ((uint32_t*)g_inf.cb_pos.contents)[0] = (uint32_t)pos;
    ((uint32_t*)g_inf.cb_seq.contents)[0] = (uint32_t)seqLen;
    id<MTLBuffer> cb_pos = g_inf.cb_pos;
    id<MTLBuffer> cb_seq = g_inf.cb_seq;

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    NSUInteger tpg_norm = (dim / 32) * 32; if (tpg_norm == 0) tpg_norm = 32;
    if (tpg_norm > g_ps_rmsnorm_save.maxTotalThreadsPerThreadgroup) tpg_norm = g_ps_rmsnorm_save.maxTotalThreadsPerThreadgroup;

    PS(g_ps_copy_mem); BUF(sl->hidden, 0); BUF(sl->normed, 1);
    DT(dim,1,1, 256,1,1); BARRIER();

    PS(g_ps_rmsnorm_save); BUF(sl->normed, 0); BUF(norm, 1); BUF(sl->rmsScale, 2);
    CB(g_inf.cb_dim, 3); CB(g_inf.cb_eps, 4); DTG(1,1,1, tpg_norm,1,1); BARRIER();

    MV(sl->normed, w[0], sl->Q, g_inf.cb_Kdim, g_inf.cb_dim);
    MV(sl->normed, w[1], sl->K, g_inf.cb_Kdim, g_inf.cb_Nkvdim);
    MV(sl->normed, w[2], sl->V, g_inf.cb_Kdim, g_inf.cb_Nkvdim);
    BARRIER();

    PS(g_ps_bias_add);
    BUF(sl->Q, 0); BUF(bqBuf, 1); CB(g_inf.cb_dim, 2); DT(dim,1,1, 256,1,1);
    BUF(sl->K, 0); BUF(bkBuf, 1); CB(g_inf.cb_kvDim, 2); DT(kvDim,1,1, 256,1,1);
    BUF(sl->V, 0); BUF(bvBuf, 1); CB(g_inf.cb_kvDim, 2); DT(kvDim,1,1, 256,1,1);
    BARRIER();

    int nPQ = nHeads * (headDim / 2), nPK = nKVHeads * (headDim / 2);
    PS(g_ps_rope_rh);
    BUF(sl->Q, 0); CB(g_inf.cb_headDim, 1); CB(g_inf.cb_nHeads, 2); CB(cb_pos, 3); CB(g_inf.cb_theta, 4);
    DT(nPQ,1,1, MIN(256,(int)g_ps_rope_rh.maxTotalThreadsPerThreadgroup),1,1);
    BUF(sl->K, 0); CB(g_inf.cb_headDim, 1); CB(g_inf.cb_nKVHeads, 2); CB(cb_pos, 3); CB(g_inf.cb_theta, 4);
    DT(nPK,1,1, MIN(256,(int)g_ps_rope_rh.maxTotalThreadsPerThreadgroup),1,1);
    BARRIER();

    int cOff = pos * kvDim * (int)sizeof(float);
    PS(g_ps_copy_mem);
    BUF(sl->K, 0); BUFO(sl->kCache[layer], cOff, 1); DT(kvDim,1,1, 256,1,1);
    BUF(sl->V, 0); BUFO(sl->vCache[layer], cOff, 1); DT(kvDim,1,1, 256,1,1);
    BARRIER();

    PS(g_ps_dec_attn);
    BUF(sl->Q, 0); BUF(sl->kCache[layer], 1); BUF(sl->vCache[layer], 2); BUF(sl->attnOut, 3);
    CB(g_inf.cb_kvDim, 4); CB(g_inf.cb_headDim, 5); CB(g_inf.cb_nHeads, 6);
    CB(g_inf.cb_nKVHeads, 7); CB(cb_seq, 8);
    DTG(nHeads,1,1, headDim,1,1); BARRIER();

    MV(sl->attnOut, w[3], sl->proj, g_inf.cb_Kdim, g_inf.cb_dim); BARRIER();
    PS(g_ps_add_inplace); BUF(sl->hidden, 0); BUF(sl->proj, 1);
    DT(dim,1,1, 256,1,1); BARRIER();

    PS(g_ps_copy_mem); BUF(sl->hidden, 0); BUF(sl->normed2, 1);
    DT(dim,1,1, 256,1,1); BARRIER();
    PS(g_ps_rmsnorm_save); BUF(sl->normed2, 0); BUF(n2, 1); BUF(sl->rmsScale2, 2);
    CB(g_inf.cb_dim, 3); CB(g_inf.cb_eps, 4); DTG(1,1,1, tpg_norm,1,1); BARRIER();

    MV(sl->normed2, w[4], sl->gatePre, g_inf.cb_Kdim, g_inf.cb_Nffn);
    MV(sl->normed2, w[5], sl->upOut, g_inf.cb_Kdim, g_inf.cb_Nffn);
    BARRIER();
    PS(g_ps_silu_gate_mul); BUF(sl->gatePre, 0); BUF(sl->upOut, 1); BUF(sl->ffnMid, 2);
    DT(ffnDim,1,1, 256,1,1); BARRIER();
    MV(sl->ffnMid, w[6], sl->proj, g_inf.cb_Nffn, g_inf.cb_dim); BARRIER();
    PS(g_ps_add_inplace); BUF(sl->hidden, 0); BUF(sl->proj, 1);
    DT(dim,1,1, 256,1,1);

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return 0;
}

// Run final norm + lm_head after all layers. Uses resident finalNorm + lmHead.
int mtl_stream_step_final(int pos, float* logitsOut) {
    if (!g_inf.built) return -1;
    inf_slot_t* sl = &g_inf.slots[0];
    int dim=g_inf.dim, vocabSize=g_inf.vocabSize;

    NSUInteger tpg_norm = (dim / 32) * 32; if (tpg_norm == 0) tpg_norm = 32;
    if (tpg_norm > g_ps_rmsnorm_save.maxTotalThreadsPerThreadgroup) tpg_norm = g_ps_rmsnorm_save.maxTotalThreadsPerThreadgroup;

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    PS(g_ps_rmsnorm_save); BUF(sl->hidden, 0); BUF(g_inf.finalNorm, 1); BUF(sl->rmsScale, 2);
    CB(g_inf.cb_dim, 3); CB(g_inf.cb_eps, 4); DTG(1,1,1, tpg_norm,1,1); BARRIER();
    MV(sl->hidden, g_inf.lmHead, sl->logits, g_inf.cb_Kdim, g_inf.cb_Nvocab);

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    memcpy(logitsOut, ((__bridge id<MTLBuffer>)sl->logits).contents, vocabSize * sizeof(float));
    return 0;
}

// Upload hidden state into slot 0 (call before streaming layer loop).
void mtl_stream_set_hidden(const float* hiddenIn) {
    if (!g_inf.built) return;
    memcpy(((__bridge id<MTLBuffer>)g_inf.slots[0].hidden).contents, hiddenIn, g_inf.dim * sizeof(float));
}

#undef PS
#undef BUF
#undef BUFO
#undef CB
#undef DT
#undef DTG
#undef BARRIER
#undef MV
