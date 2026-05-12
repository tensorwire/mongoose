// mlp_metal_darwin.m — Self-contained Metal MLP training dispatch.
// Loads mlp_train.metallib with ALL needed kernels (matmul, BN, BCE, ReLU, AdamW, etc.)
// Does NOT reference any globals from metal_impl_darwin.m.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

extern id<MTLDevice> g_device;
extern id<MTLCommandQueue> g_queue;

static id<MTLLibrary> g_mlp_lib = nil;

// All pipeline states — loaded from mlp_train.metallib
static id<MTLComputePipelineState> ps_bias_add = nil;
static id<MTLComputePipelineState> ps_bn_sum = nil;
static id<MTLComputePipelineState> ps_bn_var = nil;
static id<MTLComputePipelineState> ps_bn_norm = nil;
static id<MTLComputePipelineState> ps_bn_bwd_stats = nil;
static id<MTLComputePipelineState> ps_bn_bwd_dx = nil;
static id<MTLComputePipelineState> ps_bce = nil;
static id<MTLComputePipelineState> ps_reduce_mean = nil;
static id<MTLComputePipelineState> ps_reduce_mean_rows = nil;
static id<MTLComputePipelineState> ps_dropout_fwd = nil;
static id<MTLComputePipelineState> ps_dropout_bwd = nil;
static id<MTLComputePipelineState> ps_copy = nil;
static id<MTLComputePipelineState> ps_zero = nil;
static id<MTLComputePipelineState> ps_scale = nil;
static id<MTLComputePipelineState> ps_relu = nil;
static id<MTLComputePipelineState> ps_relu_bwd = nil;
static id<MTLComputePipelineState> ps_adamw = nil;
static id<MTLComputePipelineState> ps_gemm_bt = nil;
static id<MTLComputePipelineState> ps_gemm_tn = nil;
static id<MTLComputePipelineState> ps_gemm_nn = nil;

static id<MTLComputePipelineState> mk_ps(NSString* name) {
    id<MTLFunction> fn = [g_mlp_lib newFunctionWithName:name];
    if (!fn) { NSLog(@"[MLP-Metal] '%@' not found", name); return nil; }
    NSError* e = nil;
    id<MTLComputePipelineState> p = [g_device newComputePipelineStateWithFunction:fn error:&e];
    if (!p) { NSLog(@"[MLP-Metal] pipeline '%@': %@", name, e); }
    return p;
}

int mtl_mlp_init_path(const char* path) {
    if (g_mlp_lib) return 1;
    if (!g_device) return 0;

    if (path) {
        NSString* p = [NSString stringWithUTF8String:path];
        if ([[NSFileManager defaultManager] fileExistsAtPath:p]) {
            NSError* e = nil;
            g_mlp_lib = [g_device newLibraryWithURL:[NSURL fileURLWithPath:p] error:&e];
            if (g_mlp_lib) { NSLog(@"[MLP-Metal] loaded %@", p); }
        }
    }
    if (!g_mlp_lib) { NSLog(@"[MLP-Metal] mlp_train.metallib not found at %s", path ? path : "(nil)"); return 0; }

    ps_bias_add       = mk_ps(@"mlp_bias_add");
    ps_bn_sum         = mk_ps(@"mlp_bn_sum");
    ps_bn_var         = mk_ps(@"mlp_bn_var");
    ps_bn_norm        = mk_ps(@"mlp_bn_norm");
    ps_bn_bwd_stats   = mk_ps(@"mlp_bn_bwd_stats");
    ps_bn_bwd_dx      = mk_ps(@"mlp_bn_bwd_dx");
    ps_bce            = mk_ps(@"mlp_bce_with_logits");
    ps_reduce_mean    = mk_ps(@"mlp_reduce_mean");
    ps_reduce_mean_rows = mk_ps(@"mlp_reduce_mean_rows");
    ps_dropout_fwd    = mk_ps(@"mlp_dropout_fwd");
    ps_dropout_bwd    = mk_ps(@"mlp_dropout_bwd");
    ps_copy           = mk_ps(@"mlp_copy");
    ps_zero           = mk_ps(@"mlp_zero");
    ps_scale          = mk_ps(@"mlp_scale_inplace");
    ps_relu           = mk_ps(@"mlp_relu_inplace");
    ps_relu_bwd       = mk_ps(@"mlp_relu_backward");
    ps_adamw          = mk_ps(@"mlp_adamw");
    ps_gemm_bt        = mk_ps(@"mlp_gemm_bt");
    ps_gemm_tn        = mk_ps(@"mlp_gemm_tn");
    ps_gemm_nn        = mk_ps(@"mlp_gemm_nn");

    if (!ps_bias_add || !ps_bn_sum || !ps_bce || !ps_adamw || !ps_gemm_bt) {
        NSLog(@"[MLP-Metal] missing required pipelines"); return 0;
    }
    NSLog(@"[MLP-Metal] 20 pipeline states ready");
    return 1;
}

int mtl_mlp_ready(void) { return ps_bias_add != nil ? 1 : 0; }

// Constant buffer helpers — cached to avoid per-dispatch allocation
static NSMutableDictionary* g_cb_u_cache = nil;
static NSMutableDictionary* g_cb_f_cache = nil;

static id<MTLBuffer> cb_u(uint32_t v) {
    if (!g_cb_u_cache) g_cb_u_cache = [NSMutableDictionary new];
    NSNumber* key = @(v);
    id<MTLBuffer> buf = g_cb_u_cache[key];
    if (!buf) {
        buf = [g_device newBufferWithBytes:&v length:4 options:MTLResourceStorageModeShared];
        g_cb_u_cache[key] = buf;
    }
    return buf;
}

static id<MTLBuffer> cb_f(float v) {
    if (!g_cb_f_cache) g_cb_f_cache = [NSMutableDictionary new];
    NSNumber* key = @(v);
    id<MTLBuffer> buf = g_cb_f_cache[key];
    if (!buf) {
        buf = [g_device newBufferWithBytes:&v length:4 options:MTLResourceStorageModeShared];
        g_cb_f_cache[key] = buf;
    }
    return buf;
}

// GEMM dispatch helpers — each variant uses correct grid dimensions for its output shape
static void dispatch_gemm(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> ps,
    void* a, void* b, void* c, int M, int K, int N, int outRows, int outCols) {
    [enc setComputePipelineState:ps];
    [enc setBuffer:(__bridge id<MTLBuffer>)a offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)b offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)c offset:0 atIndex:2];
    [enc setBuffer:cb_u(M) offset:0 atIndex:3];
    [enc setBuffer:cb_u(K) offset:0 atIndex:4];
    [enc setBuffer:cb_u(N) offset:0 atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake((outCols+31)/32, (outRows+31)/32, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 32, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}
// gemm_bt: C[M,N] = A[M,K] @ B[N,K]^T — output is [M rows, N cols]
#define GEMM_BT(a,b,c, M,K,N) dispatch_gemm(enc, ps_gemm_bt, a,b,c, M,K,N, M, N)
// gemm_tn: C[K,N] = A[M,K]^T @ B[M,N] — output is [K rows, N cols]
#define GEMM_TN(a,b,c, M,K,N) dispatch_gemm(enc, ps_gemm_tn, a,b,c, M,K,N, K, N)
// gemm_nn: C[M,N] = A[M,K] @ B[K,N] — output is [M rows, N cols]
#define GEMM_NN(a,b,c, M,K,N) dispatch_gemm(enc, ps_gemm_nn, a,b,c, M,K,N, M, N)

void mtl_mlp_train_step(
    int nLayers, const int* inDims, const int* outDims, const int* hasBN,
    void** W, void** bias,
    void** gamma, void** beta, void** runMean, void** runVar,
    void** bnMean, void** bnVar,
    void** mW, void** vW, void** mB, void** vB,
    void** mG, void** vG, void** mBt, void** vBt,
    void** act, void** preBN, void** preReLU, void** masks,
    void** dW, void** dB, void** dGamma, void** dBeta,
    void* gradBuf, void* lossBuf, void* lossScalar,
    void* input, void* targets,
    int B, float lr, float wd, float beta1Val, float beta2Val,
    float bc1, float bc2, float bnMom, float eps, float dropP,
    uint32_t dropSeed, uint32_t dropCounter
) {
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    id<MTLBuffer> cB = cb_u(B);
    id<MTLBuffer> cLr = cb_f(lr);
    id<MTLBuffer> cWd = cb_f(wd);
    id<MTLBuffer> cB1 = cb_f(beta1Val);
    id<MTLBuffer> cB2 = cb_f(beta2Val);
    id<MTLBuffer> cBc1 = cb_f(bc1);
    id<MTLBuffer> cBc2 = cb_f(bc2);
    id<MTLBuffer> cEps = cb_f(eps);
    id<MTLBuffer> cMom = cb_f(bnMom);
    id<MTLBuffer> cBnE = cb_f(1e-5f);
    id<MTLBuffer> cWd0 = cb_f(0.0f);

    // ======================== FORWARD ========================
    for (int li = 0; li < nLayers; li++) {
        int iD = inDims[li], oD = outDims[li];
        int last = (li == nLayers - 1);
        int n = B * oD;
        id<MTLBuffer> cD = cb_u(oD);
        void* layerIn = (li == 0) ? input : act[li-1];

        // Matmul: act = layerIn @ W^T
        GEMM_BT(layerIn, W[li], act[li], B, iD, oD);

        // Bias
        [enc setComputePipelineState:ps_bias_add];
        [enc setBuffer:(__bridge id<MTLBuffer>)act[li] offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)bias[li] offset:0 atIndex:1];
        [enc setBuffer:cB offset:0 atIndex:2]; [enc setBuffer:cD offset:0 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // Save pre-BN
        [enc setComputePipelineState:ps_copy];
        [enc setBuffer:(__bridge id<MTLBuffer>)preBN[li] offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)act[li] offset:0 atIndex:1];
        [enc setBuffer:cb_u(n) offset:0 atIndex:2];
        [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // BatchNorm
        if (hasBN[li] && !last) {
            int tpg = B < 1024 ? ((B+31)/32)*32 : 1024;

            [enc setComputePipelineState:ps_bn_sum];
            [enc setBuffer:(__bridge id<MTLBuffer>)act[li] offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)bnMean[li] offset:0 atIndex:1];
            [enc setBuffer:cB offset:0 atIndex:2]; [enc setBuffer:cD offset:0 atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(oD,1,1) threadsPerThreadgroup:MTLSizeMake(tpg,1,1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            [enc setComputePipelineState:ps_bn_var];
            [enc setBuffer:(__bridge id<MTLBuffer>)act[li] offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)bnMean[li] offset:0 atIndex:1];
            [enc setBuffer:(__bridge id<MTLBuffer>)bnVar[li] offset:0 atIndex:2];
            [enc setBuffer:cB offset:0 atIndex:3]; [enc setBuffer:cD offset:0 atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(oD,1,1) threadsPerThreadgroup:MTLSizeMake(tpg,1,1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            [enc setComputePipelineState:ps_bn_norm];
            [enc setBuffer:(__bridge id<MTLBuffer>)act[li] offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)bnMean[li] offset:0 atIndex:1];
            [enc setBuffer:(__bridge id<MTLBuffer>)bnVar[li] offset:0 atIndex:2];
            [enc setBuffer:(__bridge id<MTLBuffer>)gamma[li] offset:0 atIndex:3];
            [enc setBuffer:(__bridge id<MTLBuffer>)beta[li] offset:0 atIndex:4];
            [enc setBuffer:(__bridge id<MTLBuffer>)runMean[li] offset:0 atIndex:5];
            [enc setBuffer:(__bridge id<MTLBuffer>)runVar[li] offset:0 atIndex:6];
            [enc setBuffer:cB offset:0 atIndex:7]; [enc setBuffer:cD offset:0 atIndex:8];
            [enc setBuffer:cMom offset:0 atIndex:9]; [enc setBuffer:cBnE offset:0 atIndex:10];
            [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }

        if (!last) {
            // Save pre-ReLU
            [enc setComputePipelineState:ps_copy];
            [enc setBuffer:(__bridge id<MTLBuffer>)preReLU[li] offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)act[li] offset:0 atIndex:1];
            [enc setBuffer:cb_u(n) offset:0 atIndex:2];
            [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // ReLU
            [enc setComputePipelineState:ps_relu];
            [enc setBuffer:(__bridge id<MTLBuffer>)act[li] offset:0 atIndex:0];
            [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // Dropout
            if (dropP > 0) {
                [enc setComputePipelineState:ps_dropout_fwd];
                [enc setBuffer:(__bridge id<MTLBuffer>)act[li] offset:0 atIndex:0];
                [enc setBuffer:(__bridge id<MTLBuffer>)masks[li] offset:0 atIndex:1];
                [enc setBuffer:cb_u(n) offset:0 atIndex:2];
                [enc setBuffer:cb_f(dropP) offset:0 atIndex:3];
                [enc setBuffer:cb_u(dropSeed) offset:0 atIndex:4];
                [enc setBuffer:cb_u(dropCounter+li) offset:0 atIndex:5];
                [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
        }
    }

    // ======================== LOSS ========================
    [enc setComputePipelineState:ps_bce];
    [enc setBuffer:(__bridge id<MTLBuffer>)act[nLayers-1] offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)targets offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)gradBuf offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)lossBuf offset:0 atIndex:3];
    [enc setBuffer:cB offset:0 atIndex:4];
    [enc dispatchThreads:MTLSizeMake(B,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    {
        int tpg = B < 1024 ? ((B+31)/32)*32 : 1024;
        [enc setComputePipelineState:ps_reduce_mean];
        [enc setBuffer:(__bridge id<MTLBuffer>)lossBuf offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)lossScalar offset:0 atIndex:1];
        [enc setBuffer:cB offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(tpg,1,1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }

    // ======================== BACKWARD ========================
    for (int li = nLayers-1; li >= 0; li--) {
        int iD = inDims[li], oD = outDims[li];
        int last = (li == nLayers-1);
        int n = B * oD;
        id<MTLBuffer> cD = cb_u(oD);
        void* curGrad = last ? gradBuf : act[li];

        if (!last && dropP > 0) {
            [enc setComputePipelineState:ps_dropout_bwd];
            [enc setBuffer:(__bridge id<MTLBuffer>)curGrad offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)masks[li] offset:0 atIndex:1];
            [enc setBuffer:cb_u(n) offset:0 atIndex:2];
            [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }

        if (!last) {
            [enc setComputePipelineState:ps_relu_bwd];
            [enc setBuffer:(__bridge id<MTLBuffer>)curGrad offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)curGrad offset:0 atIndex:1];
            [enc setBuffer:(__bridge id<MTLBuffer>)preReLU[li] offset:0 atIndex:2];
            [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }

        if (hasBN[li] && !last) {
            int tpg = B < 1024 ? ((B+31)/32)*32 : 1024;

            [enc setComputePipelineState:ps_bn_bwd_stats];
            [enc setBuffer:(__bridge id<MTLBuffer>)curGrad offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)preBN[li] offset:0 atIndex:1];
            [enc setBuffer:(__bridge id<MTLBuffer>)bnMean[li] offset:0 atIndex:2];
            [enc setBuffer:(__bridge id<MTLBuffer>)bnVar[li] offset:0 atIndex:3];
            [enc setBuffer:(__bridge id<MTLBuffer>)dGamma[li] offset:0 atIndex:4];
            [enc setBuffer:(__bridge id<MTLBuffer>)dBeta[li] offset:0 atIndex:5];
            [enc setBuffer:cB offset:0 atIndex:6]; [enc setBuffer:cD offset:0 atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(oD,1,1) threadsPerThreadgroup:MTLSizeMake(tpg,1,1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            [enc setComputePipelineState:ps_bn_bwd_dx];
            [enc setBuffer:(__bridge id<MTLBuffer>)curGrad offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)preBN[li] offset:0 atIndex:1];
            [enc setBuffer:(__bridge id<MTLBuffer>)bnMean[li] offset:0 atIndex:2];
            [enc setBuffer:(__bridge id<MTLBuffer>)bnVar[li] offset:0 atIndex:3];
            [enc setBuffer:(__bridge id<MTLBuffer>)gamma[li] offset:0 atIndex:4];
            [enc setBuffer:(__bridge id<MTLBuffer>)dGamma[li] offset:0 atIndex:5];
            [enc setBuffer:(__bridge id<MTLBuffer>)dBeta[li] offset:0 atIndex:6];
            [enc setBuffer:cB offset:0 atIndex:7]; [enc setBuffer:cD offset:0 atIndex:8];
            [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }

        // Zero dW, compute weight gradient
        int nW = oD * iD;
        [enc setComputePipelineState:ps_zero];
        [enc setBuffer:(__bridge id<MTLBuffer>)dW[li] offset:0 atIndex:0];
        [enc dispatchThreads:MTLSizeMake(nW,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        void* layerIn = (li == 0) ? input : act[li-1];
        GEMM_TN(curGrad, layerIn, dW[li], B, oD, iD);

        // Scale dW by 1/B
        [enc setComputePipelineState:ps_scale];
        [enc setBuffer:(__bridge id<MTLBuffer>)dW[li] offset:0 atIndex:0];
        [enc setBuffer:cb_f(1.0f/(float)B) offset:0 atIndex:1];
        [enc dispatchThreads:MTLSizeMake(nW,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // Bias gradient
        {
            int tpg = B < 1024 ? ((B+31)/32)*32 : 1024;
            [enc setComputePipelineState:ps_reduce_mean_rows];
            [enc setBuffer:(__bridge id<MTLBuffer>)curGrad offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)dB[li] offset:0 atIndex:1];
            [enc setBuffer:cB offset:0 atIndex:2]; [enc setBuffer:cD offset:0 atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(oD,1,1) threadsPerThreadgroup:MTLSizeMake(tpg,1,1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }

        // Input gradient
        if (li > 0) {
            GEMM_NN(curGrad, W[li], act[li-1], B, oD, iD);
        }

        // AdamW: W
        [enc setComputePipelineState:ps_adamw];
        [enc setBuffer:(__bridge id<MTLBuffer>)W[li] offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)dW[li] offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)mW[li] offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)vW[li] offset:0 atIndex:3];
        [enc setBuffer:cLr offset:0 atIndex:4]; [enc setBuffer:cB1 offset:0 atIndex:5];
        [enc setBuffer:cB2 offset:0 atIndex:6]; [enc setBuffer:cBc1 offset:0 atIndex:7];
        [enc setBuffer:cBc2 offset:0 atIndex:8]; [enc setBuffer:cEps offset:0 atIndex:9];
        [enc setBuffer:cWd offset:0 atIndex:10];
        [enc dispatchThreads:MTLSizeMake(nW,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // AdamW: bias
        [enc setBuffer:(__bridge id<MTLBuffer>)bias[li] offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)dB[li] offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)mB[li] offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)vB[li] offset:0 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(oD,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // AdamW: BN gamma/beta
        if (hasBN[li] && !last) {
            [enc setBuffer:(__bridge id<MTLBuffer>)gamma[li] offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)dGamma[li] offset:0 atIndex:1];
            [enc setBuffer:(__bridge id<MTLBuffer>)mG[li] offset:0 atIndex:2];
            [enc setBuffer:(__bridge id<MTLBuffer>)vG[li] offset:0 atIndex:3];
            [enc setBuffer:cWd0 offset:0 atIndex:10];
            [enc dispatchThreads:MTLSizeMake(oD,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            [enc setBuffer:(__bridge id<MTLBuffer>)beta[li] offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)dBeta[li] offset:0 atIndex:1];
            [enc setBuffer:(__bridge id<MTLBuffer>)mBt[li] offset:0 atIndex:2];
            [enc setBuffer:(__bridge id<MTLBuffer>)vBt[li] offset:0 atIndex:3];
            [enc dispatchThreads:MTLSizeMake(oD,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
    }

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

// ======================== TEST DISPATCH HELPERS ========================

void mtl_mlp_gemm_bt(void* a, void* b, void* c, int M, int K, int N) {
    if (!ps_gemm_bt) return;
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    GEMM_BT(a, b, c, M, K, N);
    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
}

void mtl_mlp_gemm_tn(void* a, void* b, void* c, int M, int K, int N) {
    if (!ps_gemm_tn) return;
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    GEMM_TN(a, b, c, M, K, N);
    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
}

void mtl_mlp_gemm_nn(void* a, void* b, void* c, int M, int K, int N) {
    if (!ps_gemm_nn) return;
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    GEMM_NN(a, b, c, M, K, N);
    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
}

void mtl_mlp_bias_add(void* out, void* bias, int B, int D) {
    if (!ps_bias_add) return;
    int n = B * D;
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:ps_bias_add];
    [enc setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)bias offset:0 atIndex:1];
    [enc setBuffer:cb_u(B) offset:0 atIndex:2];
    [enc setBuffer:cb_u(D) offset:0 atIndex:3];
    [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
}

void mtl_mlp_bn_forward(void* x, void* mean, void* var, void* gamma, void* beta,
    void* runMean, void* runVar, int B, int D, float momentum) {
    if (!ps_bn_sum) return;
    int n = B * D;
    int tpg = B < 1024 ? ((B+31)/32)*32 : 1024;
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:ps_bn_sum];
    [enc setBuffer:(__bridge id<MTLBuffer>)x offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)mean offset:0 atIndex:1];
    [enc setBuffer:cb_u(B) offset:0 atIndex:2];
    [enc setBuffer:cb_u(D) offset:0 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(D,1,1) threadsPerThreadgroup:MTLSizeMake(tpg,1,1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [enc setComputePipelineState:ps_bn_var];
    [enc setBuffer:(__bridge id<MTLBuffer>)x offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)mean offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)var offset:0 atIndex:2];
    [enc setBuffer:cb_u(B) offset:0 atIndex:3];
    [enc setBuffer:cb_u(D) offset:0 atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(D,1,1) threadsPerThreadgroup:MTLSizeMake(tpg,1,1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [enc setComputePipelineState:ps_bn_norm];
    [enc setBuffer:(__bridge id<MTLBuffer>)x offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)mean offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)var offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)gamma offset:0 atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)beta offset:0 atIndex:4];
    [enc setBuffer:(__bridge id<MTLBuffer>)runMean offset:0 atIndex:5];
    [enc setBuffer:(__bridge id<MTLBuffer>)runVar offset:0 atIndex:6];
    [enc setBuffer:cb_u(B) offset:0 atIndex:7];
    [enc setBuffer:cb_u(D) offset:0 atIndex:8];
    [enc setBuffer:cb_f(momentum) offset:0 atIndex:9];
    [enc setBuffer:cb_f(1e-5f) offset:0 atIndex:10];
    [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];

    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
}

void mtl_mlp_bn_backward(void* dOut, void* x, void* mean, void* var,
    void* gamma, void* dGamma, void* dBeta, int B, int D) {
    if (!ps_bn_bwd_stats) return;
    int n = B * D;
    int tpg = B < 1024 ? ((B+31)/32)*32 : 1024;
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:ps_bn_bwd_stats];
    [enc setBuffer:(__bridge id<MTLBuffer>)dOut offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)x offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)mean offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)var offset:0 atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)dGamma offset:0 atIndex:4];
    [enc setBuffer:(__bridge id<MTLBuffer>)dBeta offset:0 atIndex:5];
    [enc setBuffer:cb_u(B) offset:0 atIndex:6];
    [enc setBuffer:cb_u(D) offset:0 atIndex:7];
    [enc dispatchThreadgroups:MTLSizeMake(D,1,1) threadsPerThreadgroup:MTLSizeMake(tpg,1,1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [enc setComputePipelineState:ps_bn_bwd_dx];
    [enc setBuffer:(__bridge id<MTLBuffer>)dOut offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)x offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)mean offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)var offset:0 atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)gamma offset:0 atIndex:4];
    [enc setBuffer:(__bridge id<MTLBuffer>)dGamma offset:0 atIndex:5];
    [enc setBuffer:(__bridge id<MTLBuffer>)dBeta offset:0 atIndex:6];
    [enc setBuffer:cb_u(B) offset:0 atIndex:7];
    [enc setBuffer:cb_u(D) offset:0 atIndex:8];
    [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];

    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
}

void mtl_mlp_bce(void* logits, void* targets, void* grad, void* lossBuf,
    void* lossScalar, int B) {
    if (!ps_bce) return;
    int tpg = B < 1024 ? ((B+31)/32)*32 : 1024;
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:ps_bce];
    [enc setBuffer:(__bridge id<MTLBuffer>)logits offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)targets offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)grad offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)lossBuf offset:0 atIndex:3];
    [enc setBuffer:cb_u(B) offset:0 atIndex:4];
    [enc dispatchThreads:MTLSizeMake(B,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [enc setComputePipelineState:ps_reduce_mean];
    [enc setBuffer:(__bridge id<MTLBuffer>)lossBuf offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)lossScalar offset:0 atIndex:1];
    [enc setBuffer:cb_u(B) offset:0 atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(tpg,1,1)];

    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
}

void mtl_mlp_adamw_step(void* param, void* grad, void* m, void* v,
    float lr, float beta1, float beta2, float bc1, float bc2, float eps, float wd, int n) {
    if (!ps_adamw) return;
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:ps_adamw];
    [enc setBuffer:(__bridge id<MTLBuffer>)param offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)grad offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)m offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)v offset:0 atIndex:3];
    [enc setBuffer:cb_f(lr) offset:0 atIndex:4];
    [enc setBuffer:cb_f(beta1) offset:0 atIndex:5];
    [enc setBuffer:cb_f(beta2) offset:0 atIndex:6];
    [enc setBuffer:cb_f(bc1) offset:0 atIndex:7];
    [enc setBuffer:cb_f(bc2) offset:0 atIndex:8];
    [enc setBuffer:cb_f(eps) offset:0 atIndex:9];
    [enc setBuffer:cb_f(wd) offset:0 atIndex:10];
    [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
}

void mtl_mlp_dropout_fwd(void* x, void* mask, int n, float p, uint32_t seed, uint32_t counter) {
    if (!ps_dropout_fwd) return;
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:ps_dropout_fwd];
    [enc setBuffer:(__bridge id<MTLBuffer>)x offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)mask offset:0 atIndex:1];
    [enc setBuffer:cb_u(n) offset:0 atIndex:2];
    [enc setBuffer:cb_f(p) offset:0 atIndex:3];
    [enc setBuffer:cb_u(seed) offset:0 atIndex:4];
    [enc setBuffer:cb_u(counter) offset:0 atIndex:5];
    [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
}

void mtl_mlp_dropout_bwd(void* dx, void* mask, int n) {
    if (!ps_dropout_bwd) return;
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:ps_dropout_bwd];
    [enc setBuffer:(__bridge id<MTLBuffer>)dx offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)mask offset:0 atIndex:1];
    [enc setBuffer:cb_u(n) offset:0 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
}
