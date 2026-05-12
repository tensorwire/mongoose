// sq4_infer_metal_darwin.m — Standalone SQ4 fused inference engine.
// One command buffer per token, zero CPU in hot path.
// Uses shared kernels from the inline library (via mtl_make_pipeline)
// and SQ4 matvec from sq4_matvec.metallib.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

extern id<MTLDevice> g_device;
extern id<MTLCommandQueue> g_queue;

// SQ4 pipeline states — loaded from infer.metallib
static id<MTLComputePipelineState> g_sq4_ps_matvec = nil;
static id<MTLComputePipelineState> g_sq4_ps_outlier = nil;

// From metal_impl_darwin.m
extern id<MTLComputePipelineState> mtl_make_pipeline(NSString* name);

// SQ4 weight
typedef struct {
    void* packed;       // [rows*cols/2] uint8
    void* bands;        // [8] float32
    void* outlier_idx;  // [outlier_count] uint32 (may be NULL)
    void* outlier_val;  // [outlier_count] float32 (may be NULL)
    int rows, cols;
    int outlier_count;
} sq4_weight_t;

// Inference state
static struct {
    int dim, kvDim, headDim, nHeads, nKVHeads, ffnDim, vocabSize, nLayers, maxSeq;
    bool built;

    // Per-layer
    void** norm1; void** norm2;
    void** bq; void** bk; void** bv;
    sq4_weight_t* wq; sq4_weight_t* wk; sq4_weight_t* wv; sq4_weight_t* wo;
    sq4_weight_t* wgate; sq4_weight_t* wup; sq4_weight_t* wdown;

    // Final
    void* finalNorm;
    sq4_weight_t lmHead;

    // KV cache
    void** kCache; void** vCache;

    // Scratch buffers
    void* hidden; void* normed; void* Q; void* K; void* V; void* attnOut;
    void* normed2; void* gatePre; void* upOut; void* ffnMid; void* proj; void* logits;

    // Cached constants
    id<MTLBuffer> cb_dim; id<MTLBuffer> cb_kvDim; id<MTLBuffer> cb_headDim;
    id<MTLBuffer> cb_nHeads; id<MTLBuffer> cb_nKVHeads;
    id<MTLBuffer> cb_ffnDim; id<MTLBuffer> cb_eps; id<MTLBuffer> cb_theta;
    id<MTLBuffer> cb_pos; id<MTLBuffer> cb_seq;
    id<MTLBuffer> cb_Ndim; id<MTLBuffer> cb_Nkvdim; id<MTLBuffer> cb_Nffn; id<MTLBuffer> cb_Nvocab;

    // Shared pipeline states (from inline library)
    id<MTLComputePipelineState> ps_rmsnorm_out;
    id<MTLComputePipelineState> ps_rmsnorm_save;
    id<MTLComputePipelineState> ps_rope_rh;
    id<MTLComputePipelineState> ps_dec_attn;
    id<MTLComputePipelineState> ps_silu_gate_mul;
    id<MTLComputePipelineState> ps_add_inplace;
    id<MTLComputePipelineState> ps_copy_mem;
    id<MTLComputePipelineState> ps_bias_add;
} g_sq4 = {0};

static id<MTLBuffer> sq4i_buf(int nFloats) {
    return [g_device newBufferWithLength:nFloats * sizeof(float) options:MTLResourceStorageModeShared];
}

static id<MTLBuffer> sq4i_const(uint32_t v) {
    return [g_device newBufferWithBytes:&v length:4 options:MTLResourceStorageModeShared];
}

static id<MTLBuffer> sq4i_constf(float v) {
    return [g_device newBufferWithBytes:&v length:4 options:MTLResourceStorageModeShared];
}

int mtl_sq4_infer_build(int dim, int kvDim, int headDim,
    int nHeads, int nKVHeads, int ffnDim,
    int vocabSize, int nLayers, int maxSeq,
    float ropeTheta, float rmsEps) {
    if (g_sq4.built) return 0;
    if (!g_device) return -1;

    g_sq4.dim=dim; g_sq4.kvDim=kvDim; g_sq4.headDim=headDim;
    g_sq4.nHeads=nHeads; g_sq4.nKVHeads=nKVHeads; g_sq4.ffnDim=ffnDim;
    g_sq4.vocabSize=vocabSize; g_sq4.nLayers=nLayers; g_sq4.maxSeq=maxSeq;

    // Load ALL pipeline states from infer.metallib
    NSArray* searchPaths = @[
        [[[NSBundle mainBundle] executablePath] stringByDeletingLastPathComponent],
        [[[[NSBundle mainBundle] executablePath] stringByDeletingLastPathComponent] stringByAppendingPathComponent:@"kernels"],
        @"kernels", @".",
        [[NSString stringWithUTF8String:getenv("HOME") ?: ""] stringByAppendingPathComponent:@"go/src/github.com/tensorwire/mongoose/kernels"]
    ];
    id<MTLLibrary> inferLib = nil;
    for (NSString* dir in searchPaths) {
        NSString* p = [dir stringByAppendingPathComponent:@"infer.metallib"];
        if ([[NSFileManager defaultManager] fileExistsAtPath:p]) {
            NSError* e = nil;
            inferLib = [g_device newLibraryWithURL:[NSURL fileURLWithPath:p] error:&e];
            if (inferLib) { NSLog(@"[SQ4-Infer] loaded %@", p); break; }
        }
    }
    if (!inferLib) { NSLog(@"[SQ4-Infer] infer.metallib not found"); return -1; }

    id<MTLComputePipelineState> (^mkps)(NSString*) = ^(NSString* name) {
        id<MTLFunction> fn = [inferLib newFunctionWithName:name];
        if (!fn) { NSLog(@"[SQ4-Infer] kernel %@ not found", name); return (id<MTLComputePipelineState>)nil; }
        NSError* e = nil;
        return [g_device newComputePipelineStateWithFunction:fn error:&e];
    };

    g_sq4.ps_rope_rh = mkps(@"rope_rotate_half");
    g_sq4.ps_dec_attn = mkps(@"decode_attn");
    g_sq4_ps_matvec = mkps(@"sq4_matvec");
    g_sq4_ps_outlier = mkps(@"sq4_outlier_correct");
    if (!g_sq4_ps_matvec) { NSLog(@"[SQ4-Infer] sq4_matvec not in infer.metallib"); return -1; }

    // Shared kernels from inline compute library
    g_sq4.ps_rmsnorm_out = mtl_make_pipeline(@"rmsnorm_out");
    g_sq4.ps_rmsnorm_save = mtl_make_pipeline(@"rmsnorm_save");
    g_sq4.ps_silu_gate_mul = mtl_make_pipeline(@"silu_gate_mul");
    g_sq4.ps_add_inplace = mtl_make_pipeline(@"add_inplace");
    g_sq4.ps_copy_mem = mtl_make_pipeline(@"copy_mem");
    g_sq4.ps_bias_add = mtl_make_pipeline(@"bias_add");

    if (!g_sq4.ps_rmsnorm_out || !g_sq4.ps_rope_rh || !g_sq4.ps_dec_attn || !g_sq4.ps_silu_gate_mul) {
        NSLog(@"[SQ4-Infer] missing pipeline states");
        return -1;
    }

    // Allocate per-layer storage
    g_sq4.norm1 = (void**)calloc(nLayers, 8);
    g_sq4.norm2 = (void**)calloc(nLayers, 8);
    g_sq4.bq = (void**)calloc(nLayers, 8);
    g_sq4.bk = (void**)calloc(nLayers, 8);
    g_sq4.bv = (void**)calloc(nLayers, 8);
    g_sq4.wq = (sq4_weight_t*)calloc(nLayers, sizeof(sq4_weight_t));
    g_sq4.wk = (sq4_weight_t*)calloc(nLayers, sizeof(sq4_weight_t));
    g_sq4.wv = (sq4_weight_t*)calloc(nLayers, sizeof(sq4_weight_t));
    g_sq4.wo = (sq4_weight_t*)calloc(nLayers, sizeof(sq4_weight_t));
    g_sq4.wgate = (sq4_weight_t*)calloc(nLayers, sizeof(sq4_weight_t));
    g_sq4.wup = (sq4_weight_t*)calloc(nLayers, sizeof(sq4_weight_t));
    g_sq4.wdown = (sq4_weight_t*)calloc(nLayers, sizeof(sq4_weight_t));
    g_sq4.kCache = (void**)calloc(nLayers, 8);
    g_sq4.vCache = (void**)calloc(nLayers, 8);

    for (int l = 0; l < nLayers; l++) {
        g_sq4.norm1[l] = (__bridge_retained void*)sq4i_buf(dim);
        g_sq4.norm2[l] = (__bridge_retained void*)sq4i_buf(dim);
        g_sq4.bq[l] = (__bridge_retained void*)sq4i_buf(dim);
        g_sq4.bk[l] = (__bridge_retained void*)sq4i_buf(kvDim);
        g_sq4.bv[l] = (__bridge_retained void*)sq4i_buf(kvDim);
        g_sq4.kCache[l] = (__bridge_retained void*)sq4i_buf(maxSeq * kvDim);
        g_sq4.vCache[l] = (__bridge_retained void*)sq4i_buf(maxSeq * kvDim);
    }

    g_sq4.finalNorm = (__bridge_retained void*)sq4i_buf(dim);
    g_sq4.hidden = (__bridge_retained void*)sq4i_buf(dim);
    g_sq4.normed = (__bridge_retained void*)sq4i_buf(dim);
    g_sq4.Q = (__bridge_retained void*)sq4i_buf(dim);
    g_sq4.K = (__bridge_retained void*)sq4i_buf(kvDim);
    g_sq4.V = (__bridge_retained void*)sq4i_buf(kvDim);
    g_sq4.attnOut = (__bridge_retained void*)sq4i_buf(dim);
    g_sq4.normed2 = (__bridge_retained void*)sq4i_buf(dim);
    g_sq4.gatePre = (__bridge_retained void*)sq4i_buf(ffnDim);
    g_sq4.upOut = (__bridge_retained void*)sq4i_buf(ffnDim);
    g_sq4.ffnMid = (__bridge_retained void*)sq4i_buf(ffnDim);
    g_sq4.proj = (__bridge_retained void*)sq4i_buf(dim);
    g_sq4.logits = (__bridge_retained void*)sq4i_buf(vocabSize);

    g_sq4.cb_dim = sq4i_const(dim);
    g_sq4.cb_kvDim = sq4i_const(kvDim);
    g_sq4.cb_headDim = sq4i_const(headDim);
    g_sq4.cb_nHeads = sq4i_const(nHeads);
    g_sq4.cb_nKVHeads = sq4i_const(nKVHeads);
    g_sq4.cb_ffnDim = sq4i_const(ffnDim);
    g_sq4.cb_eps = sq4i_constf(rmsEps);
    g_sq4.cb_theta = sq4i_constf(ropeTheta);
    g_sq4.cb_pos = sq4i_const(0);
    g_sq4.cb_seq = sq4i_const(0);
    g_sq4.cb_Ndim = sq4i_const(dim);
    g_sq4.cb_Nkvdim = sq4i_const(kvDim);
    g_sq4.cb_Nffn = sq4i_const(ffnDim);
    g_sq4.cb_Nvocab = sq4i_const(vocabSize);

    g_sq4.built = true;
    NSLog(@"[SQ4-Infer] built pipeline: dim=%d layers=%d vocab=%d", dim, nLayers, vocabSize);
    return 0;
}

int mtl_sq4_infer_ready(void) { return g_sq4.built ? 1 : 0; }

// Set FP32 data (norms, biases)
void mtl_sq4_infer_set_fp32(int idx, const float* data, int nFloats) {
    if (!g_sq4.built) return;
    int nL = g_sq4.nLayers;
    if (idx < nL * 12) {
        int layer = idx / 12, w = idx % 12;
        switch (w) {
            case 0: memcpy(((__bridge id<MTLBuffer>)g_sq4.norm1[layer]).contents, data, nFloats*4); break;
            case 4: memcpy(((__bridge id<MTLBuffer>)g_sq4.bq[layer]).contents, data, nFloats*4); break;
            case 5: memcpy(((__bridge id<MTLBuffer>)g_sq4.bk[layer]).contents, data, nFloats*4); break;
            case 6: memcpy(((__bridge id<MTLBuffer>)g_sq4.bv[layer]).contents, data, nFloats*4); break;
            case 8: memcpy(((__bridge id<MTLBuffer>)g_sq4.norm2[layer]).contents, data, nFloats*4); break;
        }
    } else if (idx == nL*12) {
        memcpy(((__bridge id<MTLBuffer>)g_sq4.finalNorm).contents, data, nFloats*4);
    } else if (idx == nL*12+1) {
        // lm_head as FP32 (tied weights dequanted from embed SQ4)
        // Need to store as SQ4-compatible: the fused step uses SQ4MV for lm_head.
        // Since the data is FP32, we need to re-encode it as SQ4 on the fly.
        // Simpler: store the FP32 data and use a flag to dispatch FP32 matmul for lm_head.
        // For now: quantize inline to SQ4.
        int rows = g_sq4.vocabSize, cols = g_sq4.dim;
        int n = rows * cols;

        // Per-tensor band calibration
        float* absVals = (float*)malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) absVals[i] = data[i] < 0 ? -data[i] : data[i];
        // Simple sort for band boundaries — use insertion sort on a sample for speed
        // Actually just compute percentile bands without full sort
        float globalMax = 0;
        for (int i = 0; i < n; i++) if (absVals[i] > globalMax) globalMax = absVals[i];

        float bands[8];
        for (int b = 0; b < 8; b++) {
            float lo = globalMax * b / 8.0f;
            float hi = globalMax * (b + 1) / 8.0f;
            float sum = 0; int cnt = 0;
            for (int i = 0; i < n; i++) {
                if (absVals[i] >= lo && absVals[i] < hi) { sum += absVals[i]; cnt++; }
            }
            bands[b] = cnt > 0 ? sum / cnt : (lo + hi) / 2;
        }

        uint8_t* packed = (uint8_t*)calloc(n / 2, 1);
        int halfCols = cols / 2;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                int idx2 = r * cols + c;
                float v = data[idx2];
                float av = absVals[idx2];
                int band = 7;
                for (int b = 0; b < 7; b++) {
                    float boundary = globalMax * (b + 1) / 8.0f;
                    if (av < boundary) { band = b; break; }
                }
                int signBit = v < 0 ? 1 : 0;
                uint8_t nibble = (signBit << 3) | (band & 0x07);
                int shift = (c & 1) * 4;
                packed[r * halfCols + c / 2] |= nibble << shift;
            }
        }

        g_sq4.lmHead.rows = rows;
        g_sq4.lmHead.cols = cols;
        g_sq4.lmHead.outlier_count = 0;
        g_sq4.lmHead.packed = (__bridge_retained void*)[g_device newBufferWithBytes:packed length:n/2 options:MTLResourceStorageModeShared];
        g_sq4.lmHead.bands = (__bridge_retained void*)[g_device newBufferWithBytes:bands length:32 options:MTLResourceStorageModeShared];

        free(absVals);
        free(packed);
    }
}

// Set SQ4 weight
void mtl_sq4_infer_set_sq4(int idx, const uint8_t* packed, int packedBytes,
    const float* bands, const uint32_t* outlierIdx, const float* outlierVal,
    int outlierCount, int rows, int cols) {
    if (!g_sq4.built) return;
    int nL = g_sq4.nLayers;
    sq4_weight_t* target = NULL;
    if (idx < nL * 12) {
        int layer = idx / 12, w = idx % 12;
        switch (w) {
            case 1: target = &g_sq4.wq[layer]; break;
            case 2: target = &g_sq4.wk[layer]; break;
            case 3: target = &g_sq4.wv[layer]; break;
            case 7: target = &g_sq4.wo[layer]; break;
            case 9: target = &g_sq4.wgate[layer]; break;
            case 10: target = &g_sq4.wup[layer]; break;
            case 11: target = &g_sq4.wdown[layer]; break;
        }
    } else if (idx == nL*12+1) {
        target = &g_sq4.lmHead;
    }
    if (!target) return;

    target->rows = rows; target->cols = cols; target->outlier_count = outlierCount;
    target->packed = (__bridge_retained void*)[g_device newBufferWithBytes:packed length:packedBytes options:MTLResourceStorageModeShared];
    target->bands = (__bridge_retained void*)[g_device newBufferWithBytes:bands length:8*sizeof(float) options:MTLResourceStorageModeShared];
    if (outlierCount > 0 && outlierIdx && outlierVal) {
        target->outlier_idx = (__bridge_retained void*)[g_device newBufferWithBytes:outlierIdx length:outlierCount*sizeof(uint32_t) options:MTLResourceStorageModeShared];
        target->outlier_val = (__bridge_retained void*)[g_device newBufferWithBytes:outlierVal length:outlierCount*sizeof(float) options:MTLResourceStorageModeShared];
    }
}

// Dispatch helpers
#define SPS(ps) [enc setComputePipelineState:ps]
#define SBUF(ref, idx) [enc setBuffer:(__bridge id<MTLBuffer>)(ref) offset:0 atIndex:idx]
#define SBUFO(ref, off, idx) [enc setBuffer:(__bridge id<MTLBuffer>)(ref) offset:(off) atIndex:idx]
#define SCB(cb, idx) [enc setBuffer:cb offset:0 atIndex:idx]
#define SDT(x,y,z, tx,ty,tz) [enc dispatchThreads:MTLSizeMake(x,y,z) threadsPerThreadgroup:MTLSizeMake(tx,ty,tz)]
#define SDTG(gx,gy,gz, tx,ty,tz) [enc dispatchThreadgroups:MTLSizeMake(gx,gy,gz) threadsPerThreadgroup:MTLSizeMake(tx,ty,tz)]
#define SBAR() [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]
#ifndef SQ4_MIN
#define SQ4_MIN(a,b) ((a)<(b)?(a):(b))
#endif

// SQ4 matvec dispatch (inline, uses sq4 pipeline states)
#define SQ4MV(act, w, out, cb_cols, cb_rows) do { \
    SPS(g_sq4_ps_matvec); SBUF(act, 0); SBUF((w).packed, 1); SBUF((w).bands, 2); SBUF(out, 3); \
    SCB(cb_cols, 4); SCB(cb_rows, 5); \
    SDTG(((w).rows + 3) / 4, 1, 1, 256, 1, 1); \
    if ((w).outlier_count > 0 && (w).outlier_idx) { \
        SBAR(); \
        SPS(g_sq4_ps_outlier); SBUF((w).outlier_idx, 0); SBUF((w).outlier_val, 1); \
        SBUF((w).packed, 2); SBUF((w).bands, 3); SBUF(act, 4); SBUF(out, 5); \
        SCB(cb_cols, 6); \
        [enc setBuffer:sq4i_const((w).outlier_count) offset:0 atIndex:7]; \
        SDTG(((w).outlier_count + 255) / 256, 1, 1, 256, 1, 1); \
    } \
} while(0)

int mtl_sq4_infer_step(const float* hiddenIn, const float* cosData, const float* sinData,
    int pos, float* logitsOut) {
    if (!g_sq4.built) return -1;
    int dim=g_sq4.dim, kvDim=g_sq4.kvDim, headDim=g_sq4.headDim;
    int nHeads=g_sq4.nHeads, nKVHeads=g_sq4.nKVHeads, ffnDim=g_sq4.ffnDim;
    int nLayers=g_sq4.nLayers, vocabSize=g_sq4.vocabSize;
    int seqLen = pos + 1;

    memcpy(((__bridge id<MTLBuffer>)g_sq4.hidden).contents, hiddenIn, dim * sizeof(float));

    ((uint32_t*)g_sq4.cb_pos.contents)[0] = (uint32_t)pos;
    ((uint32_t*)g_sq4.cb_seq.contents)[0] = (uint32_t)seqLen;

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    NSUInteger tpg_norm = (dim / 32) * 32;
    if (tpg_norm == 0) tpg_norm = 32;
    if (tpg_norm > g_sq4.ps_rmsnorm_out.maxTotalThreadsPerThreadgroup)
        tpg_norm = g_sq4.ps_rmsnorm_out.maxTotalThreadsPerThreadgroup;

    id<MTLBuffer> cb_Ndim = g_sq4.cb_Ndim;
    id<MTLBuffer> cb_Nkvdim = g_sq4.cb_Nkvdim;
    id<MTLBuffer> cb_Nffn = g_sq4.cb_Nffn;
    id<MTLBuffer> cb_Nvocab = g_sq4.cb_Nvocab;

    for (int l = 0; l < nLayers; l++) {
        // RMSNorm: normed = rmsnorm(hidden, norm1)
        SPS(g_sq4.ps_rmsnorm_out); SBUF(g_sq4.hidden, 0); SBUF(g_sq4.normed, 1); SBUF(g_sq4.norm1[l], 2);
        SCB(g_sq4.cb_dim, 3); SCB(g_sq4.cb_eps, 4); SDTG(1,1,1, tpg_norm,1,1); SBAR();

        // QKV SQ4 matvec
        SQ4MV(g_sq4.normed, g_sq4.wq[l], g_sq4.Q, g_sq4.cb_dim, cb_Ndim);
        SQ4MV(g_sq4.normed, g_sq4.wk[l], g_sq4.K, g_sq4.cb_dim, cb_Nkvdim);
        SQ4MV(g_sq4.normed, g_sq4.wv[l], g_sq4.V, g_sq4.cb_dim, cb_Nkvdim);
        SBAR();

        // Bias add
        SPS(g_sq4.ps_bias_add);
        SBUF(g_sq4.Q, 0); SBUF(g_sq4.bq[l], 1); SCB(g_sq4.cb_dim, 2); SDT(dim,1,1, 256,1,1);
        SBUF(g_sq4.K, 0); SBUF(g_sq4.bk[l], 1); SCB(g_sq4.cb_kvDim, 2); SDT(kvDim,1,1, 256,1,1);
        SBUF(g_sq4.V, 0); SBUF(g_sq4.bv[l], 1); SCB(g_sq4.cb_kvDim, 2); SDT(kvDim,1,1, 256,1,1);
        SBAR();

        // RoPE
        int nPQ = nHeads * (headDim / 2), nPK = nKVHeads * (headDim / 2);
        SPS(g_sq4.ps_rope_rh);
        SBUF(g_sq4.Q, 0); SCB(g_sq4.cb_headDim, 1); SCB(g_sq4.cb_nHeads, 2); SCB(g_sq4.cb_pos, 3); SCB(g_sq4.cb_theta, 4);
        SDT(nPQ,1,1, SQ4_MIN(256,(int)g_sq4.ps_rope_rh.maxTotalThreadsPerThreadgroup),1,1);
        SBUF(g_sq4.K, 0); SCB(g_sq4.cb_headDim, 1); SCB(g_sq4.cb_nKVHeads, 2); SCB(g_sq4.cb_pos, 3); SCB(g_sq4.cb_theta, 4);
        SDT(nPK,1,1, SQ4_MIN(256,(int)g_sq4.ps_rope_rh.maxTotalThreadsPerThreadgroup),1,1);
        SBAR();

        // K/V → cache
        int cOff = pos * kvDim * (int)sizeof(float);
        SPS(g_sq4.ps_copy_mem);
        SBUF(g_sq4.K, 0); SBUFO(g_sq4.kCache[l], cOff, 1); SDT(kvDim,1,1, 256,1,1);
        SBUF(g_sq4.V, 0); SBUFO(g_sq4.vCache[l], cOff, 1); SDT(kvDim,1,1, 256,1,1);
        SBAR();

        // Decode attention
        SPS(g_sq4.ps_dec_attn);
        SBUF(g_sq4.Q, 0); SBUF(g_sq4.kCache[l], 1); SBUF(g_sq4.vCache[l], 2); SBUF(g_sq4.attnOut, 3);
        SCB(g_sq4.cb_kvDim, 4); SCB(g_sq4.cb_headDim, 5); SCB(g_sq4.cb_nHeads, 6);
        SCB(g_sq4.cb_nKVHeads, 7); SCB(g_sq4.cb_seq, 8);
        SDTG(nHeads,1,1, headDim,1,1); SBAR();

        // O-proj + residual
        SQ4MV(g_sq4.attnOut, g_sq4.wo[l], g_sq4.proj, g_sq4.cb_dim, cb_Ndim); SBAR();
        SPS(g_sq4.ps_add_inplace); SBUF(g_sq4.hidden, 0); SBUF(g_sq4.proj, 1);
        SDT(dim,1,1, 256,1,1); SBAR();

        // Post-attn norm
        SPS(g_sq4.ps_rmsnorm_out); SBUF(g_sq4.hidden, 0); SBUF(g_sq4.normed2, 1); SBUF(g_sq4.norm2[l], 2);
        SCB(g_sq4.cb_dim, 3); SCB(g_sq4.cb_eps, 4); SDTG(1,1,1, tpg_norm,1,1); SBAR();

        // FFN
        SQ4MV(g_sq4.normed2, g_sq4.wgate[l], g_sq4.gatePre, g_sq4.cb_dim, cb_Nffn);
        SQ4MV(g_sq4.normed2, g_sq4.wup[l], g_sq4.upOut, g_sq4.cb_dim, cb_Nffn);
        SBAR();
        SPS(g_sq4.ps_silu_gate_mul); SBUF(g_sq4.gatePre, 0); SBUF(g_sq4.upOut, 1); SBUF(g_sq4.ffnMid, 2);
        SDT(ffnDim,1,1, 256,1,1); SBAR();
        SQ4MV(g_sq4.ffnMid, g_sq4.wdown[l], g_sq4.proj, g_sq4.cb_ffnDim, cb_Ndim); SBAR();
        SPS(g_sq4.ps_add_inplace); SBUF(g_sq4.hidden, 0); SBUF(g_sq4.proj, 1);
        SDT(dim,1,1, 256,1,1); SBAR();
    }

    // Final RMSNorm + lm_head
    SPS(g_sq4.ps_rmsnorm_save); SBUF(g_sq4.hidden, 0); SBUF(g_sq4.finalNorm, 1); SBUF(g_sq4.normed, 2);
    SCB(g_sq4.cb_dim, 3); SCB(g_sq4.cb_eps, 4); SDTG(1,1,1, tpg_norm,1,1); SBAR();
    SQ4MV(g_sq4.hidden, g_sq4.lmHead, g_sq4.logits, g_sq4.cb_dim, cb_Nvocab);

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    memcpy(logitsOut, ((__bridge id<MTLBuffer>)g_sq4.logits).contents, vocabSize * sizeof(float));
    return 0;
}

void mtl_sq4_infer_reset_kv(void) {
    if (!g_sq4.built) return;
    int kvDim = g_sq4.kvDim, maxSeq = g_sq4.maxSeq;
    for (int l = 0; l < g_sq4.nLayers; l++) {
        memset(((__bridge id<MTLBuffer>)g_sq4.kCache[l]).contents, 0, maxSeq * kvDim * sizeof(float));
        memset(((__bridge id<MTLBuffer>)g_sq4.vCache[l]).contents, 0, maxSeq * kvDim * sizeof(float));
    }
}
