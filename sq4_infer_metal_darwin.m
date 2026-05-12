// sq4_infer_metal_darwin.m — SQ4 fused inference engine (Metal).
// Clean rewrite: one command buffer per token, slab-allocated weights.
// Kernel: sq4_matvec from infer.metallib (uint4 vectorized, 16-entry tg LUT).

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

extern id<MTLDevice> g_device;
extern id<MTLCommandQueue> g_queue;
extern id<MTLComputePipelineState> mtl_make_pipeline(NSString* name);

typedef struct {
    long long packed_offset;
    int bands_offset;   // float index into bands slab
    int outlier_offset; // element index into outlier slabs
    int outlier_count;
    int rows, cols;
    int lut_offset;     // element offset into LUT texture (16 entries per tensor)
    long long lin_packed_offset; // byte offset in MLX-padded packed slab
    int lin_padded_cols;   // cols rounded up to MLX block_size (512)
} sq4_wt;

static struct {
    int dim, kvDim, headDim, nHeads, nKVHeads, ffnDim, vocabSize, nLayers, maxSeq;
    float ropeTheta, rmsEps;
    bool built;

    // Weight slabs
    void* packed_slab;
    void* bands_slab;
    void* outlier_idx_slab;
    void* outlier_val_slab;

    // Per-layer SQ4 descriptors
    sq4_wt *wq, *wk, *wv, *wo, *wgate, *wup, *wdown;
    sq4_wt lmHead;

    // Per-layer FP32 (norms, biases, KV cache) — void* to avoid ARC issues with C arrays
    void **norm1, **norm2, **bq, **bk, **bv, **kCache, **vCache;
    void* finalNorm;
    void* embedBuf;

    // Scratch
    void* hidden; void* normed; void* normed2; void* Q; void* K; void* V; void* attnOut; void* proj;
    void* gatePre; void* upOut; void* ffnMid; void* logits;

    // Constant buffers
    void* cb_dim; void* cb_kvDim; void* cb_headDim; void* cb_nHeads; void* cb_nKVHeads;
    void* cb_ffnDim; void* cb_eps; void* cb_theta;
    void* cb_pos; void* cb_seq;
    void* cb_Ndim; void* cb_Nkvdim; void* cb_Nffn; void* cb_Nvocab;

    // Per-weight outlier count constant buffers
    void** cb_oc;  // [7 * nLayers + 1] (7 per layer + lmHead)

    // Pipeline states
    void* ps_sq4mv; void* ps_rmsnorm_out; void* ps_rmsnorm_save;
    void* ps_rope; void* ps_attn; void* ps_silu_gate_mul;
    void* ps_add_inplace; void* ps_copy; void* ps_bias_add; void* ps_argmax;
    void* ps_fused_brk;
    void* ps_sq4mv_amx;
    void* ps_dequant; void* ps_fp16mv; void* ps_sq4mv_fp16; void* ps_outlier_fp16;
    void* ps_sq4mv_bl; void* ps_sq4mv_shuf; void* ps_fused_gus;
    void* ps_sq4mv_tex;
    void* lut_texture;
    void* ps_sq4mv_lin; void* ps_fused_gus_lin; void* ps_sq4mv_fast;
    void* ps_mlx_qmv; // MLX's affine_qmv_fast kernel
    void* fast_packed;
    void* fast_row_scales;
    void* fast_row_biases;
    void* lin_packed_slab;  // re-encoded linear INT4 nibbles
    void* lin_scales_slab;  // per-group scales
    void* lin_biases_slab;  // per-group biases
    void* mlx_packed_slab;  // MLX-padded packed nibbles (rows padded to 512)
    void* mlx_scales_slab;  // MLX-padded per-group scales
    void* mlx_biases_slab;  // MLX-padded per-group biases
    // Copies of shared slab data for lazy linear encoding
    uint8_t* shared_packed_copy;
    float* shared_bands_copy;
    uint32_t* shared_outlier_idx_copy;
    float* shared_outlier_val_copy;
    long long shared_packed_bytes;
    int shared_bands_floats;
    int shared_outlier_count;

    // Staging buffer for dequanted FP16 weights (largest matrix in one layer)
    void* fp16_stage;
    int fp16_stage_size; // in halfs
    void* cb_dequant_count; // for dequant kernel (nWeights)
    void* cb_outlier_count; // for outlier correction kernel (outlier_count)

    void* argmax_result;
} S = {0};

#define LIN_GROUP_SIZE 32

// Bridge cast helpers: void* ↔ ObjC types
#define B(ptr) ((__bridge id<MTLBuffer>)(ptr))
#define PS(ptr) ((__bridge id<MTLComputePipelineState>)(ptr))
#define TX(ptr) ((__bridge id<MTLTexture>)(ptr))
#define BR(obj) ((__bridge_retained void*)(obj))

static void* mkbuf(int nBytes) {
    return BR([g_device newBufferWithLength:nBytes options:MTLResourceStorageModePrivate]);
}

static void* mkconst(uint32_t v) {
    return BR([g_device newBufferWithBytes:&v length:4 options:MTLResourceStorageModeShared]);
}

static void* mkconstf(float v) {
    return BR([g_device newBufferWithBytes:&v length:4 options:MTLResourceStorageModeShared]);
}

static void* upload_private(const void* data, long long nBytes) {
    id<MTLBuffer> staging = [g_device newBufferWithBytes:data length:(NSUInteger)nBytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> priv = [g_device newBufferWithLength:(NSUInteger)nBytes options:MTLResourceStorageModePrivate];
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit copyFromBuffer:staging sourceOffset:0 toBuffer:priv destinationOffset:0 size:nBytes];
    [blit endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return BR(priv);
}

// ============================================================================
// API: Build
// ============================================================================

void mtl_sq4_infer_alloc_slabs(long long totalPackedBytes, int totalBandsFloats, int totalOutliers) {
    S.packed_slab = BR([g_device newBufferWithLength:(NSUInteger)totalPackedBytes options:MTLResourceStorageModeShared]);
    S.bands_slab = BR([g_device newBufferWithLength:totalBandsFloats * 4 options:MTLResourceStorageModeShared]);
    if (totalOutliers > 0) {
        S.outlier_idx_slab = BR([g_device newBufferWithLength:totalOutliers * 4 options:MTLResourceStorageModeShared]);
        S.outlier_val_slab = BR([g_device newBufferWithLength:totalOutliers * 4 options:MTLResourceStorageModeShared]);
    }
}

void mtl_sq4_infer_upload_packed(long long packedOffset, const uint8_t* mag, int magBytes,
    const uint8_t* sign_data, int signBytes, int nWeights) {
    uint8_t* dst = (uint8_t*)B(S.packed_slab).contents + packedOffset;
    for (int i = 0; i < nWeights; i++) {
        int bit_pos = i * 3;
        int byte_idx = bit_pos / 8;
        int bit_off = bit_pos % 8;
        uint8_t raw = mag[byte_idx] >> bit_off;
        if (bit_off > 5 && byte_idx + 1 < magBytes) raw |= mag[byte_idx + 1] << (8 - bit_off);
        int band = raw & 0x07;
        int sign_bit = (sign_data[i / 8] >> (i % 8)) & 1;
        uint8_t nibble = (sign_bit << 3) | band;
        int dst_byte = i / 2;
        int shift = (i & 1) * 4;
        if (shift == 0)
            dst[dst_byte] = nibble;
        else
            dst[dst_byte] |= nibble << 4;
    }
}

void mtl_sq4_infer_upload_bands(int floatOffset, const float* data, int nFloats) {
    memcpy((float*)B(S.bands_slab).contents + floatOffset, data, nFloats * 4);
}

void mtl_sq4_infer_upload_outliers(int offset, const uint32_t* idx, const float* val, int count) {
    if (count <= 0 || !S.outlier_idx_slab) return;
    memcpy((uint32_t*)B(S.outlier_idx_slab).contents + offset, idx, count * 4);
    memcpy((float*)B(S.outlier_val_slab).contents + offset, val, count * 4);
}

void mtl_sq4_infer_upload_embed(const float* data, int nFloats) {
    S.embedBuf = upload_private(data, nFloats * 4);
}

void mtl_sq4_infer_finalize_slabs(void) {
    id<MTLBuffer> sp = B(S.packed_slab);
    id<MTLBuffer> sb = B(S.bands_slab);

    // Save copies of shared slab data BEFORE promoting to private
    S.shared_packed_bytes = (long long)sp.length;
    S.shared_bands_floats = (int)(sb.length / sizeof(float));
    S.shared_packed_copy = (uint8_t*)malloc(sp.length);
    S.shared_bands_copy = (float*)malloc(sb.length);
    memcpy(S.shared_packed_copy, sp.contents, sp.length);
    memcpy(S.shared_bands_copy, sb.contents, sb.length);
    if (S.outlier_idx_slab) {
        id<MTLBuffer> si = B(S.outlier_idx_slab);
        id<MTLBuffer> sv = B(S.outlier_val_slab);
        S.shared_outlier_count = (int)(si.length / 4);
        S.shared_outlier_idx_copy = (uint32_t*)malloc(si.length);
        S.shared_outlier_val_copy = (float*)malloc(sv.length);
        memcpy(S.shared_outlier_idx_copy, si.contents, si.length);
        memcpy(S.shared_outlier_val_copy, sv.contents, sv.length);
    }

    // Promote to private
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];

    id<MTLBuffer> pp = [g_device newBufferWithLength:sp.length options:MTLResourceStorageModePrivate];
    [blit copyFromBuffer:sp sourceOffset:0 toBuffer:pp destinationOffset:0 size:sp.length];

    id<MTLBuffer> pb = [g_device newBufferWithLength:sb.length options:MTLResourceStorageModePrivate];
    [blit copyFromBuffer:sb sourceOffset:0 toBuffer:pb destinationOffset:0 size:sb.length];

    if (S.outlier_idx_slab) {
        id<MTLBuffer> si = B(S.outlier_idx_slab);
        id<MTLBuffer> pi = [g_device newBufferWithLength:si.length options:MTLResourceStorageModePrivate];
        [blit copyFromBuffer:si sourceOffset:0 toBuffer:pi destinationOffset:0 size:si.length];
        id<MTLBuffer> sv = B(S.outlier_val_slab);
        id<MTLBuffer> pv = [g_device newBufferWithLength:sv.length options:MTLResourceStorageModePrivate];
        [blit copyFromBuffer:sv sourceOffset:0 toBuffer:pv destinationOffset:0 size:sv.length];
        S.outlier_idx_slab = BR(pi);
        S.outlier_val_slab = BR(pv);
    }

    [blit endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    S.packed_slab = BR(pp);
    S.bands_slab = BR(pb);

    // Build linear INT4 re-encoding from SQ4 band means
    // For each tensor's packed nibbles: dequant → find per-group scale+bias → re-encode as linear int4
    {
        long long totalPacked = (long long)sp.length;
        int totalWeights = totalPacked * 2; // 2 nibbles per byte
        int totalGroups = totalWeights / LIN_GROUP_SIZE;

        uint8_t* srcPacked = (uint8_t*)sp.contents;
        float* srcBands = (float*)sb.contents;

        uint8_t* linPacked = (uint8_t*)calloc(1, totalPacked);
        float* linScales = (float*)malloc(totalGroups * sizeof(float));
        float* linBiases = (float*)malloc(totalGroups * sizeof(float));

        // Walk through all weights in slab order
        // The packed slab has all tensors concatenated. We need per-tensor band tables.
        // Use bands_slab: every 8 floats is one tensor's bands.
        int nTensors = (int)(sb.length / (8 * sizeof(float)));

        // Build global table16 per tensor for dequant
        // We don't know which byte belongs to which tensor without the descriptors.
        // But at finalize time, descriptors aren't set yet!
        // Solution: build the table from the contiguous bands slab — tensor T's bands
        // are at offset T*8 in the bands array.

        // Simple approach: iterate all bytes, figure out which tensor they belong to
        // by their position. But we don't have tensor boundaries here.
        // Alternative: defer the linear re-encoding until AFTER descriptors are set.
        // For now: re-encode the ENTIRE slab assuming all nibbles use bands[0..7] of
        // their respective tensor. We need to know tensor boundaries.

        // Actually, the Go side knows tensor boundaries and passes them via set_sq4_desc.
        // Let's do the re-encoding LAZILY — when the first inference call happens,
        // we have all descriptors and can walk the tensors properly.

        free(linPacked);
        free(linScales);
        free(linBiases);
        // Mark as not-yet-built
        S.lin_packed_slab = nil;
        S.lin_scales_slab = nil;
        S.lin_biases_slab = nil;
    }

    // Build LUT texture slab from bands: 16 floats per tensor (8 positive + 8 negative)
    // Read from the SHARED bands slab (before we replace it with private)
    // Actually sb is the old shared slab — read from it
    int nTensors = (int)(sb.length / (8 * sizeof(float)));
    int lutTotal = nTensors * 16;
    float* lutData = (float*)malloc(lutTotal * sizeof(float));
    float* bandsData = (float*)sb.contents;
    for (int t = 0; t < nTensors; t++) {
        for (int i = 0; i < 8; i++) {
            lutData[t * 16 + i] = bandsData[t * 8 + i];
            lutData[t * 16 + i + 8] = -bandsData[t * 8 + i];
        }
    }
    // Create 1D texture: r32float, width = lutTotal
    MTLTextureDescriptor* texDesc = [MTLTextureDescriptor new];
    texDesc.textureType = MTLTextureType1D;
    texDesc.pixelFormat = MTLPixelFormatR32Float;
    texDesc.width = lutTotal;
    texDesc.usage = MTLTextureUsageShaderRead;
    texDesc.storageMode = MTLStorageModeShared;
    id<MTLTexture> lutTex = [g_device newTextureWithDescriptor:texDesc];
    [lutTex replaceRegion:MTLRegionMake1D(0, lutTotal)
              mipmapLevel:0
                withBytes:lutData
              bytesPerRow:lutTotal * sizeof(float)];
    free(lutData);
    S.lut_texture = BR(lutTex);
    NSLog(@"[SQ4] LUT texture: %d tensors × 16 entries = %d texels", nTensors, lutTotal);

    NSLog(@"[SQ4] slabs promoted to private memory");
}

int mtl_sq4_infer_build(int dim, int kvDim, int headDim,
    int nHeads, int nKVHeads, int ffnDim,
    int vocabSize, int nLayers, int maxSeq,
    float ropeTheta, float rmsEps) {
    if (S.built) return 0;
    if (!g_device) return -1;

    S.dim=dim; S.kvDim=kvDim; S.headDim=headDim;
    S.nHeads=nHeads; S.nKVHeads=nKVHeads; S.ffnDim=ffnDim;
    S.vocabSize=vocabSize; S.nLayers=nLayers; S.maxSeq=maxSeq;
    S.ropeTheta=ropeTheta; S.rmsEps=rmsEps;

    // Load pipeline states from infer.metallib
    NSArray* paths = @[
        [[[NSBundle mainBundle] executablePath] stringByDeletingLastPathComponent],
        [[[[NSBundle mainBundle] executablePath] stringByDeletingLastPathComponent] stringByAppendingPathComponent:@"kernels"],
        @"kernels", @".",
        [[NSString stringWithUTF8String:getenv("HOME") ?: ""] stringByAppendingPathComponent:@"go/src/github.com/tensorwire/mongoose/kernels"]
    ];
    id<MTLLibrary> lib = nil;
    for (NSString* dir in paths) {
        NSString* p = [dir stringByAppendingPathComponent:@"infer.metallib"];
        if ([[NSFileManager defaultManager] fileExistsAtPath:p]) {
            NSError* e = nil;
            lib = [g_device newLibraryWithURL:[NSURL fileURLWithPath:p] error:&e];
            if (lib) { NSLog(@"[SQ4] loaded %@", p); break; }
        }
    }
    if (!lib) { NSLog(@"[SQ4] infer.metallib not found"); return -1; }

    id<MTLComputePipelineState> (^ps)(NSString*) = ^(NSString* name) {
        id<MTLFunction> fn = [lib newFunctionWithName:name];
        if (!fn) { NSLog(@"[SQ4] kernel %@ not found", name); return (id<MTLComputePipelineState>)nil; }
        NSError* e = nil;
        return [g_device newComputePipelineStateWithFunction:fn error:&e];
    };

    S.ps_sq4mv = BR(ps(@"sq4_matvec"));
    S.ps_rope = BR(ps(@"rope_rotate_half"));
    S.ps_attn = BR(ps(@"decode_attn"));
    S.ps_rmsnorm_out = BR(mtl_make_pipeline(@"rmsnorm_out"));
    S.ps_rmsnorm_save = BR(mtl_make_pipeline(@"rmsnorm_save"));
    S.ps_silu_gate_mul = BR(mtl_make_pipeline(@"silu_gate_mul"));
    S.ps_add_inplace = BR(mtl_make_pipeline(@"add_inplace"));
    S.ps_copy = BR(mtl_make_pipeline(@"copy_mem"));
    S.ps_bias_add = BR(mtl_make_pipeline(@"bias_add"));
    S.ps_argmax = BR(ps(@"argmax_sample"));
    S.ps_fused_brk = nil;
    S.ps_sq4mv_amx = nil;
    S.ps_fused_gus = BR(ps(@"sq4_fused_gate_up_silu"));
    S.ps_sq4mv_tex = nil;
    S.ps_sq4mv_lin = BR(ps(@"sq4_matvec_linear"));
    S.ps_fused_gus_lin = nil;
    S.ps_sq4mv_fast = BR(ps(@"sq4_matvec_fast"));

    // MLX-derived quantized matvec kernel (compiled from source, in our metallib)
    S.ps_mlx_qmv = BR(ps(@"sq4_mlx_qmv"));
    if (S.ps_mlx_qmv) NSLog(@"[SQ4] MLX-derived qmv kernel loaded");
    S.ps_dequant = BR(ps(@"sq4_dequant_to_half"));
    S.ps_fp16mv = BR(ps(@"fp16_matvec"));
    S.ps_sq4mv_fp16 = BR(ps(@"sq4_matvec_fp16"));
    S.ps_outlier_fp16 = BR(ps(@"sq4_outlier_apply_fp16"));
    S.ps_sq4mv_bl = BR(ps(@"sq4_matvec_branchless"));
    S.ps_sq4mv_shuf = BR(ps(@"sq4_matvec_shuffle"));

    // Staging buffer: largest weight matrix per layer at FP16
    int maxWeights = dim * ffnDim; // gate/up are ffnDim × dim
    if (dim * dim > maxWeights) maxWeights = dim * dim; // wq/wo are dim × dim
    S.fp16_stage_size = maxWeights;
    S.fp16_stage = mkbuf(maxWeights * 2);

    // Pre-allocated constant buffers for per-weight dequant counts
    // (7 weight types per layer + lmHead — but we reuse one buffer and update it)
    S.cb_dequant_count = mkconst(0);
    S.cb_outlier_count = mkconst(0);

    if (!PS(S.ps_sq4mv) || !PS(S.ps_rmsnorm_out) || !PS(S.ps_rope) || !PS(S.ps_attn)) {
        NSLog(@"[SQ4] missing required pipeline states"); return -1;
    }

    // Per-layer allocations
    S.wq = (sq4_wt*)calloc(nLayers, sizeof(sq4_wt));
    S.wk = (sq4_wt*)calloc(nLayers, sizeof(sq4_wt));
    S.wv = (sq4_wt*)calloc(nLayers, sizeof(sq4_wt));
    S.wo = (sq4_wt*)calloc(nLayers, sizeof(sq4_wt));
    S.wgate = (sq4_wt*)calloc(nLayers, sizeof(sq4_wt));
    S.wup = (sq4_wt*)calloc(nLayers, sizeof(sq4_wt));
    S.wdown = (sq4_wt*)calloc(nLayers, sizeof(sq4_wt));
    S.norm1 = (void**)calloc(nLayers, 8);
    S.norm2 = (void**)calloc(nLayers, 8);
    S.bq = (void**)calloc(nLayers, 8);
    S.bk = (void**)calloc(nLayers, 8);
    S.bv = (void**)calloc(nLayers, 8);
    S.kCache = (void**)calloc(nLayers, 8);
    S.vCache = (void**)calloc(nLayers, 8);
    S.cb_oc = (void**)calloc(7 * nLayers + 1, 8);

    for (int l = 0; l < nLayers; l++) {
        S.norm1[l] = mkbuf(dim * 4);
        S.norm2[l] = mkbuf(dim * 4);
        S.bq[l] = mkbuf(dim * 4);
        S.bk[l] = mkbuf(kvDim * 4);
        S.bv[l] = mkbuf(kvDim * 4);
        S.kCache[l] = mkbuf(maxSeq * kvDim * 4);
        S.vCache[l] = mkbuf(maxSeq * kvDim * 4);
    }
    S.finalNorm = mkbuf(dim * 4);

    // Scratch — pad to next 512 for MLX kernel overread safety
    int padDim = ((dim + 511) / 512) * 512;
    int padFFN = ((ffnDim + 511) / 512) * 512;
    S.hidden = mkbuf(padDim * 4);
    S.normed = mkbuf(padDim * 4);
    S.normed2 = mkbuf(padDim * 4);
    S.Q = mkbuf(padDim * 4);
    S.K = mkbuf(kvDim * 4);
    S.V = mkbuf(kvDim * 4);
    S.attnOut = mkbuf(padDim * 4);
    S.proj = mkbuf(padDim * 4);
    S.gatePre = mkbuf(padFFN * 4);
    S.upOut = mkbuf(padFFN * 4);
    S.ffnMid = mkbuf(padFFN * 4);
    S.logits = BR([g_device newBufferWithLength:vocabSize * 4 options:MTLResourceStorageModeShared]);

    // Zero-fill scratch buffers — MLX kernel reads padded regions; NaN/Inf in
    // uninitialized GPU memory would propagate through 0*NaN=NaN
    {
        id<MTLCommandBuffer> zcmd = [g_queue commandBuffer];
        id<MTLBlitCommandEncoder> zblit = [zcmd blitCommandEncoder];
        void* zbufs[] = {S.hidden, S.normed, S.normed2, S.Q, S.attnOut, S.proj,
                         S.gatePre, S.upOut, S.ffnMid};
        int zsizes[] = {padDim*4, padDim*4, padDim*4, padDim*4, padDim*4, padDim*4,
                        padFFN*4, padFFN*4, padFFN*4};
        for (int i = 0; i < 9; i++)
            [zblit fillBuffer:B(zbufs[i]) range:NSMakeRange(0, zsizes[i]) value:0];
        [zblit endEncoding];
        [zcmd commit];
        [zcmd waitUntilCompleted];
    }

    // Constants
    S.cb_dim = mkconst(dim); S.cb_kvDim = mkconst(kvDim);
    S.cb_headDim = mkconst(headDim);
    S.cb_nHeads = mkconst(nHeads); S.cb_nKVHeads = mkconst(nKVHeads);
    S.cb_ffnDim = mkconst(ffnDim);
    S.cb_eps = mkconstf(rmsEps); S.cb_theta = mkconstf(ropeTheta);
    S.cb_pos = mkconst(0); S.cb_seq = mkconst(0);
    S.cb_Ndim = mkconst(dim); S.cb_Nkvdim = mkconst(kvDim);
    S.cb_Nffn = mkconst(ffnDim); S.cb_Nvocab = mkconst(vocabSize);

    uint32_t zero = 0;
    S.argmax_result = BR([g_device newBufferWithBytes:&zero length:4 options:MTLResourceStorageModeShared]);

    S.built = true;
    NSLog(@"[SQ4] built: dim=%d layers=%d vocab=%d heads=%d kv=%d", dim, nLayers, vocabSize, nHeads, nKVHeads);

    // Self-test: verify dequant→fp16_matvec matches scalar sq4_matvec
    if (S.ps_dequant && S.ps_fp16mv) {
        int tR = 16, tC = 32;
        int tN = tR * tC;

        // Create test data: packed nibbles + bands
        uint8_t testPacked[tN/2];
        float testBands[8] = {0.001f, 0.005f, 0.01f, 0.02f, 0.04f, 0.08f, 0.15f, 0.3f};
        float testAct[32];
        for (int i = 0; i < tN; i++) {
            uint8_t nib = (i * 7 + 3) & 0x0F; // deterministic pattern
            int sh = (i & 1) * 4;
            if (sh == 0) testPacked[i/2] = nib;
            else testPacked[i/2] |= nib << 4;
        }
        for (int i = 0; i < 32; i++) testAct[i] = (i % 7) * 0.1f - 0.3f;

        id<MTLBuffer> bPacked = [g_device newBufferWithBytes:testPacked length:tN/2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bBands = [g_device newBufferWithBytes:testBands length:32 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bAct = [g_device newBufferWithBytes:testAct length:32*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bOut1 = [g_device newBufferWithLength:tR*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bOut2 = [g_device newBufferWithLength:tR*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bStage = [g_device newBufferWithLength:tN*2 options:MTLResourceStorageModeShared];
        uint32_t tK = tC, tNN = tR, tCount = tN, tOC = 0;
        id<MTLBuffer> cbK = [g_device newBufferWithBytes:&tK length:4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> cbN = [g_device newBufferWithBytes:&tNN length:4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> cbCnt = [g_device newBufferWithBytes:&tCount length:4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> cbOC = [g_device newBufferWithBytes:&tOC length:4 options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> tcmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> tenc = [tcmd computeCommandEncoder];

        // Scalar SQ4 matvec
        [tenc setComputePipelineState:PS(S.ps_sq4mv)];
        [tenc setBuffer:bAct offset:0 atIndex:0];
        [tenc setBuffer:bPacked offset:0 atIndex:1];
        [tenc setBuffer:bBands offset:0 atIndex:2];
        [tenc setBuffer:bOut1 offset:0 atIndex:3];
        [tenc setBuffer:cbK offset:0 atIndex:4];
        [tenc setBuffer:cbN offset:0 atIndex:5];
        [tenc setBuffer:bPacked offset:0 atIndex:6]; // dummy outlier
        [tenc setBuffer:bPacked offset:0 atIndex:7];
        [tenc setBuffer:cbOC offset:0 atIndex:8];
        [tenc dispatchThreadgroups:MTLSizeMake((tR+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [tenc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // Dequant → FP16 staging
        [tenc setComputePipelineState:PS(S.ps_dequant)];
        [tenc setBuffer:bPacked offset:0 atIndex:0];
        [tenc setBuffer:bBands offset:0 atIndex:1];
        [tenc setBuffer:bStage offset:0 atIndex:2];
        [tenc setBuffer:cbCnt offset:0 atIndex:3];
        [tenc dispatchThreads:MTLSizeMake(tN,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [tenc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // FP16 matvec
        [tenc setComputePipelineState:PS(S.ps_fp16mv)];
        [tenc setBuffer:bAct offset:0 atIndex:0];
        [tenc setBuffer:bStage offset:0 atIndex:1];
        [tenc setBuffer:bOut2 offset:0 atIndex:2];
        [tenc setBuffer:cbK offset:0 atIndex:3];
        [tenc setBuffer:cbN offset:0 atIndex:4];
        [tenc dispatchThreadgroups:MTLSizeMake((tR+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];

        [tenc endEncoding];
        [tcmd commit];
        [tcmd waitUntilCompleted];

        float* r1 = (float*)bOut1.contents;
        float* r2 = (float*)bOut2.contents;
        uint16_t* staged = (uint16_t*)bStage.contents;

        // Decode FP16 → float for printing (IEEE 754 half)
        float sf[8];
        for (int i = 0; i < 8; i++) {
            uint16_t h = staged[i];
            uint32_t sign = (h >> 15) & 1;
            uint32_t exp = (h >> 10) & 0x1F;
            uint32_t mant = h & 0x3FF;
            uint32_t f;
            if (exp == 0) f = (sign << 31) | (mant << 13); // denorm
            else if (exp == 31) f = (sign << 31) | 0x7F800000 | (mant << 13); // inf/nan
            else f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
            memcpy(&sf[i], &f, 4);
        }
        fprintf(stderr, "[SELF-TEST] dequant first 8: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
            sf[0], sf[1], sf[2], sf[3], sf[4], sf[5], sf[6], sf[7]);

        // CPU reference dequant for comparison
        float table16[16];
        for (int i = 0; i < 8; i++) { table16[i] = testBands[i]; table16[i+8] = -testBands[i]; }
        fprintf(stderr, "[SELF-TEST] CPU ref   first 8: ");
        for (int i = 0; i < 8; i++) {
            int nib = (testPacked[i/2] >> ((i&1)*4)) & 0x0F;
            fprintf(stderr, "%.4f ", table16[nib]);
        }
        fprintf(stderr, "\n");

        float maxDiff = 0;
        for (int r = 0; r < tR; r++) {
            float d = r1[r] - r2[r];
            if (d < 0) d = -d;
            if (d > maxDiff) maxDiff = d;
        }
        fprintf(stderr, "[SELF-TEST] scalar[0..3]=%.4f %.4f %.4f %.4f\n", r1[0], r1[1], r1[2], r1[3]);
        fprintf(stderr, "[SELF-TEST] fp16  [0..3]=%.4f %.4f %.4f %.4f\n", r2[0], r2[1], r2[2], r2[3]);
        fprintf(stderr, "[SELF-TEST] max diff=%.6f (%s)\n", maxDiff, maxDiff < 0.01 ? "PASS" : "FAIL");
    }

    return 0;
}

int mtl_sq4_infer_ready(void) { return S.built ? 1 : 0; }

// ============================================================================
// API: Set weights
// ============================================================================

void mtl_sq4_infer_set_fp32(int idx, const float* data, int nFloats) {
    if (!S.built) return;
    int nL = S.nLayers;
    // Slot layout per layer: 0=norm1, 1=wq, 2=wk, 3=wv, 4=bq, 5=bk, 6=bv, 7=wo, 8=norm2, 9=gate, 10=up, 11=down
    if (idx < nL * 12) {
        int layer = idx / 12, slot = idx % 12;
        void* target = nil;
        switch (slot) {
            case 0: target = S.norm1[layer]; break;
            case 4: target = S.bq[layer]; break;
            case 5: target = S.bk[layer]; break;
            case 6: target = S.bv[layer]; break;
            case 8: target = S.norm2[layer]; break;
        }
        if (target) {
            id<MTLBuffer> staging = [g_device newBufferWithBytes:data length:nFloats*4 options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
            [blit copyFromBuffer:staging sourceOffset:0 toBuffer:B(target) destinationOffset:0 size:nFloats*4];
            [blit endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    } else if (idx == nL * 12) {
        id<MTLBuffer> staging = [g_device newBufferWithBytes:data length:nFloats*4 options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        [blit copyFromBuffer:staging sourceOffset:0 toBuffer:B(S.finalNorm) destinationOffset:0 size:nFloats*4];
        [blit endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void mtl_sq4_infer_set_sq4_desc(int idx, long long packedOffset,
    int bandsOffset, int outlierOffset, int outlierCount, int rows, int cols) {
    if (!S.built) return;
    int nL = S.nLayers;
    sq4_wt* target = NULL;
    int ocIdx = -1;
    if (idx < nL * 12) {
        int layer = idx / 12, slot = idx % 12;
        switch (slot) {
            case 1: target = &S.wq[layer]; ocIdx = layer*7+0; break;
            case 2: target = &S.wk[layer]; ocIdx = layer*7+1; break;
            case 3: target = &S.wv[layer]; ocIdx = layer*7+2; break;
            case 7: target = &S.wo[layer]; ocIdx = layer*7+3; break;
            case 9: target = &S.wgate[layer]; ocIdx = layer*7+4; break;
            case 10: target = &S.wup[layer]; ocIdx = layer*7+5; break;
            case 11: target = &S.wdown[layer]; ocIdx = layer*7+6; break;
        }
    } else if (idx == nL * 12 + 1) {
        target = &S.lmHead;
        ocIdx = 7 * nL;
    }
    if (!target) return;
    target->packed_offset = packedOffset;
    target->bands_offset = bandsOffset;
    target->outlier_offset = outlierOffset;
    target->outlier_count = outlierCount;
    target->rows = rows;
    target->cols = cols;
    target->lut_offset = (bandsOffset / 8) * 16; // 8 bands per tensor → 16 LUT entries
    if (ocIdx >= 0) {
        uint32_t oc = (uint32_t)outlierCount;
        S.cb_oc[ocIdx] = BR([g_device newBufferWithBytes:&oc length:4 options:MTLResourceStorageModeShared]);
    }
}

// ============================================================================
// Dispatch helpers
// ============================================================================

#define ENC(ps)         [enc setComputePipelineState:PS(ps)]
#define BUF(b, i)       [enc setBuffer:B(b) offset:0 atIndex:i]
#define BUFO(b, off, i) [enc setBuffer:B(b) offset:(off) atIndex:i]
#define CB(b, i)        [enc setBuffer:B(b) offset:0 atIndex:i]
#define DT(x,tx)        [enc dispatchThreads:MTLSizeMake(x,1,1) threadsPerThreadgroup:MTLSizeMake(tx,1,1)]
#define DTG(gx,tx)      [enc dispatchThreadgroups:MTLSizeMake(gx,1,1) threadsPerThreadgroup:MTLSizeMake(tx,1,1)]
#define BAR()           [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]

// Dequant SQ4 → FP16 staging, then FP16 matvec. Two dispatches, no LUT in hot path.
// Outlier correction applied to FP16 staging buffer before matvec.
static void sq4mv(id<MTLComputeCommandEncoder> enc, void* act,
    sq4_wt* w, void* out, void* cbK, void* cbN, int ocIdx) {
    int nWeights = w->rows * w->cols;

    if (0 && S.ps_dequant && S.ps_fp16mv && nWeights <= S.fp16_stage_size) {
        // Phase 1: bulk dequant SQ4 → FP16 staging
        uint32_t cnt = (uint32_t)nWeights;
        ENC(S.ps_dequant);
        BUFO(S.packed_slab, w->packed_offset, 0);
        BUFO(S.bands_slab, w->bands_offset * 4, 1);
        BUF(S.fp16_stage, 2);
        [enc setBytes:&cnt length:4 atIndex:3];
        DT(nWeights, 256);

        // Apply outlier corrections to FP16 staging
        if (w->outlier_count > 0 && S.outlier_idx_slab && S.ps_outlier_fp16) {
            BAR();
            uint32_t oc = (uint32_t)w->outlier_count;
            ENC(S.ps_outlier_fp16);
            BUFO(S.outlier_idx_slab, w->outlier_offset * 4, 0);
            BUFO(S.outlier_val_slab, w->outlier_offset * 4, 1);
            BUF(S.fp16_stage, 2);
            [enc setBytes:&oc length:4 atIndex:3];
            DT(w->outlier_count, 256);
        }
        BAR();

        // Phase 2: FP16 matvec — clean half-precision data, no LUT
        ENC(S.ps_fp16mv);
        BUF(act, 0);
        BUF(S.fp16_stage, 1);
        BUF(out, 2);
        CB(cbK, 3); CB(cbN, 4);
        DTG((w->rows + 3) / 4, 256);
        return;
    }

    // AMX path: 16 rows per threadgroup, 64 threads (2 simdgroups)
    // Only for K <= 896 (2 simdgroups × 8 rows × K × 2 must fit 32KB tg mem)
    if (S.ps_sq4mv_amx && (w->rows % 8 == 0) && (w->cols % 8 == 0) && w->cols <= 896) {
        ENC(S.ps_sq4mv_amx);
        BUF(act, 0);
        BUFO(S.packed_slab, w->packed_offset, 1);
        BUFO(S.bands_slab, w->bands_offset * 4, 2);
        BUF(out, 3);
        CB(cbK, 4); CB(cbN, 5);
        if (w->outlier_count > 0 && S.outlier_idx_slab) {
            BUFO(S.outlier_idx_slab, w->outlier_offset * 4, 6);
            BUFO(S.outlier_val_slab, w->outlier_offset * 4, 7);
        } else {
            BUF(S.packed_slab, 6); BUF(S.packed_slab, 7);
        }
        CB(S.cb_oc[ocIdx], 8);
        DTG((w->rows + 15) / 16, 64);  // 2 simdgroups × 32 threads
        return;
    }

    // MLX-derived kernel: padded slab (rows zero-padded to multiple of 512)
    // Skip for K < 512 (e.g. kvDim=128 where padding waste is extreme)
    if (S.ps_mlx_qmv && S.mlx_packed_slab && w->lin_padded_cols >= 512) {
        int padK = w->lin_padded_cols;
        int group_off = (w->lin_packed_offset * 2) / LIN_GROUP_SIZE;
        ENC(S.ps_mlx_qmv);
        BUFO(S.mlx_packed_slab, w->lin_packed_offset, 0);
        BUFO(S.mlx_scales_slab, group_off * 4, 1);
        BUFO(S.mlx_biases_slab, group_off * 4, 2);
        BUF(act, 3);
        BUF(out, 4);
        int inSize = padK, outSize = w->rows;
        [enc setBytes:&inSize length:4 atIndex:5];
        [enc setBytes:&outSize length:4 atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(1, (w->rows + 7) / 8, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
        return;
    }

    // Fast per-row path — Q8-style inner loop, scale applied once per row
    if (S.ps_sq4mv_fast && S.fast_packed) {
        int row_off = w->lut_offset; // reused field: row offset into fast scale/bias arrays
        ENC(S.ps_sq4mv_fast);
        BUF(act, 0);
        BUFO(S.fast_packed, w->packed_offset, 1);
        BUFO(S.fast_row_scales, row_off * 4, 2);
        BUFO(S.fast_row_biases, row_off * 4, 3);
        BUF(out, 4);
        CB(cbK, 5); CB(cbN, 6);
        if (w->outlier_count > 0 && S.outlier_idx_slab) {
            BUFO(S.outlier_idx_slab, w->outlier_offset * 4, 7);
            BUFO(S.outlier_val_slab, w->outlier_offset * 4, 8);
        } else {
            BUF(S.fast_packed, 7); BUF(S.fast_packed, 8);
        }
        CB(S.cb_oc[ocIdx], 9);
        DTG((w->rows + 3) / 4, 256);
        return;
    }

    // Linear INT4 path — MLX-style mask-and-multiply, no LUT
    if (S.ps_sq4mv_lin && S.lin_packed_slab) {
        int group_off = (w->packed_offset * 2) / LIN_GROUP_SIZE;
        ENC(S.ps_sq4mv_lin);
        BUF(act, 0);
        BUFO(S.lin_packed_slab, w->packed_offset, 1);
        BUFO(S.lin_scales_slab, group_off * 4, 2);
        BUFO(S.lin_biases_slab, group_off * 4, 3);
        BUF(out, 4);
        CB(cbK, 5); CB(cbN, 6);
        if (w->outlier_count > 0 && S.outlier_idx_slab) {
            BUFO(S.outlier_idx_slab, w->outlier_offset * 4, 7);
            BUFO(S.outlier_val_slab, w->outlier_offset * 4, 8);
        } else {
            BUF(S.lin_packed_slab, 7); BUF(S.lin_packed_slab, 8);
        }
        CB(S.cb_oc[ocIdx], 9);
        DTG((w->rows + 3) / 4, 256);
        return;
    }

    // Texture LUT path — dequant through texture cache
    if (S.ps_sq4mv_tex && S.lut_texture) {
        ENC(S.ps_sq4mv_tex);
        BUF(act, 0);
        BUFO(S.packed_slab, w->packed_offset, 1);
        uint32_t lo = (uint32_t)w->lut_offset;
        [enc setBytes:&lo length:4 atIndex:2];
        BUF(out, 3);
        CB(cbK, 4); CB(cbN, 5);
        if (w->outlier_count > 0 && S.outlier_idx_slab) {
            BUFO(S.outlier_idx_slab, w->outlier_offset * 4, 6);
            BUFO(S.outlier_val_slab, w->outlier_offset * 4, 7);
        } else {
            BUF(S.packed_slab, 6); BUF(S.packed_slab, 7);
        }
        CB(S.cb_oc[ocIdx], 8);
        [enc setTexture:TX(S.lut_texture) atIndex:0];
        DTG((w->rows + 3) / 4, 256);
        return;
    }

    // Scalar threadgroup LUT fallback
    ENC(S.ps_sq4mv);
    BUF(act, 0);
    BUFO(S.packed_slab, w->packed_offset, 1);
    BUFO(S.bands_slab, w->bands_offset * 4, 2);
    BUF(out, 3);
    CB(cbK, 4); CB(cbN, 5);
    if (w->outlier_count > 0 && S.outlier_idx_slab) {
        BUFO(S.outlier_idx_slab, w->outlier_offset * 4, 6);
        BUFO(S.outlier_val_slab, w->outlier_offset * 4, 7);
    } else {
        BUF(S.packed_slab, 6); BUF(S.packed_slab, 7);
    }
    CB(S.cb_oc[ocIdx], 8);
    DTG((w->rows + 3) / 4, 256);
}

// Build linear INT4 re-encoding from SQ4 descriptors (lazy, called once)
static void build_linear_encoding(void) {
    if (S.lin_packed_slab) return; // already built

    int nL = S.nLayers;
    sq4_wt* all_wts[] = {S.wq, S.wk, S.wv, S.wo, S.wgate, S.wup, S.wdown};
    int n_wt_types = 7;

    // Calculate total size needed
    if (!S.shared_packed_copy || !S.shared_bands_copy) return;

    long long totalPacked = S.shared_packed_bytes;
    long long totalWeights = totalPacked * 2;
    int totalGroups = (int)(totalWeights / LIN_GROUP_SIZE);

    // Also compute per-ROW scale+bias for the fast kernel
    // We'll compute these alongside the per-group encoding

    fprintf(stderr, "[SQ4] alloc re-encoding: %lld bytes packed, %d groups\n", totalPacked, totalGroups);
    uint8_t* newPacked = (uint8_t*)calloc(1, totalPacked);
    float* newScales = (float*)calloc(totalGroups, sizeof(float));
    float* newBiases = (float*)calloc(totalGroups, sizeof(float));
    if (!newPacked || !newScales || !newBiases) {
        fprintf(stderr, "[SQ4] FATAL: calloc failed for re-encoding buffers\n");
        return;
    }
    uint8_t* srcPacked = S.shared_packed_copy;
    float* srcBands = S.shared_bands_copy;

    uint32_t* oIdx = S.shared_outlier_idx_copy;
    float* oVal = S.shared_outlier_val_copy;
    int totalOutliers = S.shared_outlier_count;

    // Process each weight tensor
    for (int t = 0; t < n_wt_types; t++) {
        for (int l = 0; l < nL; l++) {
            sq4_wt* w = &all_wts[t][l];
            if (w->rows == 0 || w->cols == 0) continue;

            float table16[16];
            int boff = w->bands_offset;
            for (int i = 0; i < 8; i++) {
                table16[i] = srcBands[boff + i];
                table16[i + 8] = -srcBands[boff + i];
            }

            int nWeights = w->rows * w->cols;
            int nGroups = nWeights / LIN_GROUP_SIZE;
            long long poff = w->packed_offset;
            int goff = (int)(poff * 2 / LIN_GROUP_SIZE);

            for (int g = 0; g < nGroups; g++) {
                int wstart = g * LIN_GROUP_SIZE;
                // Dequant group to find min/max, incorporating outlier corrections
                float gmin = 1e30f, gmax = -1e30f;
                float vals[LIN_GROUP_SIZE];
                for (int j = 0; j < LIN_GROUP_SIZE; j++) {
                    int wi = wstart + j;
                    long long byteIdx = poff + wi / 2;
                    int shift = (wi & 1) * 4;
                    int nib = (srcPacked[byteIdx] >> shift) & 0x0F;
                    vals[j] = table16[nib];
                    // Check if this weight is an outlier — binary search, replace with true value
                    if (w->outlier_count > 0 && oIdx) {
                        uint32_t flatIdx = (uint32_t)wi;
                        int lo = w->outlier_offset, hi = lo + w->outlier_count;
                        while (lo < hi) {
                            int mid = (lo + hi) >> 1;
                            if (oIdx[mid] < flatIdx) lo = mid + 1; else hi = mid;
                        }
                        if (lo < w->outlier_offset + w->outlier_count && oIdx[lo] == flatIdx)
                            vals[j] = oVal[lo];
                    }
                    if (vals[j] < gmin) gmin = vals[j];
                    if (vals[j] > gmax) gmax = vals[j];
                }

                float scale = (gmax - gmin) / 15.0f;
                if (scale < 1e-10f) scale = 1e-10f;
                float bias = gmin;
                newScales[goff + g] = scale;
                newBiases[goff + g] = bias;

                // Re-encode as linear INT4
                for (int j = 0; j < LIN_GROUP_SIZE; j++) {
                    int code = (int)((vals[j] - bias) / scale + 0.5f);
                    if (code < 0) code = 0;
                    if (code > 15) code = 15;
                    int wi = wstart + j;
                    long long byteIdx = poff + wi / 2;
                    int shift = (wi & 1) * 4;
                    if (shift == 0)
                        newPacked[byteIdx] = (newPacked[byteIdx] & 0xF0) | (code & 0x0F);
                    else
                        newPacked[byteIdx] = (newPacked[byteIdx] & 0x0F) | ((code & 0x0F) << 4);
                }
            }
        }
    }

    // Also handle lm_head
    {
        sq4_wt* w = &S.lmHead;
        if (w->rows > 0 && w->cols > 0) {
            float table16[16];
            int boff = w->bands_offset;
            for (int i = 0; i < 8; i++) {
                table16[i] = srcBands[boff + i];
                table16[i + 8] = -srcBands[boff + i];
            }
            int nWeights = w->rows * w->cols;
            int nGroups = nWeights / LIN_GROUP_SIZE;
            long long poff = w->packed_offset;
            int goff = (int)(poff * 2 / LIN_GROUP_SIZE);
            for (int g = 0; g < nGroups; g++) {
                int wstart = g * LIN_GROUP_SIZE;
                float gmin = 1e30f, gmax = -1e30f;
                float vals[LIN_GROUP_SIZE];
                for (int j = 0; j < LIN_GROUP_SIZE; j++) {
                    int wi = wstart + j;
                    long long byteIdx = poff + wi / 2;
                    int shift = (wi & 1) * 4;
                    int nib = (srcPacked[byteIdx] >> shift) & 0x0F;
                    vals[j] = table16[nib];
                    if (w->outlier_count > 0 && oIdx) {
                        uint32_t flatIdx = (uint32_t)wi;
                        int lo = w->outlier_offset, hi = lo + w->outlier_count;
                        while (lo < hi) { int mid = (lo+hi)>>1; if (oIdx[mid]<flatIdx) lo=mid+1; else hi=mid; }
                        if (lo < w->outlier_offset + w->outlier_count && oIdx[lo] == flatIdx)
                            vals[j] = oVal[lo];
                    }
                    if (vals[j] < gmin) gmin = vals[j];
                    if (vals[j] > gmax) gmax = vals[j];
                }
                float scale = (gmax - gmin) / 15.0f;
                if (scale < 1e-10f) scale = 1e-10f;
                newScales[goff + g] = scale;
                newBiases[goff + g] = gmin;
                for (int j = 0; j < LIN_GROUP_SIZE; j++) {
                    int code = (int)((vals[j] - gmin) / scale + 0.5f);
                    if (code < 0) code = 0;
                    if (code > 15) code = 15;
                    int wi = wstart + j;
                    long long byteIdx = poff + wi / 2;
                    int shift = (wi & 1) * 4;
                    if (shift == 0)
                        newPacked[byteIdx] = (newPacked[byteIdx] & 0xF0) | (code & 0x0F);
                    else
                        newPacked[byteIdx] = (newPacked[byteIdx] & 0x0F) | ((code & 0x0F) << 4);
                }
            }
        }
    }

    // Diagnostic
    {
        float t16[16];
        for (int i = 0; i < 8; i++) { t16[i] = srcBands[i]; t16[i+8] = -srcBands[i]; }
        fprintf(stderr, "[LIN] g0: scale=%.6f bias=%.6f\n", newScales[0], newBiases[0]);
        fprintf(stderr, "[LIN] orig nibs:"); for (int i=0;i<8;i++) fprintf(stderr," %d",(srcPacked[i/2]>>((i&1)*4))&0xF); fprintf(stderr,"\n");
        fprintf(stderr, "[LIN] new  nibs:"); for (int i=0;i<8;i++) fprintf(stderr," %d",(newPacked[i/2]>>((i&1)*4))&0xF); fprintf(stderr,"\n");
        fprintf(stderr, "[LIN] orig vals:"); for (int i=0;i<8;i++) fprintf(stderr," %.4f",t16[(srcPacked[i/2]>>((i&1)*4))&0xF]); fprintf(stderr,"\n");
        fprintf(stderr, "[LIN] recon vals:"); for (int i=0;i<8;i++) { int n=(newPacked[i/2]>>((i&1)*4))&0xF; fprintf(stderr," %.4f",n*newScales[0]+newBiases[0]); } fprintf(stderr,"\n");
    }

    // Upload to GPU
    S.lin_packed_slab = upload_private(newPacked, totalPacked);
    S.lin_scales_slab = upload_private(newScales, totalGroups * sizeof(float));
    S.lin_biases_slab = upload_private(newBiases, totalGroups * sizeof(float));

    // Build MLX-padded slab: rows padded to next multiple of 512 (MLX block_size)
    if (S.ps_mlx_qmv) {
        #define MLX_BLOCK 512
        long long mlxTotalPacked = 0; int mlxTotalGroups = 0;
        for (int t = 0; t < n_wt_types; t++)
            for (int l = 0; l < nL; l++) {
                sq4_wt* w = &all_wts[t][l];
                if (w->rows == 0 || w->cols == 0) continue;
                int padK = (w->cols + MLX_BLOCK - 1) / MLX_BLOCK * MLX_BLOCK;
                mlxTotalPacked += w->rows * padK / 2;
                mlxTotalGroups += w->rows * (padK / LIN_GROUP_SIZE);
            }
        { sq4_wt* w = &S.lmHead; if (w->rows > 0 && w->cols > 0) {
            int padK = (w->cols + MLX_BLOCK - 1) / MLX_BLOCK * MLX_BLOCK;
            mlxTotalPacked += w->rows * padK / 2;
            mlxTotalGroups += w->rows * (padK / LIN_GROUP_SIZE);
        }}

        uint8_t* mp = (uint8_t*)calloc(1, mlxTotalPacked);
        float* ms = (float*)calloc(mlxTotalGroups, sizeof(float));
        float* mb = (float*)calloc(mlxTotalGroups, sizeof(float));
        long long mpo = 0; int mgo = 0;

        for (int t = 0; t < n_wt_types; t++)
            for (int l = 0; l < nL; l++) {
                sq4_wt* w = &all_wts[t][l];
                if (w->rows == 0 || w->cols == 0) continue;
                int K = w->cols, padK = (K + MLX_BLOCK - 1) / MLX_BLOCK * MLX_BLOCK;
                int sg = (int)(w->packed_offset * 2 / LIN_GROUP_SIZE);
                w->lin_packed_offset = mpo;
                w->lin_padded_cols = padK;
                for (int r = 0; r < w->rows; r++) {
                    memcpy(mp + mpo + r*(padK/2), newPacked + w->packed_offset + r*(K/2), K/2);
                    int rg = K / LIN_GROUP_SIZE, pg = padK / LIN_GROUP_SIZE;
                    memcpy(ms + mgo + r*pg, newScales + sg + r*rg, rg * sizeof(float));
                    memcpy(mb + mgo + r*pg, newBiases + sg + r*rg, rg * sizeof(float));
                }
                mpo += w->rows * (padK / 2);
                mgo += w->rows * (padK / LIN_GROUP_SIZE);
            }
        { sq4_wt* w = &S.lmHead; if (w->rows > 0 && w->cols > 0) {
            int K = w->cols, padK = (K + MLX_BLOCK - 1) / MLX_BLOCK * MLX_BLOCK;
            int sg = (int)(w->packed_offset * 2 / LIN_GROUP_SIZE);
            w->lin_packed_offset = mpo;
            w->lin_padded_cols = padK;
            for (int r = 0; r < w->rows; r++) {
                memcpy(mp + mpo + r*(padK/2), newPacked + w->packed_offset + r*(K/2), K/2);
                int rg = K / LIN_GROUP_SIZE, pg = padK / LIN_GROUP_SIZE;
                memcpy(ms + mgo + r*pg, newScales + sg + r*rg, rg * sizeof(float));
                memcpy(mb + mgo + r*pg, newBiases + sg + r*rg, rg * sizeof(float));
            }
            mpo += w->rows * (padK / 2);
            mgo += w->rows * (padK / LIN_GROUP_SIZE);
        }}

        // Diagnostic: compare first row dequant from lin vs mlx slab
        {
            sq4_wt* w0 = &all_wts[0][0]; // wq layer 0
            int K = w0->cols;
            int padK = w0->lin_padded_cols;
            long long linOff = w0->packed_offset;
            long long mlxOff = w0->lin_packed_offset;
            int linGrp = (linOff * 2) / LIN_GROUP_SIZE;
            int mlxGrp = (mlxOff * 2) / LIN_GROUP_SIZE;
            fprintf(stderr, "[MLX-DIAG] wq0: K=%d padK=%d linOff=%d mlxOff=%d\n", K, padK, linOff, mlxOff);
            fprintf(stderr, "[MLX-DIAG] lin row0 nibs:");
            for (int i = 0; i < 16; i++) {
                int bi = linOff + i/2;
                int nib = (newPacked[bi] >> ((i&1)*4)) & 0xF;
                fprintf(stderr, " %d", nib);
            }
            fprintf(stderr, "\n[MLX-DIAG] mlx row0 nibs:");
            for (int i = 0; i < 16; i++) {
                int bi = mlxOff + i/2;
                int nib = (mp[bi] >> ((i&1)*4)) & 0xF;
                fprintf(stderr, " %d", nib);
            }
            fprintf(stderr, "\n[MLX-DIAG] lin scales[0..3]: %.6f %.6f %.6f %.6f\n",
                newScales[linGrp], newScales[linGrp+1], newScales[linGrp+2], newScales[linGrp+3]);
            fprintf(stderr, "[MLX-DIAG] mlx scales[0..3]: %.6f %.6f %.6f %.6f\n",
                ms[mlxGrp], ms[mlxGrp+1], ms[mlxGrp+2], ms[mlxGrp+3]);
            // Dequant first 8 vals from each
            fprintf(stderr, "[MLX-DIAG] lin dequant:");
            for (int i = 0; i < 8; i++) {
                int bi = linOff + i/2;
                int nib = (newPacked[bi] >> ((i&1)*4)) & 0xF;
                int grp = linGrp + i / LIN_GROUP_SIZE;
                float v = nib * newScales[grp] + newBiases[grp];
                fprintf(stderr, " %.4f", v);
            }
            fprintf(stderr, "\n[MLX-DIAG] mlx dequant:");
            for (int i = 0; i < 8; i++) {
                int bi = mlxOff + i/2;
                int nib = (mp[bi] >> ((i&1)*4)) & 0xF;
                int grp = mlxGrp + i / (padK > 0 ? LIN_GROUP_SIZE : 1);
                float v = nib * ms[grp] + mb[grp];
                fprintf(stderr, " %.4f", v);
            }
            fprintf(stderr, "\n");
            // Check padded region
            int padStart = K/2; // first padded byte in row 0
            fprintf(stderr, "[MLX-DIAG] mlx pad nibs (byte %d):", padStart);
            for (int i = 0; i < 8; i++) {
                int bi = mlxOff + padStart + i/2;
                int nib = (mp[bi] >> ((i&1)*4)) & 0xF;
                fprintf(stderr, " %d", nib);
            }
            int padGrpStart = mlxGrp + K/LIN_GROUP_SIZE;
            fprintf(stderr, "\n[MLX-DIAG] mlx pad scales: %.6f %.6f\n", ms[padGrpStart], ms[padGrpStart+1]);
            // Check row 1 alignment
            int linRow1Off = linOff + K/2;  // row 1 in lin slab
            int mlxRow1Off = mlxOff + padK/2; // row 1 in mlx slab
            fprintf(stderr, "[MLX-DIAG] row1 lin nibs:");
            for (int i = 0; i < 8; i++) { int bi = linRow1Off + i/2; fprintf(stderr, " %d", (newPacked[bi] >> ((i&1)*4)) & 0xF); }
            fprintf(stderr, "\n[MLX-DIAG] row1 mlx nibs:");
            for (int i = 0; i < 8; i++) { int bi = mlxRow1Off + i/2; fprintf(stderr, " %d", (mp[bi] >> ((i&1)*4)) & 0xF); }
            fprintf(stderr, "\n[MLX-DIAG] row1 match: %s\n",
                memcmp(newPacked+linRow1Off, mp+mlxRow1Off, K/2) == 0 ? "YES" : "NO");
        }

        S.mlx_packed_slab = upload_private(mp, mlxTotalPacked);
        S.mlx_scales_slab = upload_private(ms, mlxTotalGroups * sizeof(float));
        S.mlx_biases_slab = upload_private(mb, mlxTotalGroups * sizeof(float));
        free(mp); free(ms); free(mb);
        NSLog(@"[SQ4] MLX padded slab: %lld bytes packed, %d groups", mlxTotalPacked, mlxTotalGroups);
        #undef MLX_BLOCK
    }

    free(newPacked); free(newScales); free(newBiases);
    // Build fast (per-row) encoding: re-encode nibbles with per-tensor scale+bias
    // Each tensor's 8 band means map to 8 int4 codes via one affine per tensor.
    // All rows in a tensor share the same scale+bias → stored per-row for fast indexing.
    {
        // Count total rows across all weight tensors
        int totalRows = 0;
        for (int t = 0; t < n_wt_types; t++)
            for (int l = 0; l < nL; l++) totalRows += all_wts[t][l].rows;
        totalRows += S.lmHead.rows;

        uint8_t* fastPacked = (uint8_t*)calloc(1, totalPacked);
        float* fastScales = (float*)malloc(totalRows * sizeof(float));
        float* fastBiases = (float*)malloc(totalRows * sizeof(float));

        // For each tensor: compute scale+bias from its 8 band means,
        // re-encode nibbles, fill per-row scale+bias arrays
        int rowOff = 0;
        for (int t = 0; t < n_wt_types; t++) {
            for (int l = 0; l < nL; l++) {
                sq4_wt* w = &all_wts[t][l];
                if (w->rows == 0 || w->cols == 0) continue;

                // Get band means for this tensor
                float table16[16];
                int boff = w->bands_offset;
                for (int i = 0; i < 8; i++) {
                    table16[i] = srcBands[boff + i];
                    table16[i + 8] = -srcBands[boff + i];
                }

                // Find min/max across all 16 possible dequant values
                float bmin = table16[0], bmax = table16[0];
                for (int i = 1; i < 16; i++) {
                    if (table16[i] < bmin) bmin = table16[i];
                    if (table16[i] > bmax) bmax = table16[i];
                }
                float scale = (bmax - bmin) / 15.0f;
                if (scale < 1e-10f) scale = 1e-10f;
                float bias = bmin;

                // Build nibble mapping: old_nibble → new_code
                uint8_t remap[16];
                for (int i = 0; i < 16; i++) {
                    int code = (int)((table16[i] - bias) / scale + 0.5f);
                    if (code < 0) code = 0;
                    if (code > 15) code = 15;
                    remap[i] = (uint8_t)code;
                }

                // Re-encode packed nibbles
                long long poff = w->packed_offset;
                int nWeights = w->rows * w->cols;
                for (int i = 0; i < nWeights; i++) {
                    long long byteIdx = poff + i / 2;
                    int shift = (i & 1) * 4;
                    uint8_t oldNib = (srcPacked[byteIdx] >> shift) & 0x0F;
                    uint8_t newNib = remap[oldNib];
                    if (shift == 0)
                        fastPacked[byteIdx] = (fastPacked[byteIdx] & 0xF0) | newNib;
                    else
                        fastPacked[byteIdx] = (fastPacked[byteIdx] & 0x0F) | (newNib << 4);
                }

                // Fill per-row scale+bias (same for all rows in this tensor)
                for (int r = 0; r < w->rows; r++) {
                    fastScales[rowOff + r] = scale;
                    fastBiases[rowOff + r] = bias;
                }
                w->lut_offset = rowOff; // reuse this field for row offset into fast arrays
                rowOff += w->rows;
            }
        }

        // lm_head
        {
            sq4_wt* w = &S.lmHead;
            if (w->rows > 0 && w->cols > 0) {
                float table16[16];
                int boff = w->bands_offset;
                for (int i = 0; i < 8; i++) {
                    table16[i] = srcBands[boff + i];
                    table16[i + 8] = -srcBands[boff + i];
                }
                float bmin = table16[0], bmax = table16[0];
                for (int i = 1; i < 16; i++) {
                    if (table16[i] < bmin) bmin = table16[i];
                    if (table16[i] > bmax) bmax = table16[i];
                }
                float scale = (bmax - bmin) / 15.0f;
                if (scale < 1e-10f) scale = 1e-10f;
                float bias = bmin;
                uint8_t remap[16];
                for (int i = 0; i < 16; i++) {
                    int code = (int)((table16[i] - bias) / scale + 0.5f);
                    if (code < 0) code = 0; if (code > 15) code = 15;
                    remap[i] = (uint8_t)code;
                }
                long long poff = w->packed_offset;
                int nWeights = w->rows * w->cols;
                for (int i = 0; i < nWeights; i++) {
                    long long byteIdx = poff + i / 2;
                    int shift = (i & 1) * 4;
                    uint8_t oldNib = (srcPacked[byteIdx] >> shift) & 0x0F;
                    uint8_t newNib = remap[oldNib];
                    if (shift == 0)
                        fastPacked[byteIdx] = (fastPacked[byteIdx] & 0xF0) | newNib;
                    else
                        fastPacked[byteIdx] = (fastPacked[byteIdx] & 0x0F) | (newNib << 4);
                }
                for (int r = 0; r < w->rows; r++) {
                    fastScales[rowOff + r] = scale;
                    fastBiases[rowOff + r] = bias;
                }
                w->lut_offset = rowOff;
                rowOff += w->rows;
            }
        }

        S.fast_packed = upload_private(fastPacked, totalPacked);
        S.fast_row_scales = upload_private(fastScales, rowOff * sizeof(float));
        S.fast_row_biases = upload_private(fastBiases, rowOff * sizeof(float));
        free(fastPacked); free(fastScales); free(fastBiases);
        NSLog(@"[SQ4] fast per-row encoding built: %d rows", rowOff);
    }

    // Free shared copies — no longer needed
    free(S.shared_packed_copy); S.shared_packed_copy = NULL;
    free(S.shared_bands_copy); S.shared_bands_copy = NULL;
    free(S.shared_outlier_idx_copy); S.shared_outlier_idx_copy = NULL;
    free(S.shared_outlier_val_copy); S.shared_outlier_val_copy = NULL;
    NSLog(@"[SQ4] linear re-encoding built: %d groups", totalGroups);

    // GPU diagnostic: compare linear vs MLX kernel output on wq layer 0
    if (S.ps_mlx_qmv && S.mlx_packed_slab && S.ps_sq4mv_lin && S.lin_packed_slab) {
        sq4_wt* w0 = &S.wq[0];
        int K = w0->cols, N = w0->rows;
        int padK = w0->lin_padded_cols;

        // Create test input: ones vector
        float* testIn = (float*)calloc(padK, sizeof(float));
        for (int i = 0; i < K; i++) testIn[i] = (i % 7) * 0.1f - 0.3f;
        id<MTLBuffer> bIn = [g_device newBufferWithBytes:testIn length:padK*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bOutLin = [g_device newBufferWithLength:N*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bOutMLX = [g_device newBufferWithLength:N*4 options:MTLResourceStorageModeShared];
        free(testIn);

        int linGrpOff = (w0->packed_offset * 2) / LIN_GROUP_SIZE;
        int mlxGrpOff = (w0->lin_packed_offset * 2) / LIN_GROUP_SIZE;

        id<MTLCommandBuffer> tcmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> tenc = [tcmd computeCommandEncoder];

        // Linear kernel
        [tenc setComputePipelineState:PS(S.ps_sq4mv_lin)];
        [tenc setBuffer:bIn offset:0 atIndex:0];
        [tenc setBuffer:B(S.lin_packed_slab) offset:w0->packed_offset atIndex:1];
        [tenc setBuffer:B(S.lin_scales_slab) offset:linGrpOff*4 atIndex:2];
        [tenc setBuffer:B(S.lin_biases_slab) offset:linGrpOff*4 atIndex:3];
        [tenc setBuffer:bOutLin offset:0 atIndex:4];
        uint32_t tK = K, tN = N, tOC = 0;
        [tenc setBytes:&tK length:4 atIndex:5];
        [tenc setBytes:&tN length:4 atIndex:6];
        [tenc setBuffer:B(S.lin_packed_slab) offset:0 atIndex:7];
        [tenc setBuffer:B(S.lin_packed_slab) offset:0 atIndex:8];
        [tenc setBytes:&tOC length:4 atIndex:9];
        [tenc dispatchThreadgroups:MTLSizeMake((N+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [tenc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // MLX kernel
        [tenc setComputePipelineState:PS(S.ps_mlx_qmv)];
        [tenc setBuffer:B(S.mlx_packed_slab) offset:w0->lin_packed_offset atIndex:0];
        [tenc setBuffer:B(S.mlx_scales_slab) offset:mlxGrpOff*4 atIndex:1];
        [tenc setBuffer:B(S.mlx_biases_slab) offset:mlxGrpOff*4 atIndex:2];
        [tenc setBuffer:bIn offset:0 atIndex:3];
        [tenc setBuffer:bOutMLX offset:0 atIndex:4];
        int inSz = padK, outSz = N;
        [tenc setBytes:&inSz length:4 atIndex:5];
        [tenc setBytes:&outSz length:4 atIndex:6];
        [tenc dispatchThreadgroups:MTLSizeMake(1, (N+7)/8, 1) threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];

        [tenc endEncoding];
        [tcmd commit];
        [tcmd waitUntilCompleted];

        float* rLin = (float*)bOutLin.contents;
        float* rMLX = (float*)bOutMLX.contents;
        float maxDiff = 0;
        int worstIdx = 0;
        for (int i = 0; i < N; i++) {
            float d = rLin[i] - rMLX[i];
            if (d < 0) d = -d;
            if (d > maxDiff) { maxDiff = d; worstIdx = i; }
        }
        fprintf(stderr, "[GPU-DIAG] linear[0..3]: %.6f %.6f %.6f %.6f\n", rLin[0], rLin[1], rLin[2], rLin[3]);
        fprintf(stderr, "[GPU-DIAG] mlx   [0..3]: %.6f %.6f %.6f %.6f\n", rMLX[0], rMLX[1], rMLX[2], rMLX[3]);
        fprintf(stderr, "[GPU-DIAG] maxDiff=%.6f at row %d (lin=%.6f mlx=%.6f) %s\n",
            maxDiff, worstIdx, rLin[worstIdx], rMLX[worstIdx], maxDiff < 0.01 ? "PASS" : "FAIL");
    }
}

// ============================================================================
// API: Forward pass
// ============================================================================

int mtl_sq4_infer_step(int tokenID, int pos, float* logitsOut) {
    if (!S.built) return -1;
    if (S.ps_sq4mv_lin && !S.lin_packed_slab) {
        fprintf(stderr, "[SQ4] building linear encoding...\n");
        build_linear_encoding();
        fprintf(stderr, "[SQ4] linear encoding done\n");
    }
    int dim=S.dim, kvDim=S.kvDim, headDim=S.headDim;
    int nHeads=S.nHeads, nKVHeads=S.nKVHeads, ffnDim=S.ffnDim;
    int nLayers=S.nLayers, vocabSize=S.vocabSize;
    int seqLen = pos + 1;

    static int timing_count = 0;
    static double timing_total = 0;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    ((uint32_t*)B(S.cb_pos).contents)[0] = (uint32_t)pos;
    ((uint32_t*)B(S.cb_seq).contents)[0] = (uint32_t)seqLen;

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    NSUInteger tpg = (dim / 32) * 32;
    if (tpg == 0) tpg = 32;
    if (tpg > PS(S.ps_rmsnorm_out).maxTotalThreadsPerThreadgroup)
        tpg = PS(S.ps_rmsnorm_out).maxTotalThreadsPerThreadgroup;

    // Embed gather
    int embedOff = tokenID * dim * 4;
    ENC(S.ps_copy); BUFO(S.embedBuf, embedOff, 0); BUF(S.hidden, 1); DT(dim, 256); BAR();

    for (int l = 0; l < nLayers; l++) {
        // RMSNorm 1
        ENC(S.ps_rmsnorm_out);
        BUF(S.hidden, 0); BUF(S.normed, 1); BUF(S.norm1[l], 2);
        CB(S.cb_dim, 3); CB(S.cb_eps, 4); DTG(1, tpg); BAR();

        // QKV matvecs
        sq4mv(enc, S.normed, &S.wq[l], S.Q, S.cb_dim, S.cb_Ndim, l*7+0);
        sq4mv(enc, S.normed, &S.wk[l], S.K, S.cb_dim, S.cb_Nkvdim, l*7+1);
        sq4mv(enc, S.normed, &S.wv[l], S.V, S.cb_dim, S.cb_Nkvdim, l*7+2);
        BAR();

        // (in-flight diagnostic removed)

        // Fused: bias + RoPE + KV write (1 dispatch instead of 7)
        if (S.ps_fused_brk) {
            ENC(S.ps_fused_brk);
            BUF(S.Q, 0); BUF(S.K, 1); BUF(S.V, 2);
            BUF(S.bq[l], 3); BUF(S.bk[l], 4); BUF(S.bv[l], 5);
            BUF(S.kCache[l], 6); BUF(S.vCache[l], 7);
            CB(S.cb_dim, 8); CB(S.cb_kvDim, 9); CB(S.cb_headDim, 10);
            CB(S.cb_nHeads, 11); CB(S.cb_nKVHeads, 12);
            CB(S.cb_pos, 13); CB(S.cb_theta, 14);
            DTG(1, 256);
        } else {
            // Fallback: separate dispatches
            ENC(S.ps_bias_add);
            BUF(S.Q, 0); BUF(S.bq[l], 1); CB(S.cb_dim, 2); DT(dim, 256);
            BUF(S.K, 0); BUF(S.bk[l], 1); CB(S.cb_kvDim, 2); DT(kvDim, 256);
            BUF(S.V, 0); BUF(S.bv[l], 1); CB(S.cb_kvDim, 2); DT(kvDim, 256);
            BAR();
            int nPQ = nHeads * (headDim / 2), nPK = nKVHeads * (headDim / 2);
            ENC(S.ps_rope);
            BUF(S.Q, 0); CB(S.cb_headDim, 1); CB(S.cb_nHeads, 2); CB(S.cb_pos, 3); CB(S.cb_theta, 4);
            DT(nPQ, MIN(256, (int)PS(S.ps_rope).maxTotalThreadsPerThreadgroup));
            BUF(S.K, 0); CB(S.cb_headDim, 1); CB(S.cb_nKVHeads, 2); CB(S.cb_pos, 3); CB(S.cb_theta, 4);
            DT(nPK, MIN(256, (int)PS(S.ps_rope).maxTotalThreadsPerThreadgroup));
            BAR();
            int cOff = pos * kvDim * 4;
            ENC(S.ps_copy);
            BUF(S.K, 0); BUFO(S.kCache[l], cOff, 1); DT(kvDim, 256);
            BUF(S.V, 0); BUFO(S.vCache[l], cOff, 1); DT(kvDim, 256);
        }
        BAR();

        // Attention
        ENC(S.ps_attn);
        BUF(S.Q, 0); BUF(S.kCache[l], 1); BUF(S.vCache[l], 2); BUF(S.attnOut, 3);
        CB(S.cb_kvDim, 4); CB(S.cb_headDim, 5); CB(S.cb_nHeads, 6);
        CB(S.cb_nKVHeads, 7); CB(S.cb_seq, 8);
        DTG(nHeads, headDim); BAR();

        // Wo + residual
        sq4mv(enc, S.attnOut, &S.wo[l], S.proj, S.cb_dim, S.cb_Ndim, l*7+3); BAR();
        ENC(S.ps_add_inplace); BUF(S.hidden, 0); BUF(S.proj, 1); DT(dim, 256); BAR();

        // RMSNorm 2
        ENC(S.ps_rmsnorm_out);
        BUF(S.hidden, 0); BUF(S.normed2, 1); BUF(S.norm2[l], 2);
        CB(S.cb_dim, 3); CB(S.cb_eps, 4); DTG(1, tpg); BAR();

        // FFN: fused gate+up+SiLU
        if (S.ps_fused_gus_lin && S.lin_packed_slab) {
            // Fused linear: no LUT, 1 dispatch
            sq4_wt* wg = &S.wgate[l];
            sq4_wt* wu = &S.wup[l];
            int g_goff = (wg->packed_offset * 2) / LIN_GROUP_SIZE;
            int u_goff = (wu->packed_offset * 2) / LIN_GROUP_SIZE;
            ENC(S.ps_fused_gus_lin);
            BUF(S.normed2, 0);
            BUFO(S.lin_packed_slab, wg->packed_offset, 1);
            BUFO(S.lin_scales_slab, g_goff * 4, 2);
            BUFO(S.lin_biases_slab, g_goff * 4, 3);
            BUFO(S.lin_packed_slab, wu->packed_offset, 4);
            BUFO(S.lin_scales_slab, u_goff * 4, 5);
            BUFO(S.lin_biases_slab, u_goff * 4, 6);
            BUF(S.ffnMid, 7);
            CB(S.cb_dim, 8); CB(S.cb_Nffn, 9);
            if (wg->outlier_count > 0 && S.outlier_idx_slab) {
                BUFO(S.outlier_idx_slab, wg->outlier_offset * 4, 10);
                BUFO(S.outlier_val_slab, wg->outlier_offset * 4, 11);
            } else { BUF(S.lin_packed_slab, 10); BUF(S.lin_packed_slab, 11); }
            CB(S.cb_oc[l*7+4], 12);
            if (wu->outlier_count > 0 && S.outlier_idx_slab) {
                BUFO(S.outlier_idx_slab, wu->outlier_offset * 4, 13);
                BUFO(S.outlier_val_slab, wu->outlier_offset * 4, 14);
            } else { BUF(S.lin_packed_slab, 13); BUF(S.lin_packed_slab, 14); }
            CB(S.cb_oc[l*7+5], 15);
            DTG((ffnDim + 3) / 4, 256);
            BAR();
        } else if (S.ps_fused_gus) {
            sq4_wt* wg = &S.wgate[l];
            sq4_wt* wu = &S.wup[l];
            ENC(S.ps_fused_gus);
            BUF(S.normed2, 0);
            BUFO(S.packed_slab, wg->packed_offset, 1);
            BUFO(S.bands_slab, wg->bands_offset * 4, 2);
            BUFO(S.packed_slab, wu->packed_offset, 3);
            BUFO(S.bands_slab, wu->bands_offset * 4, 4);
            BUF(S.ffnMid, 5);
            CB(S.cb_dim, 6); CB(S.cb_Nffn, 7);
            if (wg->outlier_count > 0 && S.outlier_idx_slab) {
                BUFO(S.outlier_idx_slab, wg->outlier_offset * 4, 8);
                BUFO(S.outlier_val_slab, wg->outlier_offset * 4, 9);
            } else {
                BUF(S.packed_slab, 8); BUF(S.packed_slab, 9);
            }
            CB(S.cb_oc[l*7+4], 10);
            if (wu->outlier_count > 0 && S.outlier_idx_slab) {
                BUFO(S.outlier_idx_slab, wu->outlier_offset * 4, 11);
                BUFO(S.outlier_val_slab, wu->outlier_offset * 4, 12);
            } else {
                BUF(S.packed_slab, 11); BUF(S.packed_slab, 12);
            }
            CB(S.cb_oc[l*7+5], 13);
            DTG((ffnDim + 3) / 4, 256);
            BAR();
        } else {
            sq4mv(enc, S.normed2, &S.wgate[l], S.gatePre, S.cb_dim, S.cb_Nffn, l*7+4);
            sq4mv(enc, S.normed2, &S.wup[l], S.upOut, S.cb_dim, S.cb_Nffn, l*7+5);
            BAR();
            ENC(S.ps_silu_gate_mul);
            BUF(S.gatePre, 0); BUF(S.upOut, 1); BUF(S.ffnMid, 2); DT(ffnDim, 256);
            BAR();
        }
        sq4mv(enc, S.ffnMid, &S.wdown[l], S.proj, S.cb_ffnDim, S.cb_Ndim, l*7+6); BAR();
        ENC(S.ps_add_inplace); BUF(S.hidden, 0); BUF(S.proj, 1); DT(dim, 256); BAR();
    }

    // Final norm (in-place) + lm_head
    ENC(S.ps_rmsnorm_save);
    BUF(S.hidden, 0); BUF(S.finalNorm, 1); BUF(S.normed, 2);
    CB(S.cb_dim, 3); CB(S.cb_eps, 4); DTG(1, tpg); BAR();
    sq4mv(enc, S.hidden, &S.lmHead, S.logits, S.cb_dim, S.cb_Nvocab, 7*nLayers);

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec)*1000.0 + (t1.tv_nsec - t0.tv_nsec)/1e6;
    timing_total += ms;
    timing_count++;
    if (timing_count == 50 || timing_count == 200) {
        fprintf(stderr, "[SQ4-PERF] step %d: avg %.2f ms/tok (%.1f tok/s)\n",
            timing_count, timing_total/timing_count, 1000.0*timing_count/timing_total);
    }

    memcpy(logitsOut, B(S.logits).contents, vocabSize * 4);
    return 0;
}

int mtl_sq4_infer_step_sample(int tokenID, int pos) {
    if (!S.built) return -1;
    if (S.ps_sq4mv_lin && !S.lin_packed_slab) build_linear_encoding();
    int dim=S.dim, kvDim=S.kvDim, headDim=S.headDim;
    int nHeads=S.nHeads, nKVHeads=S.nKVHeads, ffnDim=S.ffnDim;
    int nLayers=S.nLayers, vocabSize=S.vocabSize;
    int seqLen = pos + 1;

    static int ss_count = 0;
    static double ss_total = 0, ss_gpu = 0, ss_cpu = 0;
    struct timespec t0, t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    ((uint32_t*)B(S.cb_pos).contents)[0] = (uint32_t)pos;
    ((uint32_t*)B(S.cb_seq).contents)[0] = (uint32_t)seqLen;

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    NSUInteger tpg = (dim / 32) * 32;
    if (tpg == 0) tpg = 32;
    if (tpg > PS(S.ps_rmsnorm_out).maxTotalThreadsPerThreadgroup)
        tpg = PS(S.ps_rmsnorm_out).maxTotalThreadsPerThreadgroup;

    int embedOff = tokenID * dim * 4;
    ENC(S.ps_copy); BUFO(S.embedBuf, embedOff, 0); BUF(S.hidden, 1); DT(dim, 256); BAR();

    for (int l = 0; l < nLayers; l++) {
        ENC(S.ps_rmsnorm_out);
        BUF(S.hidden, 0); BUF(S.normed, 1); BUF(S.norm1[l], 2);
        CB(S.cb_dim, 3); CB(S.cb_eps, 4); DTG(1, tpg); BAR();

        sq4mv(enc, S.normed, &S.wq[l], S.Q, S.cb_dim, S.cb_Ndim, l*7+0);
        sq4mv(enc, S.normed, &S.wk[l], S.K, S.cb_dim, S.cb_Nkvdim, l*7+1);
        sq4mv(enc, S.normed, &S.wv[l], S.V, S.cb_dim, S.cb_Nkvdim, l*7+2);
        BAR();

        // Fused: bias + RoPE + KV write (1 dispatch instead of 7)
        if (S.ps_fused_brk) {
            ENC(S.ps_fused_brk);
            BUF(S.Q, 0); BUF(S.K, 1); BUF(S.V, 2);
            BUF(S.bq[l], 3); BUF(S.bk[l], 4); BUF(S.bv[l], 5);
            BUF(S.kCache[l], 6); BUF(S.vCache[l], 7);
            CB(S.cb_dim, 8); CB(S.cb_kvDim, 9); CB(S.cb_headDim, 10);
            CB(S.cb_nHeads, 11); CB(S.cb_nKVHeads, 12);
            CB(S.cb_pos, 13); CB(S.cb_theta, 14);
            DTG(1, 256);
        } else {
            ENC(S.ps_bias_add);
            BUF(S.Q, 0); BUF(S.bq[l], 1); CB(S.cb_dim, 2); DT(dim, 256);
            BUF(S.K, 0); BUF(S.bk[l], 1); CB(S.cb_kvDim, 2); DT(kvDim, 256);
            BUF(S.V, 0); BUF(S.bv[l], 1); CB(S.cb_kvDim, 2); DT(kvDim, 256);
            BAR();
            int nPQ = nHeads * (headDim / 2), nPK = nKVHeads * (headDim / 2);
            ENC(S.ps_rope);
            BUF(S.Q, 0); CB(S.cb_headDim, 1); CB(S.cb_nHeads, 2); CB(S.cb_pos, 3); CB(S.cb_theta, 4);
            DT(nPQ, MIN(256, (int)PS(S.ps_rope).maxTotalThreadsPerThreadgroup));
            BUF(S.K, 0); CB(S.cb_headDim, 1); CB(S.cb_nKVHeads, 2); CB(S.cb_pos, 3); CB(S.cb_theta, 4);
            DT(nPK, MIN(256, (int)PS(S.ps_rope).maxTotalThreadsPerThreadgroup));
            BAR();
            int cOff = pos * kvDim * 4;
            ENC(S.ps_copy);
            BUF(S.K, 0); BUFO(S.kCache[l], cOff, 1); DT(kvDim, 256);
            BUF(S.V, 0); BUFO(S.vCache[l], cOff, 1); DT(kvDim, 256);
        }
        BAR();

        ENC(S.ps_attn);
        BUF(S.Q, 0); BUF(S.kCache[l], 1); BUF(S.vCache[l], 2); BUF(S.attnOut, 3);
        CB(S.cb_kvDim, 4); CB(S.cb_headDim, 5); CB(S.cb_nHeads, 6);
        CB(S.cb_nKVHeads, 7); CB(S.cb_seq, 8);
        DTG(nHeads, headDim); BAR();

        sq4mv(enc, S.attnOut, &S.wo[l], S.proj, S.cb_dim, S.cb_Ndim, l*7+3); BAR();
        ENC(S.ps_add_inplace); BUF(S.hidden, 0); BUF(S.proj, 1); DT(dim, 256); BAR();

        ENC(S.ps_rmsnorm_out);
        BUF(S.hidden, 0); BUF(S.normed2, 1); BUF(S.norm2[l], 2);
        CB(S.cb_dim, 3); CB(S.cb_eps, 4); DTG(1, tpg); BAR();

        if (S.ps_fused_gus_lin && S.lin_packed_slab) {
            sq4_wt* wg = &S.wgate[l];
            sq4_wt* wu = &S.wup[l];
            int g_goff = (wg->packed_offset * 2) / LIN_GROUP_SIZE;
            int u_goff = (wu->packed_offset * 2) / LIN_GROUP_SIZE;
            ENC(S.ps_fused_gus_lin);
            BUF(S.normed2, 0);
            BUFO(S.lin_packed_slab, wg->packed_offset, 1);
            BUFO(S.lin_scales_slab, g_goff * 4, 2);
            BUFO(S.lin_biases_slab, g_goff * 4, 3);
            BUFO(S.lin_packed_slab, wu->packed_offset, 4);
            BUFO(S.lin_scales_slab, u_goff * 4, 5);
            BUFO(S.lin_biases_slab, u_goff * 4, 6);
            BUF(S.ffnMid, 7);
            CB(S.cb_dim, 8); CB(S.cb_Nffn, 9);
            if (wg->outlier_count > 0 && S.outlier_idx_slab) {
                BUFO(S.outlier_idx_slab, wg->outlier_offset * 4, 10);
                BUFO(S.outlier_val_slab, wg->outlier_offset * 4, 11);
            } else { BUF(S.lin_packed_slab, 10); BUF(S.lin_packed_slab, 11); }
            CB(S.cb_oc[l*7+4], 12);
            if (wu->outlier_count > 0 && S.outlier_idx_slab) {
                BUFO(S.outlier_idx_slab, wu->outlier_offset * 4, 13);
                BUFO(S.outlier_val_slab, wu->outlier_offset * 4, 14);
            } else { BUF(S.lin_packed_slab, 13); BUF(S.lin_packed_slab, 14); }
            CB(S.cb_oc[l*7+5], 15);
            DTG((ffnDim + 3) / 4, 256); BAR();
        } else if (S.ps_fused_gus) {
            sq4_wt* wg = &S.wgate[l];
            sq4_wt* wu = &S.wup[l];
            ENC(S.ps_fused_gus);
            BUF(S.normed2, 0);
            BUFO(S.packed_slab, wg->packed_offset, 1);
            BUFO(S.bands_slab, wg->bands_offset * 4, 2);
            BUFO(S.packed_slab, wu->packed_offset, 3);
            BUFO(S.bands_slab, wu->bands_offset * 4, 4);
            BUF(S.ffnMid, 5);
            CB(S.cb_dim, 6); CB(S.cb_Nffn, 7);
            if (wg->outlier_count > 0 && S.outlier_idx_slab) {
                BUFO(S.outlier_idx_slab, wg->outlier_offset * 4, 8);
                BUFO(S.outlier_val_slab, wg->outlier_offset * 4, 9);
            } else { BUF(S.packed_slab, 8); BUF(S.packed_slab, 9); }
            CB(S.cb_oc[l*7+4], 10);
            if (wu->outlier_count > 0 && S.outlier_idx_slab) {
                BUFO(S.outlier_idx_slab, wu->outlier_offset * 4, 11);
                BUFO(S.outlier_val_slab, wu->outlier_offset * 4, 12);
            } else { BUF(S.packed_slab, 11); BUF(S.packed_slab, 12); }
            CB(S.cb_oc[l*7+5], 13);
            DTG((ffnDim + 3) / 4, 256); BAR();
        } else {
            sq4mv(enc, S.normed2, &S.wgate[l], S.gatePre, S.cb_dim, S.cb_Nffn, l*7+4);
            sq4mv(enc, S.normed2, &S.wup[l], S.upOut, S.cb_dim, S.cb_Nffn, l*7+5);
            BAR();
            ENC(S.ps_silu_gate_mul);
            BUF(S.gatePre, 0); BUF(S.upOut, 1); BUF(S.ffnMid, 2); DT(ffnDim, 256); BAR();
        }
        sq4mv(enc, S.ffnMid, &S.wdown[l], S.proj, S.cb_ffnDim, S.cb_Ndim, l*7+6); BAR();
        ENC(S.ps_add_inplace); BUF(S.hidden, 0); BUF(S.proj, 1); DT(dim, 256); BAR();
    }

    ENC(S.ps_rmsnorm_save);
    BUF(S.hidden, 0); BUF(S.finalNorm, 1); BUF(S.normed, 2);
    CB(S.cb_dim, 3); CB(S.cb_eps, 4); DTG(1, tpg); BAR();
    sq4mv(enc, S.hidden, &S.lmHead, S.logits, S.cb_dim, S.cb_Nvocab, 7*nLayers);
    BAR();

    // GPU-side argmax
    ENC(S.ps_argmax);
    BUF(S.logits, 0);
    [enc setBuffer:B(S.argmax_result) offset:0 atIndex:1];
    CB(S.cb_Nvocab, 2);
    DTG(1, 256);

    [enc endEncoding];
    clock_gettime(CLOCK_MONOTONIC, &t1);
    [cmd commit];
    [cmd waitUntilCompleted];
    clock_gettime(CLOCK_MONOTONIC, &t2);

    double encode_ms = (t1.tv_sec - t0.tv_sec)*1000.0 + (t1.tv_nsec - t0.tv_nsec)/1e6;
    double wait_ms = (t2.tv_sec - t1.tv_sec)*1000.0 + (t2.tv_nsec - t1.tv_nsec)/1e6;
    double total_ms = (t2.tv_sec - t0.tv_sec)*1000.0 + (t2.tv_nsec - t0.tv_nsec)/1e6;
    ss_total += total_ms; ss_cpu += encode_ms; ss_gpu += wait_ms;
    ss_count++;
    if (ss_count == 10 || ss_count == 50 || ss_count == 200) {
        fprintf(stderr, "[SQ4-PERF] step %d: total=%.2fms (encode=%.2f gpu=%.2f) → %.1f tok/s\n",
            ss_count, ss_total/ss_count, ss_cpu/ss_count, ss_gpu/ss_count,
            1000.0*ss_count/ss_total);
    }

    return ((uint32_t*)B(S.argmax_result).contents)[0];
}

// Prefill: run tokens one at a time (no batched GEMM in clean rewrite)
int mtl_sq4_infer_prefill(const int* tokenIDs, int nTokens, float* logitsOut) {
    if (!S.built || nTokens <= 0) return -1;
    float* tmpLogits = (float*)malloc(S.vocabSize * 4);
    for (int i = 0; i < nTokens; i++) {
        mtl_sq4_infer_step(tokenIDs[i], i, (i == nTokens - 1) ? logitsOut : tmpLogits);
    }
    free(tmpLogits);
    return 0;
}

void mtl_sq4_infer_reset_kv(void) {
    if (!S.built) return;
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    for (int l = 0; l < S.nLayers; l++) {
        [blit fillBuffer:B(S.kCache[l]) range:NSMakeRange(0, S.maxSeq * S.kvDim * 4) value:0];
        [blit fillBuffer:B(S.vCache[l]) range:NSMakeRange(0, S.maxSeq * S.kvDim * 4) value:0];
    }
    [blit endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}
