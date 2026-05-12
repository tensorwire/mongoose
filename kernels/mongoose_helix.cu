// mongoose_helix.cu — helix kernels
// Auto-extracted from mongoose.cu. Do not edit directly.

// === Helix DNA step — FP32 paired parameter update with rung coupling ===
__global__ void helix_dna_step_kernel(
    float* d1, float* d2,
    const float* g1, const float* g2,
    float* m1, float* m2,
    float* v1, float* v2,
    float lr, float beta1, float beta2, float bc1, float bc2,
    float eps, float wd,
    float backbone1, float glyco1, float hbond1,
    float hbond2, float glyco2, float backbone2,
    float bondStrength, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float ob1 = 1.0f - beta1, ob2 = 1.0f - beta2;

    float effGrad1 = g1[i] * glyco1 + g2[i] * hbond1 * bondStrength;
    float mi1 = beta1 * m1[i] + ob1 * effGrad1;
    float vi1 = beta2 * v1[i] + ob2 * effGrad1 * effGrad1;
    m1[i] = mi1; v1[i] = vi1;
    d1[i] -= lr * (mi1 / bc1 / (sqrtf(vi1 / bc2) + eps) + wd * backbone1 * d1[i]);

    float effGrad2 = g2[i] * glyco2 + g1[i] * hbond2 * bondStrength;
    float mi2 = beta1 * m2[i] + ob1 * effGrad2;
    float vi2 = beta2 * v2[i] + ob2 * effGrad2 * effGrad2;
    m2[i] = mi2; v2[i] = vi2;
    d2[i] -= lr * (mi2 / bc1 / (sqrtf(vi2 / bc2) + eps) + wd * backbone2 * d2[i]);
}

void mongoose_helix_dna_step(
    float* d1, float* d2, const float* g1, const float* g2,
    float* m1, float* m2, float* v1, float* v2,
    float lr, float beta1, float beta2, float bc1, float bc2,
    float eps, float wd,
    float backbone1, float glyco1, float hbond1,
    float hbond2, float glyco2, float backbone2,
    float bondStrength, int n, cudaStream_t stream
) {
    helix_dna_step_kernel<<<(n+255)/256, 256, 0, stream>>>(
        d1, d2, g1, g2, m1, m2, v1, v2,
        lr, beta1, beta2, bc1, bc2, eps, wd,
        backbone1, glyco1, hbond1, hbond2, glyco2, backbone2,
        bondStrength, n);
}

// === Needle: INT8 weight update with sub-quant delta accumulation ===
__global__ void helix_needle_kernel(
    int8_t* __restrict__ data_int8, float* __restrict__ scales,
    const float* __restrict__ grad,
    __half* __restrict__ mom, __half* __restrict__ vel,
    const float* __restrict__ rowMask,
    float lr, float beta1, float beta2, float bc1, float bc2,
    float eps, float wd, int n, int cols
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int row = i / cols;
    float maskVal = rowMask[row];
    if (maskVal == 0.0f) return;
    int ci = ((int)(maskVal - 1.0f)) * cols + (i % cols);

    float scale = scales[row] / 127.0f;
    float delta = __half2float(vel[ci]);
    float mi = __half2float(mom[ci]);
    float g = grad[i];
    float ob1 = 1.0f - beta1, ob2 = 1.0f - beta2;
    mi = beta1 * mi + ob1 * g;
    float mhat = mi / bc1;
    delta -= lr * (mhat / (sqrtf(ob2 * g * g / bc2) + eps) + wd * delta);

    float bucket = scale;
    if (delta > 0.5f * bucket || delta < -0.5f * bucket) {
        float w = (float)data_int8[i] * scale + delta;
        float qi = fminf(fmaxf(w / scale, -127.0f), 127.0f);
        float qr = rintf(qi);
        data_int8[i] = (int8_t)qr;
        delta = w - qr * scale;
    }
    mom[ci] = __float2half(mi);
    vel[ci] = __float2half(delta);
}

void mongoose_helix_needle(
    void* data_int8, float* scales, const float* grad,
    void* mom, void* vel, const void* mask,
    float lr, float beta1, float beta2, float bc1, float bc2,
    float eps, float wd, int n, int cols, cudaStream_t stream
) {
    helix_needle_kernel<<<(n+255)/256, 256, 0, stream>>>(
        (int8_t*)data_int8, scales, grad,
        (__half*)mom, (__half*)vel, (const float*)mask,
        lr, beta1, beta2, bc1, bc2, eps, wd, n, cols);
}

// === Needle paired: INT8 DNA-coupled update ===
__global__ void helix_needle_paired_kernel(
    int8_t* __restrict__ d1, int8_t* __restrict__ d2,
    float* __restrict__ s1, float* __restrict__ s2,
    const float* __restrict__ g1, const float* __restrict__ g2,
    __half* __restrict__ m1, __half* __restrict__ m2,
    __half* __restrict__ v1, __half* __restrict__ v2,
    const float* __restrict__ rowMask,
    float lr, float beta1, float beta2, float bc1, float bc2,
    float eps, float wd,
    float backbone1, float glyco1, float hbond1,
    float hbond2, float glyco2, float backbone2,
    float bondStrength, int n, int cols
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int row = i / cols;
    float maskVal = rowMask[row];
    if (maskVal == 0.0f) return;
    int ci = ((int)(maskVal - 1.0f)) * cols + (i % cols);

    float scale1 = s1[row] / 127.0f, scale2 = s2[row] / 127.0f;
    float delta1 = __half2float(v1[ci]), delta2 = __half2float(v2[ci]);
    float mi1 = __half2float(m1[ci]), mi2 = __half2float(m2[ci]);
    float ob1 = 1.0f - beta1, ob2 = 1.0f - beta2;

    float eg1 = g1[i] * glyco1 + g2[i] * hbond1 * bondStrength;
    mi1 = beta1 * mi1 + ob1 * eg1;
    delta1 -= lr * (mi1 / bc1 / (sqrtf(ob2 * eg1 * eg1 / bc2) + eps) + wd * backbone1 * delta1);

    float eg2 = g2[i] * glyco2 + g1[i] * hbond2 * bondStrength;
    mi2 = beta1 * mi2 + ob1 * eg2;
    delta2 -= lr * (mi2 / bc1 / (sqrtf(ob2 * eg2 * eg2 / bc2) + eps) + wd * backbone2 * delta2);

    float b1 = scale1;
    if (delta1 > 0.5f*b1 || delta1 < -0.5f*b1) {
        float w = (float)d1[i]*scale1 + delta1;
        float q = rintf(fminf(fmaxf(w/scale1,-127.f),127.f));
        d1[i] = (int8_t)q; delta1 = w - q*scale1;
    }
    float b2 = scale2;
    if (delta2 > 0.5f*b2 || delta2 < -0.5f*b2) {
        float w = (float)d2[i]*scale2 + delta2;
        float q = rintf(fminf(fmaxf(w/scale2,-127.f),127.f));
        d2[i] = (int8_t)q; delta2 = w - q*scale2;
    }
    m1[ci] = __float2half(mi1); v1[ci] = __float2half(delta1);
    m2[ci] = __float2half(mi2); v2[ci] = __float2half(delta2);
}

void mongoose_helix_needle_paired(
    void* d1, void* d2, float* s1, float* s2,
    const float* g1, const float* g2,
    void* m1, void* m2, void* v1, void* v2,
    const void* mask,
    float lr, float beta1, float beta2, float bc1, float bc2,
    float eps, float wd,
    float backbone1, float glyco1, float hbond1,
    float hbond2, float glyco2, float backbone2,
    float bondStrength, int n, int cols, cudaStream_t stream
) {
    helix_needle_paired_kernel<<<(n+255)/256, 256, 0, stream>>>(
        (int8_t*)d1, (int8_t*)d2, s1, s2, g1, g2,
        (__half*)m1, (__half*)m2, (__half*)v1, (__half*)v2,
        (const float*)mask,
        lr, beta1, beta2, bc1, bc2, eps, wd,
        backbone1, glyco1, hbond1, hbond2, glyco2, backbone2,
        bondStrength, n, cols);
}

// === Sparse needle: per-weight update on conductor hot positions ===
// hotIdx[i] = flat position index into data_int8 / fp32_cache / grad.
// nHot threads — one per individual weight position.
// mom/delta compacted to [nHot], indexed by thread ID.
// grad is the full-size gradient buffer — read at hotIdx[tid] for real per-weight gradient.
// If grad is NULL, falls back to signalScale * momentum (forward-only mode).

__global__ void helix_needle_sparse_kernel(
    int8_t* __restrict__ data_int8,
    float* __restrict__ scales,
    float* __restrict__ fp32_cache,
    __half* __restrict__ mom,
    __half* __restrict__ delta_buf,
    const int* __restrict__ hotIdx,
    const float* __restrict__ grad,
    float signalScale,
    float lr, float beta1,
    float wd, int nHot, int cols
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nHot) return;

    int wi = hotIdx[tid];
    int row = wi / cols;

    float scale = scales[row] / 127.0f;
    float mi = __half2float(mom[tid]);
    float delta = __half2float(delta_buf[tid]);

    float g = (grad != NULL) ? grad[wi] : mi * signalScale;
    mi = beta1 * mi + (1.0f - beta1) * g;
    delta -= lr * (mi + wd * delta);

    if (fp32_cache != NULL) {
        float bucket = scale;
        if (delta > 0.5f * bucket || delta < -0.5f * bucket) {
            float w = (float)data_int8[wi] * scale + delta;
            float qi = w / scale;
            qi = fminf(fmaxf(qi, -127.0f), 127.0f);
            float qi_r = rintf(qi);
            data_int8[wi] = (int8_t)qi_r;
            delta = w - qi_r * scale;
            fp32_cache[wi] = qi_r * scale + delta;
        } else {
            fp32_cache[wi] = (float)data_int8[wi] * scale + delta;
        }
    }

    mom[tid] = __float2half(mi);
    delta_buf[tid] = __float2half(delta);
}

void mongoose_helix_needle_sparse(
    void* data_int8, float* scales, float* fp32_cache,
    void* mom, void* delta_buf, const int* hotIdx, const float* grad,
    float signalScale, float lr, float beta1,
    float wd, int nHot, int cols, cudaStream_t stream
) {
    if (nHot <= 0) return;
    int threads = 256;
    int blocks = (nHot + threads - 1) / threads;
    helix_needle_sparse_kernel<<<blocks, threads, 0, stream>>>(
        (int8_t*)data_int8, scales, fp32_cache,
        (__half*)mom, (__half*)delta_buf, hotIdx, grad,
        signalScale, lr, beta1, wd, nHot, cols);
}

// === Inline needle: forward-only, updates FP32 cache before matmul ===
// Dispatches over ALL elements (n = rows*cols), skips frozen rows via mask.
// rowMask[row] = 0: frozen (skip). rowMask[row] > 0: compact_row + 1.
// Momentum is compacted to [nHotRows * cols], indexed by compact row from mask.
// FP32 cache updated in-place — the following matmul sees corrected weights.

__global__ void helix_needle_inline_kernel(
    int8_t* __restrict__ data_int8,
    float* __restrict__ scales,
    float* __restrict__ fp32_cache,
    __half* __restrict__ mom,
    __half* __restrict__ delta_buf,
    const float* __restrict__ rowMask,
    float signalScale,
    float lr, float beta1,
    float wd, int n, int cols
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int row = i / cols;
    int col = i % cols;
    float maskVal = rowMask[row];
    if (maskVal == 0.0f) return;
    int compactRow = (int)(maskVal - 1.0f);
    int ci = compactRow * cols + col;

    float scale = scales[row] / 127.0f;
    float mi = __half2float(mom[ci]);
    float delta = __half2float(delta_buf[ci]);

    float sg = mi * signalScale;
    mi = beta1 * mi + (1.0f - beta1) * sg;
    delta -= lr * (mi + wd * delta);

    float bucket = scale;
    if (delta > 0.5f * bucket || delta < -0.5f * bucket) {
        float w = (float)data_int8[i] * scale + delta;
        float qi = w / scale;
        qi = fminf(fmaxf(qi, -127.0f), 127.0f);
        float qi_r = rintf(qi);
        data_int8[i] = (int8_t)qi_r;
        delta = w - qi_r * scale;
        fp32_cache[i] = qi_r * scale + delta;
    } else {
        fp32_cache[i] = (float)data_int8[i] * scale + delta;
    }

    mom[ci] = __float2half(mi);
    delta_buf[ci] = __float2half(delta);
}

void mongoose_helix_needle_inline(
    void* data_int8, float* scales, float* fp32_cache,
    void* mom, void* delta_buf, const void* mask,
    float signalScale, float lr, float beta1,
    float wd, int n, int cols, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    helix_needle_inline_kernel<<<blocks, threads, 0, stream>>>(
        (int8_t*)data_int8, scales, fp32_cache,
        (__half*)mom, (__half*)delta_buf, (const float*)mask,
        signalScale, lr, beta1, wd, n, cols);
}

