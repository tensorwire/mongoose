// mongoose_quant.cu — quant kernels
// Auto-extracted from mongoose.cu. Do not edit directly.

// === INT8 Dequantization ===
// Dequantize INT8 weights to FP16 for cuBLAS mixed-precision matmul.
// Each row has a per-row absmax scale: fp16_out = int8_val * (scale / 127.0)
// This is the QLoRA/bitsandbytes pattern — store compressed, dequant on-the-fly.
// Memory-bandwidth limited, nearly free compared to the matmul itself.
//
// data_int8: [rows, cols] INT8 weights
// scales:    [rows] FP32 per-row absmax
// out_fp16:  [rows, cols] FP16 output
__global__ void dequant_int8_to_fp16_kernel(
    const int8_t* data_int8, const float* scales, __half* out_fp16,
    int rows, int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    float s = scales[row] / 127.0f;
    const int8_t* src = data_int8 + row * cols;
    __half* dst = out_fp16 + row * cols;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        dst[j] = __float2half((float)src[j] * s);
    }
}

void mongoose_dequant_int8_to_fp16(
    const void* data_int8, const float* scales, void* out_fp16,
    int rows, int cols, cudaStream_t stream
) {
    int threads = cols < 256 ? cols : 256;
    dequant_int8_to_fp16_kernel<<<rows, threads, 0, stream>>>(
        (const int8_t*)data_int8, scales, (__half*)out_fp16, rows, cols);
}

// === INT8 Dequantization to FP32 ===
// Same as above but outputs FP32. For cases where FP16 cuBLAS path isn't available
// or when we need FP32 precision (norms, biases).
__global__ void dequant_int8_to_fp32_kernel(
    const int8_t* data_int8, const float* scales, float* out_fp32,
    int rows, int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    float s = scales[row] / 127.0f;
    const int8_t* src = data_int8 + row * cols;
    float* dst = out_fp32 + row * cols;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        dst[j] = (float)src[j] * s;
    }
}

void mongoose_dequant_int8_to_fp32(
    const void* data_int8, const float* scales, float* out_fp32,
    int rows, int cols, cudaStream_t stream
) {
    int threads = cols < 256 ? cols : 256;
    dequant_int8_to_fp32_kernel<<<rows, threads, 0, stream>>>(
        (const int8_t*)data_int8, scales, out_fp32, rows, cols);
}

// === Requant FP32 back to INT8: per-row absmax scaling ===
// One block per row. Threads find absmax, compute scale, quantize.

__global__ void requant_fp32_to_int8_kernel(
    const float* fp32, int8_t* data_int8, float* scales,
    int rows, int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* src = fp32 + row * cols;
    int8_t* dst = data_int8 + row * cols;

    __shared__ float shared_max[32];
    float local_max = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float v = fabsf(src[j]);
        if (v > local_max) local_max = v;
    }
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    if (lane == 0) shared_max[wid] = local_max;
    __syncthreads();
    if (threadIdx.x == 0) {
        int nWarps = (blockDim.x + warpSize - 1) / warpSize;
        float mx = 0.0f;
        for (int i = 0; i < nWarps; i++) mx = fmaxf(mx, shared_max[i]);
        shared_max[0] = mx;
    }
    __syncthreads();

    float absmax = shared_max[0];
    scales[row] = absmax;
    if (absmax == 0.0f) {
        for (int j = threadIdx.x; j < cols; j += blockDim.x)
            dst[j] = 0;
        return;
    }
    float inv_scale = 127.0f / absmax;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float q = src[j] * inv_scale;
        q = fminf(fmaxf(q, -127.0f), 127.0f);
        dst[j] = (int8_t)rintf(q);
    }
}

void mongoose_requant_fp32_to_int8(
    const float* fp32, void* data_int8, float* scales,
    int rows, int cols, cudaStream_t stream
) {
    int threads = cols < 256 ? cols : 256;
    requant_fp32_to_int8_kernel<<<rows, threads, 0, stream>>>(
        fp32, (int8_t*)data_int8, scales, rows, cols);
}

// Dequant INT8 + FP32 delta → FP32 output.
__global__ void dequant_int8_delta_kernel(
    const int8_t* __restrict__ data, const float* __restrict__ scales,
    const float* __restrict__ delta, float* __restrict__ out, int n, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float scale = scales[i / cols] / 127.0f;
    out[i] = (float)data[i] * scale + delta[i];
}

void mongoose_dequant_int8_delta(
    const void* data, const float* scales, const float* delta,
    float* out, int n, int cols, cudaStream_t stream) {
    dequant_int8_delta_kernel<<<(n+255)/256, 256, 0, stream>>>(
        (const int8_t*)data, scales, delta, out, n, cols);
}

