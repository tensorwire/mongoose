// mongoose_q8_matvec.cu — q8_matvec kernels
// Auto-extracted from mongoose.cu. Do not edit directly.

// === Fused Q8 matvec: out[row] = sum_k(act[k] * int8_weight[row,k] * scale[row]/127) ===
// One block per output row. Threads cooperatively reduce the dot product.
__global__ void q8_matvec_kernel(
    const float* act, const int8_t* weight, const float* scales,
    float* out, int K
) {
    int row = blockIdx.x;
    float scale = scales[row] / 127.0f;
    const int8_t* wRow = weight + row * K;

    float sum = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        sum += act[k] * float(wRow[k]) * scale;
    }

    // Warp reduce
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    if (lane == 0) shared[wid] = sum;
    __syncthreads();

    if (threadIdx.x == 0) {
        int nWarps = (blockDim.x + warpSize - 1) / warpSize;
        float total = 0.0f;
        for (int i = 0; i < nWarps; i++) total += shared[i];
        out[row] = total;
    }
}

void mongoose_q8_matvec(
    const float* act, const void* weight_int8, const float* scales,
    float* out, int N, int K, cudaStream_t stream
) {
    int threads = K < 1024 ? ((K + 31) / 32) * 32 : 1024;
    if (threads < 32) threads = 32;
    q8_matvec_kernel<<<N, threads, 0, stream>>>(
        act, (const int8_t*)weight_int8, scales, out, K);
}

// === Fused Q4 matvec: out[row] = sum_k(act[k] * dequant4(packed[row,k/2])) ===
__global__ void q4_matvec_kernel(
    const float* act, const uint8_t* weight, const float* scales,
    float* out, int K
) {
    int row = blockIdx.x;
    float scale = scales[row] / 7.0f;
    int halfK = K / 2;
    const uint8_t* wRow = weight + row * halfK;

    float sum = 0.0f;
    for (int k = threadIdx.x; k < halfK; k += blockDim.x) {
        uint8_t packed = wRow[k];
        float w0 = float(int(packed & 0xF) - 8) * scale;
        float w1 = float(int(packed >> 4) - 8) * scale;
        sum += act[k * 2] * w0 + act[k * 2 + 1] * w1;
    }

    for (int offset = warpSize/2; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    if (lane == 0) shared[wid] = sum;
    __syncthreads();

    if (threadIdx.x == 0) {
        int nWarps = (blockDim.x + warpSize - 1) / warpSize;
        float total = 0.0f;
        for (int i = 0; i < nWarps; i++) total += shared[i];
        out[row] = total;
    }
}

void mongoose_q4_matvec(
    const float* act, const void* weight_packed, const float* scales,
    float* out, int N, int K, cudaStream_t stream
) {
    int halfK = K / 2;
    int threads = halfK < 1024 ? ((halfK + 31) / 32) * 32 : 1024;
    if (threads < 32) threads = 32;
    q4_matvec_kernel<<<N, threads, 0, stream>>>(
        act, (const uint8_t*)weight_packed, scales, out, K);
}

