// mongoose_sq4_matvec.cu — Generic SQ4 dequant-matvec kernel.
//
// SQ4 format: 4 bits per weight = [sign:1][band:3].
// Band table: 8 float32 values per row (the reconstruction values).
// Dequant: val = bands[nibble & 7], sign = (nibble >> 3) ? -1 : 1.
// Outliers stored separately and applied after the main matvec.
//
// One block per row, threads cooperate on the dot product.

// out[row] = sum_k(dequant(packed[row,k]) * act[k])
// packed: [rows, cols/2] bytes (2 weights per byte)
// bands: [rows, 8] floats (8 band values per row)
// act: [cols] floats (activation vector)
// out: [rows] floats
__global__ void sq4_matvec_kernel(
    const uint8_t* __restrict__ packed,
    const float* __restrict__ bands,
    const float* __restrict__ act,
    float* __restrict__ out,
    int cols
) {
    int row = blockIdx.x;
    const float* row_bands = bands + row * 8;
    int half_cols = cols / 2;
    const uint8_t* row_packed = packed + row * half_cols;

    float sum = 0.0f;
    for (int k = threadIdx.x; k < cols; k += blockDim.x) {
        int byte_idx = k >> 1;
        int shift = (k & 1) * 4;
        int nibble = (row_packed[byte_idx] >> shift) & 0x0F;
        int band = nibble & 0x07;
        float w = row_bands[band];
        if (nibble & 0x08) w = -w;
        sum += w * act[k];
    }

    // Warp reduce
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    // Block reduce
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

void mongoose_sq4_matvec(
    const void* packed, const float* bands, const float* act,
    float* out, int rows, int cols, cudaStream_t stream
) {
    int threads = (cols < 1024) ? ((cols + 31) / 32) * 32 : 1024;
    if (threads < 32) threads = 32;
    sq4_matvec_kernel<<<rows, threads, 0, stream>>>(
        (const uint8_t*)packed, bands, act, out, cols);
}

int mongoose_has_sq4_matvec() { return 1; }

// Apply outlier corrections after matvec.
// For each outlier: out[row] += (outlier_val - band_approx) * act[col]
// outlier_idx: flat indices into [rows, cols] matrix
// outlier_val: fp32 values (pre-converted from fp16)
// This kernel runs with outlierCount blocks, 1 thread each (outliers are sparse).
__global__ void sq4_outlier_correct_kernel(
    const uint32_t* __restrict__ outlier_idx,
    const float* __restrict__ outlier_val,
    const uint8_t* __restrict__ packed,
    const float* __restrict__ bands,
    const float* __restrict__ act,
    float* __restrict__ out,
    int cols
) {
    int i = blockIdx.x;
    uint32_t flat = outlier_idx[i];
    int row = flat / cols;
    int col = flat % cols;

    // Dequant the band approximation
    int half_cols = cols / 2;
    int byte_idx = row * half_cols + (col >> 1);
    int shift = (col & 1) * 4;
    int nibble = (packed[byte_idx] >> shift) & 0x0F;
    float band_approx = bands[row * 8 + (nibble & 0x07)];
    if (nibble & 0x08) band_approx = -band_approx;

    float correction = (outlier_val[i] - band_approx) * act[col];
    atomicAdd(&out[row], correction);
}

void mongoose_sq4_outlier_correct(
    const uint32_t* outlier_idx, const float* outlier_val,
    const void* packed, const float* bands, const float* act,
    float* out, int outlier_count, int cols, cudaStream_t stream
) {
    if (outlier_count <= 0) return;
    sq4_outlier_correct_kernel<<<outlier_count, 1, 0, stream>>>(
        outlier_idx, outlier_val, (const uint8_t*)packed, bands, act, out, cols);
}
