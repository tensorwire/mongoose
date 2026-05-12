// mongoose_sq4_matvec.cu — SQ4 dequant-matvec kernel (public, generic).
//
// Spec format on disk: separate magnitude (3-bit bitstream) + sign (1-bit packed).
// GPU format: pre-packed nibbles [sign:1|band:3], 2 per byte (packed at load time).
// 16-entry shared memory LUT. Vectorized uint4 reads (32 weights per load).
// Fused outlier correction (binary search per row, single kernel).

#include <cstdint>

extern "C" {

// Fused matvec + outlier correction. One block per row.
// 16-entry register LUT — avoids shared memory bank conflicts.
// Vectorized uint4 reads (32 weights per load) + float4 activations.
__global__ void sq4_matvec_kernel(
    const uint8_t* __restrict__ packed,
    const float* __restrict__ bands,
    const float* __restrict__ x,
    float* __restrict__ out,
    int rows, int cols,
    const uint32_t* __restrict__ outlier_idx,
    const float* __restrict__ outlier_val,
    int outlier_count
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    // Register-resident LUT: 16 floats (64 bytes). Avoids shared memory
    // bank conflicts from 1024 threads hitting 16 addresses simultaneously.
    float t[16];
    if (threadIdx.x < 8) {
        // Only need one thread to broadcast, but all threads build their own copy
    }
    // All threads build register LUT from global (read-only cache handles broadcast)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        t[i] = __ldg(&bands[i]);
        t[i + 8] = -t[i];
    }

    int row_bytes = cols >> 1;
    int n_uint4 = row_bytes >> 4;
    const uint4* row_packed4 = (const uint4*)(packed + (long long)row * row_bytes);

    float sum = 0.0f;

    for (int i = threadIdx.x; i < n_uint4; i += blockDim.x) {
        uint4 chunk = __ldg(&row_packed4[i]);
        int col_base = i * 32;

        #pragma unroll
        for (int q = 0; q < 4; q++) {
            uint32_t word = (&chunk.x)[q];
            int cb = col_base + q * 8;
            float4 xv0 = *((const float4*)(x + cb));
            float4 xv1 = *((const float4*)(x + cb + 4));

            sum += t[(word >>  0) & 0x0F] * xv0.x
                 + t[(word >>  4) & 0x0F] * xv0.y
                 + t[(word >>  8) & 0x0F] * xv0.z
                 + t[(word >> 12) & 0x0F] * xv0.w
                 + t[(word >> 16) & 0x0F] * xv1.x
                 + t[(word >> 20) & 0x0F] * xv1.y
                 + t[(word >> 24) & 0x0F] * xv1.z
                 + t[(word >> 28) & 0x0F] * xv1.w;
        }
    }

    // Tail: weights not covered by uint4
    int handled = n_uint4 * 32;
    for (int col = handled + threadIdx.x; col < cols; col += blockDim.x) {
        int byte_idx = col >> 1;
        int shift = (col & 1) * 4;
        int nibble = (packed[row * row_bytes + byte_idx] >> shift) & 0x0F;
        sum += t[nibble] * x[col];
    }

    // Two-stage reduction: warp shuffle then shared memory
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    __shared__ float warp_sums[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[wid] = sum;
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = 0;
        int nwarps = (blockDim.x + warpSize - 1) / warpSize;
        for (int i = 0; i < nwarps; i++) total += warp_sums[i];

        // Fused outlier correction — single thread scans row's outliers
        if (outlier_count > 0) {
            uint32_t target = (uint32_t)row * (uint32_t)cols;
            uint32_t row_end = target + (uint32_t)cols;
            int lo = 0, hi = outlier_count;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (outlier_idx[mid] < target) lo = mid + 1;
                else hi = mid;
            }
            for (int i = lo; i < outlier_count && outlier_idx[i] < row_end; i++) {
                uint32_t flat = outlier_idx[i];
                int col = flat - target;
                int byte_idx = row * row_bytes + (col >> 1);
                int shift = (col & 1) * 4;
                int nibble = (packed[byte_idx] >> shift) & 0x0F;
                total += (outlier_val[i] - t[nibble]) * x[col];
            }
        }

        out[row] = total;
    }
}

void mongoose_sq4_matvec(
    const void* packed, const float* bands, const float* act,
    float* out, int rows, int cols, cudaStream_t stream
) {
    int threads = 256;
    sq4_matvec_kernel<<<rows, threads, 0, stream>>>(
        (const uint8_t*)packed, bands, act, out, rows, cols,
        NULL, NULL, 0);
}

// Fused matvec + outlier correction in one launch
void mongoose_sq4_matvec_fused(
    const void* packed, const float* bands, const float* act,
    float* out, int rows, int cols,
    const uint32_t* outlier_idx, const float* outlier_val,
    int outlier_count, cudaStream_t stream
) {
    int threads = 256;
    sq4_matvec_kernel<<<rows, threads, 0, stream>>>(
        (const uint8_t*)packed, bands, act, out, rows, cols,
        outlier_idx, outlier_val, outlier_count);
}

int mongoose_has_sq4_matvec() { return 1; }

// ============================================================================
// TF32 Tensor Core path — tile-swizzled data + inline PTX m16n8k8 MMA.
// Requires SM 80+ (Ampere, Ada, Blackwell). Falls back to scalar on older GPUs.
//
// Data layout: fragment-order swizzle. Each 16×8 tile is stored so that
// warp lane t reads nibbles t*4..t*4+3 as the PTX a-fragment (a0,a1,a2,a3).
// One coalesced 64B read per tile per warp. Zero shuffle overhead.
// ============================================================================

#define TC_M 16
#define TC_K 8

// CPU-side tile swizzle: row-major nibbles → fragment-order tiles.
// Call once at upload time. tile_major must be calloc'd to packed_bytes.
void sq4_swizzle_for_tiles(
    const uint8_t* row_major, uint8_t* tile_major,
    int rows, int cols
) {
    int tiles_per_row = cols / TC_K;
    int tile_rows = (rows + TC_M - 1) / TC_M;

    memset(tile_major, 0, (rows * cols + 1) / 2);

    for (int tr = 0; tr < tile_rows; tr++) {
        for (int tk = 0; tk < tiles_per_row; tk++) {
            int dst_offset = (tr * tiles_per_row + tk) * (TC_M * TC_K / 2);
            for (int r = 0; r < TC_M; r++) {
                for (int c = 0; c < TC_K; c++) {
                    int src_row = tr * TC_M + r;
                    int src_col = tk * TC_K + c;
                    if (src_row >= rows || src_col >= cols) continue;

                    int src_flat = src_row * cols + src_col;
                    int src_byte = src_flat / 2;
                    int src_shift = (src_flat & 1) * 4;
                    int nibble = (row_major[src_byte] >> src_shift) & 0xF;

                    int dst_idx = (r/2)*16 + (c/2)*4 + (r%2) + (c%2)*2;
                    int dst_byte = dst_offset + dst_idx / 2;
                    int dst_shift = (dst_idx & 1) * 4;
                    tile_major[dst_byte] = (tile_major[dst_byte] & ~(0xF << dst_shift))
                                         | (nibble << dst_shift);
                }
            }
        }
    }
}

// Inline PTX TF32 matrix multiply-accumulate: D = A × B + C.
// m16n8k8: A[16,8] × B[8,8] = D[16,8]. All f32.
__device__ __forceinline__ void ptx_mma_m16n8k8_tf32(
    float& d0, float& d1, float& d2, float& d3,
    float a0, float a1, float a2, float a3,
    float b0, float b1,
    float c0, float c1, float c2, float c3
) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "f"(a0), "f"(a1), "f"(a2), "f"(a3),
          "f"(b0), "f"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3)
    );
}

// TF32 matvec on tile-swizzled data. 1 warp (32 threads) per block.
// Grid-strides over TC_M=16 row tiles. Band LUT in shared memory.
__global__ void sq4_matvec_tf32_kernel(
    const uint8_t* __restrict__ packed_swizzled,
    const float* __restrict__ band_means,
    const float* __restrict__ x,
    float* __restrict__ out,
    int rows, int cols
) {
    int lane = threadIdx.x;
    int groupID = lane / 4;
    int tig = lane % 4;
    int tiles_per_row = cols / TC_K;

    __shared__ float s_table[16];
    if (lane < 8) {
        s_table[lane] = band_means[lane];
        s_table[lane + 8] = -band_means[lane];
    }
    __syncwarp();

    for (int row_start = blockIdx.x * TC_M; row_start < rows; row_start += gridDim.x * TC_M) {
        float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
        int tile_row = row_start / TC_M;

        for (int tk = 0; tk < tiles_per_row; tk++) {
            int k = tk * TC_K;

            float b0 = (k + tig*2 < cols) ? x[k + tig*2] : 0.0f;
            float b1 = (k + tig*2+1 < cols) ? x[k + tig*2+1] : 0.0f;

            int tile_offset = (tile_row * tiles_per_row + tk) * (TC_M * TC_K / 2);
            uint16_t pair = ((const uint16_t*)(packed_swizzled + tile_offset))[lane];

            float a0 = s_table[pair & 0xF];
            float a1 = s_table[(pair >> 4) & 0xF];
            float a2 = s_table[(pair >> 8) & 0xF];
            float a3 = s_table[(pair >> 12) & 0xF];

            ptx_mma_m16n8k8_tf32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
        }

        if (tig == 0) {
            int r0 = row_start + groupID*2;
            int r1 = r0 + 1;
            if (r0 < rows) out[r0] = d0;
            if (r1 < rows) out[r1] = d2;
        }
    }
}

// Outlier correction for swizzled data. Uses pre-computed band approximations.
__global__ void sq4_outlier_correct_swizzled_kernel(
    const float* __restrict__ outlier_band_approx,
    const float* __restrict__ outlier_val,
    const uint32_t* __restrict__ outlier_idx,
    const float* __restrict__ x,
    float* __restrict__ out,
    int outlier_count, int cols
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= outlier_count) return;
    uint32_t flat = outlier_idx[tid];
    int row = flat / cols;
    int col = flat % cols;
    float correction = (outlier_val[tid] - outlier_band_approx[tid]) * x[col];
    atomicAdd(&out[row], correction);
}

// Dispatch for swizzled (TF32) path with outlier correction.
void mongoose_sq4_matvec_swizzled(
    const void* packed_swizzled, const float* bands, const float* act,
    float* out, int rows, int cols,
    const float* outlier_band_approx, const float* outlier_val,
    const uint32_t* outlier_idx, int outlier_count,
    cudaStream_t stream
) {
    int tile_blocks = (rows + TC_M - 1) / TC_M;
    int blocks = tile_blocks > 680 ? tile_blocks : 680;
    sq4_matvec_tf32_kernel<<<blocks, 32, 0, stream>>>(
        (const uint8_t*)packed_swizzled, bands, act, out, rows, cols);
    if (outlier_count > 0 && outlier_band_approx) {
        int oBlocks = (outlier_count + 255) / 256;
        sq4_outlier_correct_swizzled_kernel<<<oBlocks, 256, 0, stream>>>(
            outlier_band_approx, outlier_val, outlier_idx,
            act, out, outlier_count, cols);
    }
}

// Standalone outlier correction (backward compat)
__global__ void sq4_outlier_correct_kernel(
    const uint32_t* __restrict__ outlier_idx,
    const float* __restrict__ outlier_val,
    const uint8_t* __restrict__ packed,
    const float* __restrict__ bands,
    const float* __restrict__ act,
    float* __restrict__ out,
    int cols
) {
    __shared__ float table[16];
    if (threadIdx.x < 8) {
        table[threadIdx.x] = bands[threadIdx.x];
        table[threadIdx.x + 8] = -bands[threadIdx.x];
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t flat = outlier_idx[i];
    int row = flat / cols;
    int col = flat % cols;

    int row_bytes = cols >> 1;
    int byte_idx = row * row_bytes + (col >> 1);
    int shift = (col & 1) * 4;
    int nibble = (packed[byte_idx] >> shift) & 0x0F;

    float correction = (outlier_val[i] - table[nibble]) * act[col];
    atomicAdd(&out[row], correction);
}

void mongoose_sq4_outlier_correct(
    const uint32_t* outlier_idx, const float* outlier_val,
    const void* packed, const float* bands, const float* act,
    float* out, int outlier_count, int cols, cudaStream_t stream
) {
    if (outlier_count <= 0) return;
    int threads = 256;
    int blocks = (outlier_count + threads - 1) / threads;
    sq4_outlier_correct_kernel<<<blocks, threads, 0, stream>>>(
        outlier_idx, outlier_val, (const uint8_t*)packed, bands, act, out, cols);
}

} // extern "C"
