// SQ4 prefill GEMM using Metal 4 matmul2d TensorOp.
// C[N,M] = W_sq4[N,K] @ A[K,M] where W is SQ4 nibble-packed.
// Dequant BK-wide weight tiles into threadgroup FP16, then matmul2d.
// Dequant cost amortized across M tokens.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr int BM = 64;
constant constexpr int BN = 32;
constant constexpr int BK = 32;

// Dequant a BN×BK tile of SQ4 weights into FP32 threadgroup memory.
// packed: nibble-packed weights [N, K/2] bytes, row-major.
// bands: 8 float32 band means.
// tile_out: [BK, BN] FP16 in threadgroup memory (column-major for matmul2d B^T).
// row_start: first row (N dimension) of this tile.
// k_start: first column (K dimension) of this tile.
static void sq4_dequant_tile(
    device const uchar* packed,
    device const float* bands,
    threadgroup float* tile_out,
    uint row_start, uint k_start,
    uint N, uint K,
    uint tid, uint tpg)
{
    // Build 16-entry LUT in registers from bands
    // (bands is only 8 floats, read via device — small enough for broadcast)
    float b[8];
    for (int i = 0; i < 8; i++) b[i] = bands[i];

    uint half_K = K / 2;
    uint total = BN * BK;

    for (uint idx = tid; idx < total; idx += tpg) {
        uint local_n = idx / BK;  // row within tile
        uint local_k = idx % BK;  // col within tile
        uint global_n = row_start + local_n;
        uint global_k = k_start + local_k;

        float val = 0.0f;
        if (global_n < N && global_k < K) {
            uint flat = global_n * half_K + (global_k >> 1);
            uint shift = (global_k & 1) * 4;
            uint nibble = (packed[flat] >> shift) & 0x0F;
            uint band = nibble & 0x07;
            val = b[band];
            if (nibble & 0x08) val = -val;
        }
        // Row-major: BN rows of BK elements (stride-1 = BK, matching dextents(BK, BN))
        tile_out[local_n * BK + local_k] = val;
    }
}

// C[N,M] = W_sq4[N,K] @ A[K,M]
// W is nibble-packed, A is FP32, C is FP32.
// Grid: (ceil(N/BN), ceil(M/BM), 1)
kernel void sq4_prefill_gemm(
    device const float* A_buf    [[buffer(0)]],   // activations [K, M] column-major
    device const uchar* W_packed [[buffer(1)]],   // SQ4 weights [N, K/2] row-major
    device const float* bands    [[buffer(2)]],   // 8 band means
    device float* C_buf          [[buffer(3)]],   // output [N, M] column-major
    constant uint& M             [[buffer(4)]],
    constant uint& K             [[buffer(5)]],
    constant uint& N             [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid_2d  [[thread_position_in_threadgroup]],
    uint2 tpg_2d  [[threads_per_threadgroup]])
{
    uint tid = tid_2d.x;
    uint tpg = tpg_2d.x;

    // A is [M, K] row-major: M rows of K floats. Extent = (K, M).
    auto A = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        const_cast<device float*>(A_buf), dextents<int32_t, 2>(K, M));

    // C is [M, N] row-major: M rows of N floats. Extent = (N, M).
    auto C = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        C_buf, dextents<int32_t, 2>(N, M));

    // Slice A and C for this threadgroup's tile
    auto mA = A.slice(0, tgid.y * BM);   // rows tgid.y*BM..+BM of A
    auto mC = C.slice(tgid.x * BN, tgid.y * BM);  // output tile

    // C = A @ W^T: A not transposed, W (B) transposed, C not transposed
    // Fixed BK since we manually tile K and dequant BK weights per iteration
    constexpr auto desc = matmul2d_descriptor(
        BM, BN, BK,
        false, true, false);
    matmul2d<desc, execution_simdgroups<4>> matmulOp;

    using tileW_type = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>;
    auto acc = matmulOp.get_destination_cooperative_tensor<
        decltype(mA), tileW_type, float>();

    // Zero accumulator
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i))
            acc[i] = 0;
    }

    // Threadgroup staging for dequanted weight tile [BN, BK] in FP32
    threadgroup float tg_W[BK * BN];

    uint n_start = tgid.x * BN;

    // Tiled K loop: dequant BK columns of W at a time, multiply with A
    for (uint k = 0; k < K; k += BK) {
        // Dequant W[n_start..n_start+BN, k..k+BK] into threadgroup memory
        sq4_dequant_tile(W_packed, bands, tg_W, n_start, k, N, K, tid, tpg);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Create tensor view of the dequanted tile
        auto tileW = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
            tg_W, dextents<int32_t, 2>(BK, BN));  // extent = (K_dim, N_dim) matching B layout

        // Slice A columns k..k+BK
        auto tileA = mA.slice(k, 0);  // offset in K dimension

        // Accumulate: acc += tileA @ tileW^T
        matmulOp.run(tileA, tileW, acc);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store result
    acc.store(mC);
}
