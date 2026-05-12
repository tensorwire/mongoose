#include <metal_stdlib>
using namespace metal;

// SQ4 dequant-matvec: out[row] = sum_k(dequant(packed[row,k]) * act[k])
//
// 4 rows per threadgroup, 2 simdgroups per row (64 lanes per row).
// Vectorized: uchar4 = 4 bytes = 8 nibbles per load.
// Dequant via 16-entry signed band table (zero branching).
// Matches Q8 matvec dispatch pattern for maximum throughput.

kernel void sq4_matvec(
    device const float* act      [[buffer(0)]],
    device const uchar* packed   [[buffer(1)]],
    device const float* bands    [[buffer(2)]],
    device float* out            [[buffer(3)]],
    constant uint& K             [[buffer(4)]],
    constant uint& N             [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    // 16-entry signed band table: direct nibble → weight value (one lookup, zero branching)
    threadgroup float table16[16];
    if (sgitg == 0 && tiisg < 8) {
        table16[tiisg] = bands[tiisg];
        table16[tiisg + 8] = -bands[tiisg];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = tgid * 4 + sgitg / 2;
    if (row >= N) return;

    ushort half_sg = sgitg % 2;
    uint tid_in_row = half_sg * 32 + tiisg;
    uint half_K = K / 2;

    device const uchar4* row4 = (device const uchar4*)(packed + row * half_K);
    uint n4 = half_K / 4;

    device const float4* act4 = (device const float4*)act;

    float sum = 0.0f;
    for (uint i = tid_in_row; i < n4; i += 64) {
        uchar4 chunk = row4[i];
        uint col_base = i * 2;

        float4 a0 = act4[col_base];
        float4 a1 = act4[col_base + 1];

        #define SQ4_DEQUANT(nib) table16[(nib)]
        sum += SQ4_DEQUANT(chunk.x & 0x0F) * a0.x;
        sum += SQ4_DEQUANT((chunk.x >> 4) & 0x0F) * a0.y;
        sum += SQ4_DEQUANT(chunk.y & 0x0F) * a0.z;
        sum += SQ4_DEQUANT((chunk.y >> 4) & 0x0F) * a0.w;
        sum += SQ4_DEQUANT(chunk.z & 0x0F) * a1.x;
        sum += SQ4_DEQUANT((chunk.z >> 4) & 0x0F) * a1.y;
        sum += SQ4_DEQUANT(chunk.w & 0x0F) * a1.z;
        sum += SQ4_DEQUANT((chunk.w >> 4) & 0x0F) * a1.w;
        #undef SQ4_DEQUANT
    }

    uint handled = n4 * 8;
    for (uint k = handled + tid_in_row; k < K; k += 64) {
        uint byte_idx = k >> 1;
        uint shift = (k & 1) * 4;
        uchar nibble = (packed[row * half_K + byte_idx] >> shift) & 0x0F;
        float w = table16[nibble];
        sum += w * act[k];
    }

    sum = simd_sum(sum);

    threadgroup float shmem[8];
    if (tiisg == 0) shmem[sgitg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg % 2 == 0 && tiisg == 0) {
        out[row] = shmem[sgitg] + shmem[sgitg + 1];
    }
}

// Outlier correction: out[row] += (exact - band_approx) * act[col]
kernel void sq4_outlier_correct(
    device const uint* outlier_idx   [[buffer(0)]],
    device const float* outlier_val  [[buffer(1)]],
    device const uchar* packed       [[buffer(2)]],
    device const float* bands        [[buffer(3)]],
    device const float* act          [[buffer(4)]],
    device atomic_float* out         [[buffer(5)]],
    constant uint& cols              [[buffer(6)]],
    constant uint& outlier_count     [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= outlier_count) return;
    uint flat = outlier_idx[tid];
    uint row = flat / cols;
    uint col = flat % cols;

    uint half_cols = cols >> 1;
    uint byte_idx = row * half_cols + (col >> 1);
    uint shift = (col & 1) * 4;
    uchar nibble = (packed[byte_idx] >> shift) & 0x0F;
    float band_approx = bands[nibble & 0x07];
    if (nibble & 0x08) band_approx = -band_approx;

    float correction = (outlier_val[tid] - band_approx) * act[col];
    atomic_fetch_add_explicit(out + row, correction, memory_order_relaxed);
}
