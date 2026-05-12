#include <metal_stdlib>
using namespace metal;

// SQ4 decode matvec: 4 rows/tg, 2 simdgroups/row, uchar2 reads, 16-entry tg LUT.
// Fused outlier correction. This is the M=1 decode path.

kernel void sq4_matvec(
    device const float* act       [[buffer(0)]],
    device const uchar* packed    [[buffer(1)]],
    device const float* bands     [[buffer(2)]],
    device float* out             [[buffer(3)]],
    constant uint& K              [[buffer(4)]],
    constant uint& N              [[buffer(5)]],
    device const uint* outlier_idx  [[buffer(6)]],
    device const float* outlier_val [[buffer(7)]],
    constant uint& outlier_count    [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
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

    device const uint4* row_packed4 = (device const uint4*)(packed + row * half_K);
    uint n_uint4 = half_K >> 4;

    float sum = 0.0f;
    for (uint i = tid_in_row; i < n_uint4; i += 64) {
        uint4 chunk = row_packed4[i];
        uint base = i * 32;

        { uint w = chunk.x; uint cb = base;
          float4 a0 = *((device const float4*)(act + cb));
          float4 a1 = *((device const float4*)(act + cb + 4));
          sum += table16[(w>> 0)&0xF]*a0.x + table16[(w>> 4)&0xF]*a0.y
               + table16[(w>> 8)&0xF]*a0.z + table16[(w>>12)&0xF]*a0.w
               + table16[(w>>16)&0xF]*a1.x + table16[(w>>20)&0xF]*a1.y
               + table16[(w>>24)&0xF]*a1.z + table16[(w>>28)&0xF]*a1.w; }
        { uint w = chunk.y; uint cb = base + 8;
          float4 a0 = *((device const float4*)(act + cb));
          float4 a1 = *((device const float4*)(act + cb + 4));
          sum += table16[(w>> 0)&0xF]*a0.x + table16[(w>> 4)&0xF]*a0.y
               + table16[(w>> 8)&0xF]*a0.z + table16[(w>>12)&0xF]*a0.w
               + table16[(w>>16)&0xF]*a1.x + table16[(w>>20)&0xF]*a1.y
               + table16[(w>>24)&0xF]*a1.z + table16[(w>>28)&0xF]*a1.w; }
        { uint w = chunk.z; uint cb = base + 16;
          float4 a0 = *((device const float4*)(act + cb));
          float4 a1 = *((device const float4*)(act + cb + 4));
          sum += table16[(w>> 0)&0xF]*a0.x + table16[(w>> 4)&0xF]*a0.y
               + table16[(w>> 8)&0xF]*a0.z + table16[(w>>12)&0xF]*a0.w
               + table16[(w>>16)&0xF]*a1.x + table16[(w>>20)&0xF]*a1.y
               + table16[(w>>24)&0xF]*a1.z + table16[(w>>28)&0xF]*a1.w; }
        { uint w = chunk.w; uint cb = base + 24;
          float4 a0 = *((device const float4*)(act + cb));
          float4 a1 = *((device const float4*)(act + cb + 4));
          sum += table16[(w>> 0)&0xF]*a0.x + table16[(w>> 4)&0xF]*a0.y
               + table16[(w>> 8)&0xF]*a0.z + table16[(w>>12)&0xF]*a0.w
               + table16[(w>>16)&0xF]*a1.x + table16[(w>>20)&0xF]*a1.y
               + table16[(w>>24)&0xF]*a1.z + table16[(w>>28)&0xF]*a1.w; }
    }

    uint handled = n_uint4 * 32;
    for (uint k = handled + tid_in_row; k < K; k += 64) {
        uint byte_idx = k >> 1;
        uint shift = (k & 1) * 4;
        uchar nibble = (packed[row * half_K + byte_idx] >> shift) & 0x0F;
        sum += table16[nibble] * act[k];
    }

    sum = simd_sum(sum);

    threadgroup float shmem[8];
    if (tiisg == 0) shmem[sgitg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg % 2 == 0 && tiisg == 0) {
        float total = shmem[sgitg] + shmem[sgitg + 1];

        if (outlier_count > 0) {
            uint target = row * K;
            uint row_end = target + K;
            uint lo = 0, hi = outlier_count;
            while (lo < hi) {
                uint mid = (lo + hi) >> 1;
                if (outlier_idx[mid] < target) lo = mid + 1;
                else hi = mid;
            }
            for (uint i = lo; i < outlier_count && outlier_idx[i] < row_end; i++) {
                uint flat = outlier_idx[i];
                uint col = flat - target;
                uint byte_idx = row * half_K + (col >> 1);
                uint shift = (col & 1) * 4;
                uchar nibble = (packed[byte_idx] >> shift) & 0x0F;
                total += (outlier_val[i] - table16[nibble]) * act[col];
            }
        }

        out[row] = total;
    }
}

// Standalone outlier correction (for testing)
kernel void sq4_outlier_correct(
    device const uint* outlier_idx    [[buffer(0)]],
    device const float* outlier_val   [[buffer(1)]],
    device const uchar* packed        [[buffer(2)]],
    device const float* bands         [[buffer(3)]],
    device const float* act           [[buffer(4)]],
    device atomic_float* out          [[buffer(5)]],
    constant uint& cols               [[buffer(6)]],
    constant uint& outlier_count      [[buffer(7)]],
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
