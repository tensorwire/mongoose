#include <metal_stdlib>
using namespace metal;

// Q8 matvec: 4 rows per threadgroup, 256 threads (8 simdgroups).
// 2 simdgroups per row. Per-row scale, contiguous int8 weights.
// Vectorized char4 weight loads + float4 activation loads.
kernel void q8_matvec(
    device const float* act    [[buffer(0)]],
    device const char4* weight [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device float* out          [[buffer(3)]],
    device const uint* p_K     [[buffer(4)]],
    device const uint* p_N     [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    uint K = p_K[0], N = p_N[0];
    uint row = tgid * 4 + sgitg / 2;
    if (row >= N) return;
    ushort half_sg = sgitg % 2;
    uint K4 = K / 4;
    device const char4* wRow = weight + row * K4;
    device const float4* act4 = (device const float4*)act;
    float sum = 0.0f;
    uint tid_in_row = half_sg * 32 + tiisg;
    for (uint k4 = tid_in_row; k4 < K4; k4 += 64) {
        float4 a = act4[k4];
        char4 w = wRow[k4];
        sum += a.x * w.x + a.y * w.y + a.z * w.z + a.w * w.w;
    }
    sum *= scales[row];
    sum = simd_sum(sum);
    threadgroup float shmem[8];
    if (tiisg == 0) shmem[sgitg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg % 2 == 0 && tiisg == 0) {
        out[row] = shmem[sgitg] + shmem[sgitg + 1];
    }
}

// Q4_0 matvec: block_q4_0 format. 4 rows/tg, 2 simdgroups per row.
// qs[j] = elem[2j] | (elem[2j+1] << 4). Sequential nibble pairs.
struct block_q4_0 { half d; uchar qs[16]; };
kernel void q4_matvec(
    device const float* act     [[buffer(0)]],
    device const uchar* weight  [[buffer(1)]],
    device float* out           [[buffer(2)]],
    device const uint* p_K      [[buffer(3)]],
    device const uint* p_N      [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    uint K = p_K[0], N = p_N[0];
    uint nb = K / 32;
    uint row = tgid * 4 + sgitg / 2;
    if (row >= N) return;
    ushort half_sg = sgitg % 2;
    float sum = 0.0f;
    uint tid_in_row = half_sg * 32 + tiisg;
    device const block_q4_0* wr = (device const block_q4_0*)(weight + row * nb * 18);
    for (uint b = tid_in_row; b < nb; b += 64) {
        float d = float(wr[b].d);
        float s = 0.0f;
        uint aOff = b * 32;
        for (ushort j = 0; j < 16; j += 2) {
            float4 a = *((device const float4*)(act + aOff + j*2));
            uchar q0 = wr[b].qs[j];
            uchar q1 = wr[b].qs[j+1];
            s += a.x * ((q0 & 0xF) - 8) + a.y * ((q0 >> 4) - 8)
               + a.z * ((q1 & 0xF) - 8) + a.w * ((q1 >> 4) - 8);
        }
        sum += s * d;
    }
    sum = simd_sum(sum);
    threadgroup float shmem[8];
    if (tiisg == 0) shmem[sgitg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg % 2 == 0 && tiisg == 0)
        out[row] = shmem[sgitg] + shmem[sgitg + 1];
}

kernel void rope_rotate_half(
    device float* x          [[buffer(0)]],
    device const uint* p_hd  [[buffer(1)]],
    device const uint* p_nh  [[buffer(2)]],
    device const uint* p_pos [[buffer(3)]],
    device const float* p_th [[buffer(4)]],
    uint pid [[thread_position_in_grid]])
{
    uint headDim = p_hd[0], nHeads = p_nh[0], pos = p_pos[0];
    float theta = p_th[0];
    uint halfDim = headDim / 2;
    uint h = pid / halfDim;
    uint j = pid % halfDim;
    if (h >= nHeads) return;
    float freq = 1.0f / pow(theta, float(2*j) / float(headDim));
    float angle = float(pos) * freq;
    float cosA = cos(angle), sinA = sin(angle);
    uint base = h * headDim;
    float x0 = x[base + j], x1 = x[base + j + halfDim];
    x[base + j]           = x0 * cosA - x1 * sinA;
    x[base + j + halfDim] = x0 * sinA + x1 * cosA;
}

kernel void decode_attn(
    device const float* Q      [[buffer(0)]],
    device const float* kCache [[buffer(1)]],
    device const float* vCache [[buffer(2)]],
    device float* out          [[buffer(3)]],
    device const uint* p_kvDim   [[buffer(4)]],
    device const uint* p_headDim [[buffer(5)]],
    device const uint* p_nHeads  [[buffer(6)]],
    device const uint* p_nKVH   [[buffer(7)]],
    device const uint* p_seqLen [[buffer(8)]],
    uint h [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    uint kvDim = p_kvDim[0], headDim = p_headDim[0];
    uint nHeads = p_nHeads[0], nKVH = p_nKVH[0], seqLen = p_seqLen[0];
    if (h >= nHeads || tid >= headDim) return;
    uint kvMul = nHeads / nKVH;
    uint kvH = h / kvMul;
    float scale = 1.0f / sqrt(float(headDim));
    threadgroup float scores[4096];
    threadgroup float smax[1];
    threadgroup float ssum[1];
    uint work = (seqLen + headDim - 1) / headDim;
    for (uint w = 0; w < work; w++) {
        uint t = tid + w * headDim;
        if (t < seqLen) {
            float dot = 0.0f;
            for (uint d = 0; d < headDim; d++)
                dot += Q[h * headDim + d] * kCache[t * kvDim + kvH * headDim + d];
            scores[t] = dot * scale;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { float mx = -1e30f; for (uint t = 0; t < seqLen; t++) mx = max(mx, scores[t]); smax[0] = mx; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx = smax[0];
    for (uint w = 0; w < work; w++) { uint t = tid + w * headDim; if (t < seqLen) scores[t] = exp(scores[t] - mx); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { float s = 0.0f; for (uint t = 0; t < seqLen; t++) s += scores[t]; ssum[0] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float invSum = 1.0f / ssum[0];
    float acc = 0.0f;
    for (uint t = 0; t < seqLen; t++)
        acc += scores[t] * invSum * vCache[t * kvDim + kvH * headDim + tid];
    out[h * headDim + tid] = acc;
}

// SQ4 matvec: 4 rows per threadgroup, 2 simdgroups per row.
// Vectorized uchar4 reads (8 nibbles per load), float4 activation reads.
// 16-entry signed band table in threadgroup memory.
kernel void sq4_matvec(
    device const float* act      [[buffer(0)]],
    device const uchar* packed   [[buffer(1)]],
    device const float* bands    [[buffer(2)]],
    device float* out            [[buffer(3)]],
    device const uint* p_K       [[buffer(4)]],
    device const uint* p_N       [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    uint K = p_K[0], N = p_N[0];
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

    // Match Q8 pattern: 4 weights per iteration, 1 float4 activation per iteration.
    // uchar2 = 2 bytes = 4 nibbles = 4 weights. Coalesced with act4.
    device const uchar2* row2 = (device const uchar2*)(packed + row * half_K);
    device const float4* act4 = (device const float4*)act;
    uint K4 = K / 4;

    float sum = 0.0f;
    for (uint i = tid_in_row; i < K4; i += 64) {
        uchar2 chunk = row2[i];
        float4 a = act4[i];
        float4 w = float4(table16[chunk.x & 0x0F], table16[(chunk.x >> 4) & 0x0F],
                          table16[chunk.y & 0x0F], table16[(chunk.y >> 4) & 0x0F]);
        sum += dot(w, a);
    }

    uint handled = K4 * 4;
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
    if (sgitg % 2 == 0 && tiisg == 0)
        out[row] = shmem[sgitg] + shmem[sgitg + 1];
}

// SQ4 outlier correction
kernel void sq4_outlier_correct(
    device const uint* outlier_idx   [[buffer(0)]],
    device const float* outlier_val  [[buffer(1)]],
    device const uchar* packed       [[buffer(2)]],
    device const float* bands        [[buffer(3)]],
    device const float* act          [[buffer(4)]],
    device atomic_float* out         [[buffer(5)]],
    device const uint* p_cols        [[buffer(6)]],
    device const uint* p_count       [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    uint cols = p_cols[0], count = p_count[0];
    if (tid >= count) return;
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
