#include <metal_stdlib>
using namespace metal;

// Q8 matvec: 4 rows per threadgroup, 256 threads (8 simdgroups).
// Each simdgroup handles 1 row, 2 simdgroups share activation loads.
// Per-row scale, contiguous int8 weights.
kernel void q8_matvec(
    device const float* act    [[buffer(0)]],
    device const int8_t* weight [[buffer(1)]],
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
    uint wOff = row * K;
    float sum = 0.0f;
    uint tid_in_row = half_sg * 32 + tiisg;
    for (uint k = tid_in_row * 4; k + 3 < K; k += 256) {
        float4 a = float4(act[k], act[k+1], act[k+2], act[k+3]);
        sum += a.x * weight[wOff+k] + a.y * weight[wOff+k+1] + a.z * weight[wOff+k+2] + a.w * weight[wOff+k+3];
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

struct block_q4_0 { half d; uchar qs[16]; };

inline float bq4_dot_y(device const block_q4_0* qb, float sumy, thread float* yl, int il) {
    float d = float(qb->d);
    float acc0=0, acc1=0, acc2=0, acc3=0;
    uint qs_off = il/2;
    for (int i = 0; i < 8; i += 2) {
        ushort v = ((device const ushort*)(qb->qs))[qs_off + i/2];
        acc0 += yl[i+0] * (v & 0x000F);
        acc1 += yl[i+1] * (v & 0x0F00);
        acc2 += yl[i+8] * (v & 0x00F0);
        acc3 += yl[i+9] * (v & 0xF000);
    }
    return d * (sumy * -8.0f + acc0 + acc1 + acc2 + acc3);
}

kernel void q4_matvec(
    device const float* act     [[buffer(0)]],
    device const uchar* weight  [[buffer(1)]],
    device float* out           [[buffer(2)]],
    device const uint* p_K      [[buffer(3)]],
    device const uint* p_N      [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    uint K = p_K[0], N = p_N[0];
    const uint QK = 32;
    const short NR0 = 4;
    const short NQ = 16;
    uint nb = K / QK;
    uint r0 = (tgpig.x * 2 + sgitg) * NR0;
    if (r0 >= N) return;
    float sumf[4] = {0,0,0,0};
    short ix = tiisg / 2;
    short il = (tiisg % 2) * 8;
    uint ib0 = ix;
    uint yOff = ib0 * QK + il;
    for (uint ib = ib0; ib < nb; ib += NQ) {
        float sumy[2] = {0,0};
        float yl[16];
        for (short i = 0; i < 8; i += 2) {
            sumy[0] += act[yOff+i] + act[yOff+i+1];
            yl[i]   = act[yOff+i];
            yl[i+1] = act[yOff+i+1] / 256.0f;
            sumy[1] += act[yOff+i+16] + act[yOff+i+17];
            yl[i+8] = act[yOff+i+16] / 16.0f;
            yl[i+9] = act[yOff+i+17] / 4096.0f;
        }
        for (short row = 0; row < NR0 && r0+row < N; row++)
            sumf[row] += bq4_dot_y((device const block_q4_0*)(weight + (r0+row)*nb*18) + ib, sumy[0]+sumy[1], yl, il);
        yOff += QK * NQ;
    }
    for (short row = 0; row < NR0 && r0+row < N; row++) {
        float tot = simd_sum(sumf[row]);
        if (tiisg == 0) out[r0+row] = tot;
    }
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
