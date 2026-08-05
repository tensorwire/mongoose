#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ============================================================================
// MLX-derived INT4 quantized matvec (Apache 2.0 licensed by Apple).
// Simplified from MLX's affine_qmv_fast: no batch, float, gs=32, bits=4.
// Buffer layout: w(0) scales(1) biases(2) x(3) y(4) in_vec_size(5) out_vec_size(6)
// Dispatch: (1, ceil(N/8), 1) threadgroups, (32, 2, 1) threads
// ============================================================================

template <typename U, int values_per_thread>
inline U mlx_qdot_4(const device uint8_t* w, const thread U* xt, U s, U b, U sum) {
    U acc = 0;
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < values_per_thread / 4; i++) {
        uint16_t packed = ws[i];
        acc += xt[4*i]   * float((packed      ) & 0xF)
             + xt[4*i+1] * float((packed >>  4) & 0xF)
             + xt[4*i+2] * float((packed >>  8) & 0xF)
             + xt[4*i+3] * float((packed >> 12) & 0xF);
    }
    return s * acc + b * sum;
}

template <typename U, int values_per_thread>
inline U mlx_load_x_4(const device float* x, thread U* xt) {
    U sum = 0;
    for (int i = 0; i < values_per_thread; i += 4) {
        sum += x[i]+x[i+1]+x[i+2]+x[i+3];
        xt[i] = x[i]; xt[i+1] = x[i+1]; xt[i+2] = x[i+2]; xt[i+3] = x[i+3];
    }
    return sum;
}

kernel void sq4_mlx_qmv(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device float* biases [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]])
{
    constexpr int group_size = 32;
    constexpr int packs_per_thread = 2;
    constexpr int num_simdgroups = 2;
    constexpr int results_per_simdgroup = 4;
    constexpr int pack_factor = 8;
    constexpr int bytes_per_pack = 4;
    constexpr int values_per_thread = pack_factor * packs_per_thread;
    constexpr int block_size = values_per_thread * 32;
    constexpr int scale_step_per_thread = group_size / values_per_thread;

    const device uint8_t* ws = (const device uint8_t*)w;
    typedef float U;
    thread U x_thread[values_per_thread];
    thread U result[results_per_simdgroup] = {0};

    const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
    const int in_vec_size_g = in_vec_size / group_size;
    const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
        simd_gid * results_per_simdgroup;

    ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
    scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    x += simd_lid * values_per_thread;
    y += out_row;

    for (int k = 0; k < in_vec_size; k += block_size) {
        U sum = mlx_load_x_4<U, values_per_thread>(x, x_thread);
        for (int row = 0; row < results_per_simdgroup; row++) {
            const device uint8_t* wl = ws + row * in_vec_size_w;
            const device float* sl = scales + row * in_vec_size_g;
            const device float* bl = biases + row * in_vec_size_g;
            U s = sl[0];
            U b = bl[0];
            result[row] += mlx_qdot_4<U, values_per_thread>(wl, x_thread, s, b, sum);
        }
        ws += block_size * bytes_per_pack / pack_factor;
        scales += block_size / group_size;
        biases += block_size / group_size;
        x += block_size;
    }

    for (int row = 0; row < results_per_simdgroup; row++) {
        result[row] = simd_sum(result[row]);
        if (simd_lid == 0) y[row] = result[row];
    }
}

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

// Fused: bias_add Q/K/V + RoPE Q/K + KV cache write K/V — 7 ops in 1 kernel.
// Single threadgroup — uses threadgroup_barrier to sequence bias → RoPE → KV write.
kernel void fused_bias_rope_kv(
    device float* Q            [[buffer(0)]],
    device float* K            [[buffer(1)]],
    device float* V            [[buffer(2)]],
    device const float* bq     [[buffer(3)]],
    device const float* bk     [[buffer(4)]],
    device const float* bv     [[buffer(5)]],
    device float* kvCacheK     [[buffer(6)]],
    device float* kvCacheV     [[buffer(7)]],
    device const uint* p_dim   [[buffer(8)]],
    device const uint* p_kvDim [[buffer(9)]],
    device const uint* p_hd    [[buffer(10)]],
    device const uint* p_nh    [[buffer(11)]],
    device const uint* p_nkvh  [[buffer(12)]],
    device const uint* p_pos   [[buffer(13)]],
    device const float* p_th   [[buffer(14)]],
    uint tid [[thread_index_in_threadgroup]])
{
    uint dim = p_dim[0], kvDim = p_kvDim[0], headDim = p_hd[0];
    uint nHeads = p_nh[0], nKVHeads = p_nkvh[0], pos = p_pos[0];
    float theta = p_th[0];
    uint halfHead = headDim / 2;

    // Phase 1: Bias add (grid-stride within threadgroup)
    for (uint i = tid; i < dim; i += 256) Q[i] += bq[i];
    for (uint i = tid; i < kvDim; i += 256) { K[i] += bk[i]; V[i] += bv[i]; }
    threadgroup_barrier(mem_flags::mem_device);

    // Phase 2: RoPE on Q (all heads)
    for (uint pid = tid; pid < nHeads * halfHead; pid += 256) {
        uint h = pid / halfHead;
        uint j = pid % halfHead;
        float freq = 1.0f / pow(theta, float(2*j) / float(headDim));
        float angle = float(pos) * freq;
        float c = cos(angle), s = sin(angle);
        uint base = h * headDim;
        float x0 = Q[base + j], x1 = Q[base + j + halfHead];
        Q[base + j]            = x0 * c - x1 * s;
        Q[base + j + halfHead] = x0 * s + x1 * c;
    }
    // RoPE on K (KV heads)
    for (uint pid = tid; pid < nKVHeads * halfHead; pid += 256) {
        uint h = pid / halfHead;
        uint j = pid % halfHead;
        float freq = 1.0f / pow(theta, float(2*j) / float(headDim));
        float angle = float(pos) * freq;
        float c = cos(angle), s = sin(angle);
        uint base = h * headDim;
        float x0 = K[base + j], x1 = K[base + j + halfHead];
        K[base + j]            = x0 * c - x1 * s;
        K[base + j + halfHead] = x0 * s + x1 * c;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Phase 3: KV cache write
    for (uint i = tid; i < kvDim; i += 256) {
        uint cacheOff = pos * kvDim + i;
        kvCacheK[cacheOff] = K[i];
        kvCacheV[cacheOff] = V[i];
    }
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
    // Optional architecture attention scale. Bound to nullptr (or simply left
    // unbound) for Llama-family models, which keep 1/sqrt(headDim). Granite
    // passes attention_multiplier here, which is 1/headDim — a different value,
    // not a correction factor on top of the default.
    device const float* p_attnScale [[buffer(9)]],
    uint h [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    uint kvDim = p_kvDim[0], headDim = p_headDim[0];
    uint nHeads = p_nHeads[0], nKVH = p_nKVH[0], seqLen = p_seqLen[0];
    if (h >= nHeads || tid >= headDim) return;
    uint kvMul = nHeads / nKVH;
    uint kvH = h / kvMul;
    float scale = p_attnScale ? p_attnScale[0] : (1.0f / sqrt(float(headDim)));

    // Tiled online softmax.
    //
    // Two constraints have to be satisfied at once, and satisfying either alone
    // gives a slower kernel than the naive version:
    //
    //  1. The scores must stay SHARED. Each score is one Q·K dot over headDim.
    //     If every thread recomputes the score for every t, the kernel does
    //     headDim (=64) times the necessary dot-product work. Measured: 3x
    //     slower end to end than the code it replaced, despite removing the
    //     serial reduction.
    //  2. The reduction must stay PARALLEL. The previous version staged all
    //     scores in threadgroup memory and folded them with
    //     `if (tid == 0) for (t < seqLen)` — one thread of headDim doing
    //     O(seqLen) serial work per head, per layer, per token. In prefill,
    //     which runs this once per prompt token, that makes total cost
    //     quadratic in prompt length: 11 -> 50 ms/token as prompts grew
    //     149 -> 4209 tokens.
    //
    // So: process the sequence in tiles of TILE. Threads cooperate to compute
    // TILE scores (each score still computed exactly once), reduce the tile's
    // max and sum with simd intrinsics, then fold the tile into a running
    // (max, sum, acc) with the standard online-softmax rescale. Threadgroup
    // memory is O(TILE) rather than O(seqLen), which also removes the 4096
    // context cap that silently truncated longer prompts.
    const uint TILE = 256;
    const uint MAX_SG = 32;        // headDim <= 1024 threads / 32 per simdgroup
    threadgroup float sc[256];
    // Per-simdgroup reduction slots. A single shared accumulator written by
    // every simdgroup is a race: `if (simd_is_first()) tgRed += x` has no
    // ordering between simdgroups, and with headDim=64 there are two of them.
    // That produced results that were exactly right whenever only one simdgroup
    // had work (seqLen % 256 <= 32) and wrong otherwise — the kind of bug that
    // passes a spot check at one length and fails everywhere else.
    threadgroup float sgMax[MAX_SG];
    threadgroup float sgSum[MAX_SG];

    uint nSG   = (headDim + 31u) / 32u;
    uint sgId  = tid / 32u;
    uint lane  = tid % 32u;

    device const float* qHead = Q + h * headDim;
    device const float* kBase = kCache + kvH * headDim;
    device const float* vBase = vCache + kvH * headDim;

    float runMax = -INFINITY;
    float runSum = 0.0f;
    float acc    = 0.0f;

    for (uint base = 0; base < seqLen; base += TILE) {
        uint n = min(TILE, seqLen - base);

        // Each thread computes scores for a strided subset of the tile, so a
        // given score is computed once regardless of how many threads there are.
        for (uint i = tid; i < n; i += headDim) {
            uint t = base + i;
            float dot = 0.0f;
            device const float* kRow = kBase + t * kvDim;
            for (uint d = 0; d < headDim; d++) dot += qHead[d] * kRow[d];
            sc[i] = dot * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Tile max: reduce within each simdgroup, then across simdgroups via
        // dedicated slots. Every thread reads all nSG slots, so the result is
        // identical on all threads without a second barrier.
        float local = -INFINITY;
        for (uint i = tid; i < n; i += headDim) local = max(local, sc[i]);
        local = simd_max(local);
        if (lane == 0) sgMax[sgId] = local;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tileMax = sgMax[0];
        for (uint s = 1; s < nSG; s++) tileMax = max(tileMax, sgMax[s]);

        // exp() relative to the tile max, then tile sum, also reduced in
        // parallel. Writing exp back into sc lets the value pass below reuse it.
        float lsum = 0.0f;
        for (uint i = tid; i < n; i += headDim) {
            float w = exp(sc[i] - tileMax);
            sc[i] = w;
            lsum += w;
        }
        lsum = simd_sum(lsum);
        if (lane == 0) sgSum[sgId] = lsum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tileSum = 0.0f;
        for (uint s = 0; s < nSG; s++) tileSum += sgSum[s];

        // Fold this tile into the running accumulator. correction is 1.0 while
        // the max is unchanged, so a monotone-decreasing score sequence costs
        // nothing extra.
        float newMax = max(runMax, tileMax);
        float corr = exp(runMax - newMax);
        float tcorr = exp(tileMax - newMax);
        runSum = runSum * corr + tileSum * tcorr;

        // Value pass: every thread owns one output component (tid = d), and
        // walks the tile accumulating sc[i] * V[t][d].
        float vacc = 0.0f;
        for (uint i = 0; i < n; i++) vacc += sc[i] * vBase[(base + i) * kvDim + tid];
        acc = acc * corr + vacc * tcorr;
        runMax = newMax;

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    out[h * headDim + tid] = acc / runSum;
}

// ============================================================================
// SQ4-Fast matvec: per-row scale+bias, Q8-style inner loop.
// The inner loop is: sum += float(nibble) * act. Scale applied once per row.
// Same ALU as Q8 but reads half the bytes.
// Per-row scale+bias computed at upload from the tensor's band means.
// ============================================================================

kernel void sq4_matvec_fast(
    device const float* act         [[buffer(0)]],
    device const uchar* packed      [[buffer(1)]],
    device const float* row_scales  [[buffer(2)]],   // [N] per-row scale
    device const float* row_biases  [[buffer(3)]],   // [N] per-row bias
    device float* out               [[buffer(4)]],
    device const uint* p_K          [[buffer(5)]],
    device const uint* p_N          [[buffer(6)]],
    device const uint* outlier_idx  [[buffer(7)]],
    device const float* outlier_val [[buffer(8)]],
    device const uint* p_oc         [[buffer(9)]],
    uint tgid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    uint K = p_K[0], N = p_N[0];
    uint oc = p_oc[0];

    uint row = tgid * 4 + sgitg / 2;
    if (row >= N) return;

    ushort half_sg = sgitg % 2;
    uint tid_in_row = half_sg * 32 + tiisg;
    uint half_K = K / 2;

    device const uint4* row_packed4 = (device const uint4*)(packed + row * half_K);
    uint n_uint4 = half_K >> 4;

    float nib_sum = 0.0f;  // sum of nibble*act (unscaled)
    float act_sum = 0.0f;  // sum of act (for bias)

    for (uint i = tid_in_row; i < n_uint4; i += 64) {
        uint4 chunk = row_packed4[i];
        uint base = i * 32;
        uint ww[4] = {chunk.x, chunk.y, chunk.z, chunk.w};

        for (int qq = 0; qq < 4; qq++) {
            uint wd = ww[qq];
            uint cb = base + qq * 8;
            float4 a0 = *((device const float4*)(act + cb));
            float4 a1 = *((device const float4*)(act + cb + 4));

            nib_sum += float((wd>> 0)&0xF) * a0.x
                     + float((wd>> 4)&0xF) * a0.y
                     + float((wd>> 8)&0xF) * a0.z
                     + float((wd>>12)&0xF) * a0.w
                     + float((wd>>16)&0xF) * a1.x
                     + float((wd>>20)&0xF) * a1.y
                     + float((wd>>24)&0xF) * a1.z
                     + float((wd>>28)&0xF) * a1.w;

            act_sum += a0.x + a0.y + a0.z + a0.w + a1.x + a1.y + a1.z + a1.w;
        }
    }

    // Tail
    uint handled = n_uint4 * 32;
    for (uint k = handled + tid_in_row; k < K; k += 64) {
        uint byte_idx = k >> 1;
        uint shift = (k & 1) * 4;
        uint nib = (packed[row * half_K + byte_idx] >> shift) & 0x0F;
        float a = act[k];
        nib_sum += float(nib) * a;
        act_sum += a;
    }

    nib_sum = simd_sum(nib_sum);
    act_sum = simd_sum(act_sum);

    threadgroup float ns[8], as[8];
    if (tiisg == 0) { ns[sgitg] = nib_sum; as[sgitg] = act_sum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg % 2 == 0 && tiisg == 0) {
        float total_nib = ns[sgitg] + ns[sgitg + 1];
        float total_act = as[sgitg] + as[sgitg + 1];

        // Apply per-row scale and bias: weight = nibble * scale + bias
        // dot(weight, act) = scale * sum(nibble * act) + bias * sum(act)
        float s = row_scales[row];
        float b = row_biases[row];
        float total = s * total_nib + b * total_act;

        // Outlier correction
        if (oc > 0) {
            uint target = row * K;
            uint row_end = target + K;
            uint lo = 0, hi = oc;
            while (lo < hi) { uint mid=(lo+hi)>>1; if (outlier_idx[mid]<target) lo=mid+1; else hi=mid; }
            for (uint i = lo; i < oc && outlier_idx[i] < row_end; i++) {
                uint col = outlier_idx[i] - target;
                uint byte_idx = row * half_K + (col >> 1);
                uint shift = (col & 1) * 4;
                uint nib = (packed[byte_idx] >> shift) & 0x0F;
                float approx = float(nib) * s + b;
                total += (outlier_val[i] - approx) * act[col];
            }
        }

        out[row] = total;
    }
}

// ============================================================================
// SQ4-Linear matvec: MLX-style mask-and-multiply dequant. No LUT.
//
// Weights stored as linear INT4 with per-group scale+bias (group_size=32).
// Dequant: weight = nibble_raw * scale + bias. Pure ALU, zero memory lookup.
// Nibble extraction via bitmask (no shift): activation pre-divided by 16^position.
//
// Data layout (prepared at upload from SQ4 band means):
//   packed: uint32 words, each holding 8 nibbles LSB-first (same as nibble format)
//   scales: [N * K/group_size] float — per-group scale
//   biases: [N * K/group_size] float — per-group bias
//
// Buffer layout: act(0) packed(1) scales(2) biases(3) out(4) K(5) N(6) oc-related(7,8,9)
// ============================================================================

#define LIN_GROUP_SIZE 32

kernel void sq4_matvec_linear(
    device const float* act       [[buffer(0)]],
    device const uchar* packed    [[buffer(1)]],
    device const float* scales    [[buffer(2)]],
    device const float* biases    [[buffer(3)]],
    device float* out             [[buffer(4)]],
    device const uint* p_K        [[buffer(5)]],
    device const uint* p_N        [[buffer(6)]],
    device const uint* outlier_idx  [[buffer(7)]],
    device const float* outlier_val [[buffer(8)]],
    device const uint* p_outlier_count [[buffer(9)]],
    uint tgid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    uint K = p_K[0], N = p_N[0];
    uint oc = p_outlier_count[0];

    uint row = tgid * 4 + sgitg / 2;
    if (row >= N) return;

    ushort half_sg = sgitg % 2;
    uint tid_in_row = half_sg * 32 + tiisg;
    uint half_K = K / 2;
    uint groups_per_row = K / LIN_GROUP_SIZE;

    device const uint4* row_packed4 = (device const uint4*)(packed + row * half_K);
    device const float* row_scales = scales + row * groups_per_row;
    device const float* row_biases = biases + row * groups_per_row;

    // 32 nibbles per iteration (uint4 = 16 bytes = 32 nibbles), matching LUT kernel bandwidth
    uint n_uint4 = half_K >> 4;

    float sum = 0.0f;
    for (uint i = tid_in_row; i < n_uint4; i += 64) {
        uint4 chunk = row_packed4[i];
        uint base = i * 32;
        uint ww[4] = {chunk.x, chunk.y, chunk.z, chunk.w};

        for (int qq = 0; qq < 4; qq++) {
            uint wd = ww[qq];
            uint cb = base + qq * 8;
            uint group = cb / LIN_GROUP_SIZE;
            float s = row_scales[group];
            float b = row_biases[group];
            float4 a0 = *((device const float4*)(act + cb));
            float4 a1 = *((device const float4*)(act + cb + 4));

            // Factor out bias: (nib*s + b)*act = nib*s*act + b*act
            // b*sum(act) computed once per 8-element group
            float asum = a0.x + a0.y + a0.z + a0.w + a1.x + a1.y + a1.z + a1.w;
            sum += float((wd>> 0)&0xF) * s * a0.x
                 + float((wd>> 4)&0xF) * s * a0.y
                 + float((wd>> 8)&0xF) * s * a0.z
                 + float((wd>>12)&0xF) * s * a0.w
                 + float((wd>>16)&0xF) * s * a1.x
                 + float((wd>>20)&0xF) * s * a1.y
                 + float((wd>>24)&0xF) * s * a1.z
                 + float((wd>>28)&0xF) * s * a1.w
                 + b * asum;
        }
    }

    sum = simd_sum(sum);

    threadgroup float shmem[8];
    if (tiisg == 0) shmem[sgitg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg % 2 == 0 && tiisg == 0) {
        float total = shmem[sgitg] + shmem[sgitg + 1];

        // Outlier correction
        if (oc > 0) {
            uint target = row * K;
            uint row_end = target + K;
            uint lo = 0, hi = oc;
            while (lo < hi) {
                uint mid = (lo + hi) >> 1;
                if (outlier_idx[mid] < target) lo = mid + 1;
                else hi = mid;
            }
            for (uint i = lo; i < oc && outlier_idx[i] < row_end; i++) {
                uint flat = outlier_idx[i];
                uint col = flat - target;
                uint byte_idx = col >> 1;
                uint shift = (col & 1) * 4;
                uint nib = (packed[row * half_K + byte_idx] >> shift) & 0x0F;
                uint group = col / LIN_GROUP_SIZE;
                float band_approx = float(nib) * row_scales[group] + row_biases[group];
                total += (outlier_val[i] - band_approx) * act[col];
            }
        }

        out[row] = total;
    }
}

// ============================================================================
// SQ4 matvec with texture LUT — dequant through texture cache, not threadgroup.
// Texture holds ALL tensors' LUTs concatenated: lut_tex[tensor_offset + nibble].
// The texture cache is a separate memory path from threadgroup — zero contention.
// Same dispatch shape and buffer layout as sq4_matvec (drop-in replacement).
// Buffer 2 repurposed: bands_offset (uint) instead of bands pointer.
// ============================================================================

kernel void sq4_matvec_tex(
    device const float* act       [[buffer(0)]],
    device const uchar* packed    [[buffer(1)]],
    device const uint* p_lut_off  [[buffer(2)]],   // LUT offset in texture (in elements)
    device float* out             [[buffer(3)]],
    device const uint* p_K        [[buffer(4)]],
    device const uint* p_N        [[buffer(5)]],
    device const uint* outlier_idx  [[buffer(6)]],
    device const float* outlier_val [[buffer(7)]],
    device const uint* p_outlier_count [[buffer(8)]],
    texture1d<float, access::read> lut_tex [[texture(0)]],
    uint tgid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    uint K = p_K[0], N = p_N[0];
    uint oc = p_outlier_count[0];
    uint lut_off = p_lut_off[0];

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

        {   uint ww[4] = {chunk.x, chunk.y, chunk.z, chunk.w};
            for (int qq = 0; qq < 4; qq++) {
                uint wd = ww[qq]; uint cb = base + qq * 8;
                float4 a0 = *((device const float4*)(act + cb));
                float4 a1 = *((device const float4*)(act + cb + 4));
                sum += lut_tex.read(ushort(lut_off + ((wd>> 0)&0xF))).x * a0.x
                     + lut_tex.read(ushort(lut_off + ((wd>> 4)&0xF))).x * a0.y
                     + lut_tex.read(ushort(lut_off + ((wd>> 8)&0xF))).x * a0.z
                     + lut_tex.read(ushort(lut_off + ((wd>>12)&0xF))).x * a0.w
                     + lut_tex.read(ushort(lut_off + ((wd>>16)&0xF))).x * a1.x
                     + lut_tex.read(ushort(lut_off + ((wd>>20)&0xF))).x * a1.y
                     + lut_tex.read(ushort(lut_off + ((wd>>24)&0xF))).x * a1.z
                     + lut_tex.read(ushort(lut_off + ((wd>>28)&0xF))).x * a1.w;
            }
        }
    }

    uint handled = n_uint4 * 32;
    for (uint k = handled + tid_in_row; k < K; k += 64) {
        uint byte_idx = k >> 1;
        uint shift = (k & 1) * 4;
        uchar nibble = (packed[row * half_K + byte_idx] >> shift) & 0x0F;
        sum += lut_tex.read(ushort(lut_off + nibble)).x * act[k];
    }

    sum = simd_sum(sum);

    threadgroup float shmem[8];
    if (tiisg == 0) shmem[sgitg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg % 2 == 0 && tiisg == 0) {
        float total = shmem[sgitg] + shmem[sgitg + 1];

        if (oc > 0) {
            uint target = row * K;
            uint row_end = target + K;
            uint lo = 0, hi = oc;
            while (lo < hi) {
                uint mid = (lo + hi) >> 1;
                if (outlier_idx[mid] < target) lo = mid + 1;
                else hi = mid;
            }
            for (uint i = lo; i < oc && outlier_idx[i] < row_end; i++) {
                uint flat = outlier_idx[i];
                uint col = flat - target;
                uint byte_idx = row * half_K + (col >> 1);
                uint shift = (col & 1) * 4;
                uchar nibble = (packed[byte_idx] >> shift) & 0x0F;
                total += (outlier_val[i] - lut_tex.read(ushort(lut_off + nibble)).x) * act[col];
            }
        }

        out[row] = total;
    }
}

// SQ4 decode matvec: 4 rows/tg, 2 simdgroups/row, uint4 reads, 16-entry tg LUT.
// Fused outlier correction. M=1 decode path.
kernel void sq4_matvec(
    device const float* act       [[buffer(0)]],
    device const uchar* packed    [[buffer(1)]],
    device const float* bands     [[buffer(2)]],
    device float* out             [[buffer(3)]],
    device const uint* p_K        [[buffer(4)]],
    device const uint* p_N        [[buffer(5)]],
    device const uint* outlier_idx  [[buffer(6)]],
    device const float* outlier_val [[buffer(7)]],
    device const uint* p_outlier_count [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    uint K = p_K[0], N = p_N[0];
    uint oc = p_outlier_count[0];

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

        if (oc > 0) {
            uint target = row * K;
            uint row_end = target + K;
            uint lo = 0, hi = oc;
            while (lo < hi) {
                uint mid = (lo + hi) >> 1;
                if (outlier_idx[mid] < target) lo = mid + 1;
                else hi = mid;
            }
            for (uint i = lo; i < oc && outlier_idx[i] < row_end; i++) {
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

// ============================================================================
// AMX SQ4 matvec — 2-simdgroup bulk dequant + AMX multiply.
//
// 2 simdgroups (64 threads), each handles 8 rows = 16 rows per threadgroup.
// Phase 1: All 64 threads bulk-dequant in parallel. 2x faster than 1 simdgroup.
// Phase 2: Each simdgroup runs AMX independently. No inter-tile barriers.
//
// Threadgroup memory: 2 × 8 × K × 2 + overhead ≈ 29KB for K=896. Fits 32KB.
// Grid: ceil(N/16) threadgroups, 64 threads each.
// ============================================================================

kernel void sq4_matvec_amx(
    device const float* act       [[buffer(0)]],
    device const uchar* packed    [[buffer(1)]],
    device const float* bands     [[buffer(2)]],
    device float* out             [[buffer(3)]],
    device const uint* p_K        [[buffer(4)]],
    device const uint* p_N        [[buffer(5)]],
    device const uint* outlier_idx  [[buffer(6)]],
    device const float* outlier_val [[buffer(7)]],
    device const uint* p_outlier_count [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    uint K = p_K[0], N = p_N[0];
    uint oc = p_outlier_count[0];
    uint half_K = K / 2;

    // 2 simdgroups × 8 rows = 16 rows per threadgroup
    uint row_base = tgid * 16 + sgitg * 8;
    if (row_base >= N) return;

    // LUT shared across all simdgroups
    threadgroup half lut[16];
    if (sgitg == 0 && tiisg < 8) {
        lut[tiisg] = half(bands[tiisg]);
        lut[tiisg + 8] = -half(bands[tiisg]);
    }

    // Per-simdgroup scratch: 8 rows × K halfs + activation broadcast tile
    // 2 simdgroups × 8 × 896 × 2 = 28,672 bytes (fits 32KB)
    threadgroup half w_all[2][8 * 896];
    threadgroup half a_all[2][8][8];
    threadgroup float result[2][8][8];

    threadgroup_barrier(mem_flags::mem_threadgroup); // LUT ready

    // Phase 1: Each simdgroup dequants its own 8 rows — 128 threads total
    threadgroup half* my_w = w_all[sgitg];
    uint total_elems = 8 * K;
    for (uint i = tiisg; i < total_elems; i += 32) {
        uint r = i / K;
        uint c = i % K;
        uint row = row_base + r;
        if (row < N) {
            uint flat = row * half_K + c / 2;
            uint shift = (c & 1) * 4;
            uchar nib = (packed[flat] >> shift) & 0x0F;
            my_w[r * K + c] = lut[nib];
        } else {
            my_w[r * K + c] = 0.0h;
        }
    }
    // Need barrier here: LUT writes from sgitg=0 must be visible to all simdgroups
    // AND each simdgroup's w_all must be complete before AMX reads it.
    // But AMX only reads my_w (own simdgroup's data) — so we only need
    // intra-simdgroup visibility, which is guaranteed by lockstep execution.
    // The LUT barrier above already handles cross-simdgroup LUT visibility.
    // Still need one barrier to ensure all dequant writes are committed to
    // threadgroup memory before simdgroup_load reads them.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Each simdgroup runs AMX on its 8 rows
    simdgroup_float8x8 acc;
    acc = simdgroup_float8x8(0);

    uint n_tiles = K / 8;
    for (uint tk = 0; tk < n_tiles; tk++) {
        uint col_base = tk * 8;

        // Activation broadcast — no barrier needed within simdgroup
        if (tiisg < 8) {
            half av = half(act[col_base + tiisg]);
            a_all[sgitg][tiisg][0] = av; a_all[sgitg][tiisg][1] = av;
            a_all[sgitg][tiisg][2] = av; a_all[sgitg][tiisg][3] = av;
            a_all[sgitg][tiisg][4] = av; a_all[sgitg][tiisg][5] = av;
            a_all[sgitg][tiisg][6] = av; a_all[sgitg][tiisg][7] = av;
        }

        simdgroup_half8x8 A, B;
        simdgroup_load(A, &my_w[col_base], K);
        simdgroup_load(B, (const threadgroup half*)a_all[sgitg], 8);
        simdgroup_multiply_accumulate(acc, A, B, acc);
    }

    // Phase 3: Store and write
    simdgroup_store(acc, (threadgroup float*)result[sgitg], 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg < 8) {
        uint row = row_base + tiisg;
        if (row < N) {
            float total = result[sgitg][tiisg][0];

            if (oc > 0) {
                uint target = row * K;
                uint row_end = target + K;
                uint lo = 0, hi = oc;
                while (lo < hi) {
                    uint mid = (lo + hi) >> 1;
                    if (outlier_idx[mid] < target) lo = mid + 1;
                    else hi = mid;
                }
                for (uint i = lo; i < oc && outlier_idx[i] < row_end; i++) {
                    uint flat_idx = outlier_idx[i];
                    uint col = flat_idx - target;
                    uchar nib = (packed[row * half_K + col/2] >> ((col&1)*4)) & 0x0F;
                    float band_approx = (nib & 0x08) ? -float(lut[nib & 0x07]) : float(lut[nib & 0x07]);
                    total += (outlier_val[i] - band_approx) * act[col];
                }
            }

            out[row] = total;
        }
    }
}

// ============================================================================
// SQ4 bulk dequant: unpack N nibbles → N half-precision floats.
// Runs once per weight matrix before the matvec. Massively parallel —
// each thread dequants one weight. The matvec then reads clean FP16.
// ============================================================================

kernel void sq4_dequant_to_half(
    device const uchar* packed    [[buffer(0)]],
    device const float* bands     [[buffer(1)]],
    device half* out              [[buffer(2)]],
    constant uint& count          [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    uint byte_idx = tid / 2;
    uint shift = (tid & 1) * 4;
    uchar nibble = (packed[byte_idx] >> shift) & 0x0F;
    float val = bands[nibble & 0x07];
    if (nibble & 0x08) val = -val;
    out[tid] = half(val);
}

// FP16 matvec: reads pre-dequanted half weights. No LUT, no dequant overhead.
// Same dispatch shape as Q8: 4 rows/tg, 2 simdgroups/row, 256 threads.
// Vectorized half8 weight loads + float4 activation loads.
kernel void fp16_matvec(
    device const float* act       [[buffer(0)]],
    device const half* weights    [[buffer(1)]],
    device float* out             [[buffer(2)]],
    device const uint* p_K        [[buffer(3)]],
    device const uint* p_N        [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    uint K = p_K[0], N = p_N[0];
    uint row = tgid * 4 + sgitg / 2;
    if (row >= N) return;
    ushort half_sg = sgitg % 2;
    uint tid_in_row = half_sg * 32 + tiisg;

    device const half* wRow = weights + row * K;
    uint K8 = K / 8;

    float sum = 0.0f;
    for (uint i = tid_in_row; i < K8; i += 64) {
        uint base = i * 8;
        half4 w0 = *((device const half4*)(wRow + base));
        half4 w1 = *((device const half4*)(wRow + base + 4));
        float4 a0 = *((device const float4*)(act + base));
        float4 a1 = *((device const float4*)(act + base + 4));
        sum += float(w0.x)*a0.x + float(w0.y)*a0.y + float(w0.z)*a0.z + float(w0.w)*a0.w
             + float(w1.x)*a1.x + float(w1.y)*a1.y + float(w1.z)*a1.z + float(w1.w)*a1.w;
    }

    uint handled = K8 * 8;
    for (uint k = handled + tid_in_row; k < K; k += 64) {
        sum += float(wRow[k]) * act[k];
    }

    sum = simd_sum(sum);

    threadgroup float shmem[8];
    if (tiisg == 0) shmem[sgitg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg % 2 == 0 && tiisg == 0) {
        out[row] = shmem[sgitg] + shmem[sgitg + 1];
    }
}

// Combined SQ4 dequant + matvec: dequants inline to half, then dot product.
// Same dispatch shape as scalar sq4_matvec. No staging buffer needed.
// Avoids threadgroup LUT — dequants 8 nibbles per thread per iteration inline.
kernel void sq4_matvec_fp16(
    device const float* act       [[buffer(0)]],
    device const uchar* packed    [[buffer(1)]],
    device const float* bands     [[buffer(2)]],
    device float* out             [[buffer(3)]],
    device const uint* p_K        [[buffer(4)]],
    device const uint* p_N        [[buffer(5)]],
    device const uint* outlier_idx  [[buffer(6)]],
    device const float* outlier_val [[buffer(7)]],
    device const uint* p_outlier_count [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    uint K = p_K[0], N = p_N[0];
    uint oc = p_outlier_count[0];

    // Register-resident LUT: 8 band means → 16 values (positive + negative)
    half table16h[16];
    for (int i = 0; i < 8; i++) {
        table16h[i] = half(bands[i]);
        table16h[i+8] = -half(bands[i]);
    }

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

        // Dequant 32 nibbles to half inline, multiply by float activation
        { uint w = chunk.x; uint cb = base;
          float4 a0 = *((device const float4*)(act + cb));
          float4 a1 = *((device const float4*)(act + cb + 4));
          sum += float(table16h[(w>> 0)&0xF])*a0.x + float(table16h[(w>> 4)&0xF])*a0.y
               + float(table16h[(w>> 8)&0xF])*a0.z + float(table16h[(w>>12)&0xF])*a0.w
               + float(table16h[(w>>16)&0xF])*a1.x + float(table16h[(w>>20)&0xF])*a1.y
               + float(table16h[(w>>24)&0xF])*a1.z + float(table16h[(w>>28)&0xF])*a1.w; }
        { uint w = chunk.y; uint cb = base + 8;
          float4 a0 = *((device const float4*)(act + cb));
          float4 a1 = *((device const float4*)(act + cb + 4));
          sum += float(table16h[(w>> 0)&0xF])*a0.x + float(table16h[(w>> 4)&0xF])*a0.y
               + float(table16h[(w>> 8)&0xF])*a0.z + float(table16h[(w>>12)&0xF])*a0.w
               + float(table16h[(w>>16)&0xF])*a1.x + float(table16h[(w>>20)&0xF])*a1.y
               + float(table16h[(w>>24)&0xF])*a1.z + float(table16h[(w>>28)&0xF])*a1.w; }
        { uint w = chunk.z; uint cb = base + 16;
          float4 a0 = *((device const float4*)(act + cb));
          float4 a1 = *((device const float4*)(act + cb + 4));
          sum += float(table16h[(w>> 0)&0xF])*a0.x + float(table16h[(w>> 4)&0xF])*a0.y
               + float(table16h[(w>> 8)&0xF])*a0.z + float(table16h[(w>>12)&0xF])*a0.w
               + float(table16h[(w>>16)&0xF])*a1.x + float(table16h[(w>>20)&0xF])*a1.y
               + float(table16h[(w>>24)&0xF])*a1.z + float(table16h[(w>>28)&0xF])*a1.w; }
        { uint w = chunk.w; uint cb = base + 24;
          float4 a0 = *((device const float4*)(act + cb));
          float4 a1 = *((device const float4*)(act + cb + 4));
          sum += float(table16h[(w>> 0)&0xF])*a0.x + float(table16h[(w>> 4)&0xF])*a0.y
               + float(table16h[(w>> 8)&0xF])*a0.z + float(table16h[(w>>12)&0xF])*a0.w
               + float(table16h[(w>>16)&0xF])*a1.x + float(table16h[(w>>20)&0xF])*a1.y
               + float(table16h[(w>>24)&0xF])*a1.z + float(table16h[(w>>28)&0xF])*a1.w; }
    }

    uint handled = n_uint4 * 32;
    for (uint k = handled + tid_in_row; k < K; k += 64) {
        uint byte_idx = k >> 1;
        uint shift = (k & 1) * 4;
        uchar nibble = (packed[row * half_K + byte_idx] >> shift) & 0x0F;
        sum += float(table16h[nibble]) * act[k];
    }

    sum = simd_sum(sum);

    threadgroup float shmem[8];
    if (tiisg == 0) shmem[sgitg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg % 2 == 0 && tiisg == 0) {
        float total = shmem[sgitg] + shmem[sgitg + 1];

        if (oc > 0) {
            uint target = row * K;
            uint row_end = target + K;
            uint lo = 0, hi = oc;
            while (lo < hi) {
                uint mid = (lo + hi) >> 1;
                if (outlier_idx[mid] < target) lo = mid + 1;
                else hi = mid;
            }
            for (uint i = lo; i < oc && outlier_idx[i] < row_end; i++) {
                uint flat = outlier_idx[i];
                uint col = flat - target;
                uint byte_idx = row * half_K + (col >> 1);
                uint shift = (col & 1) * 4;
                uchar nibble = (packed[byte_idx] >> shift) & 0x0F;
                total += (outlier_val[i] - float(table16h[nibble])) * act[col];
            }
        }

        out[row] = total;
    }
}

// Apply outlier corrections to FP16 staging buffer.
// Overwrites the dequanted half value with the exact outlier value.
kernel void sq4_outlier_apply_fp16(
    device const uint* outlier_idx    [[buffer(0)]],
    device const float* outlier_val   [[buffer(1)]],
    device half* staged               [[buffer(2)]],
    device const uint* p_count        [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint count = p_count[0];
    if (tid >= count) return;
    uint flat = outlier_idx[tid];
    staged[flat] = half(outlier_val[tid]);
}

// ============================================================================
// SQ4 simd_shuffle matvec: LUT via register shuffle, zero memory access.
//
// Each thread in the simdgroup holds one entry of the 16-value LUT in a register.
// Threads 0-7: +bands[0..7]. Threads 8-15: -bands[0..7].
// Lookup = simd_shuffle(my_val, nibble) — 1-cycle register-to-register transfer.
// No threadgroup memory. No bank conflicts. No latency.
//
// Same buffer layout as sq4_matvec. Drop-in replacement.
// ============================================================================

kernel void sq4_matvec_shuffle(
    device const float* act       [[buffer(0)]],
    device const uchar* packed    [[buffer(1)]],
    device const float* bands     [[buffer(2)]],
    device float* out             [[buffer(3)]],
    device const uint* p_K        [[buffer(4)]],
    device const uint* p_N        [[buffer(5)]],
    device const uint* outlier_idx  [[buffer(6)]],
    device const float* outlier_val [[buffer(7)]],
    device const uint* p_outlier_count [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    uint K = p_K[0], N = p_N[0];
    uint oc = p_outlier_count[0];

    // Each thread holds its LUT entry: threads 0-7 = +bands, 8-15 = -bands, 16-31 = 0
    float my_lut = 0.0f;
    if (tiisg < 8) my_lut = bands[tiisg];
    else if (tiisg < 16) my_lut = -bands[tiisg - 8];

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
          sum += simd_shuffle(my_lut, (w>> 0)&0xF) * a0.x
               + simd_shuffle(my_lut, (w>> 4)&0xF) * a0.y
               + simd_shuffle(my_lut, (w>> 8)&0xF) * a0.z
               + simd_shuffle(my_lut, (w>>12)&0xF) * a0.w
               + simd_shuffle(my_lut, (w>>16)&0xF) * a1.x
               + simd_shuffle(my_lut, (w>>20)&0xF) * a1.y
               + simd_shuffle(my_lut, (w>>24)&0xF) * a1.z
               + simd_shuffle(my_lut, (w>>28)&0xF) * a1.w; }
        { uint w = chunk.y; uint cb = base + 8;
          float4 a0 = *((device const float4*)(act + cb));
          float4 a1 = *((device const float4*)(act + cb + 4));
          sum += simd_shuffle(my_lut, (w>> 0)&0xF) * a0.x
               + simd_shuffle(my_lut, (w>> 4)&0xF) * a0.y
               + simd_shuffle(my_lut, (w>> 8)&0xF) * a0.z
               + simd_shuffle(my_lut, (w>>12)&0xF) * a0.w
               + simd_shuffle(my_lut, (w>>16)&0xF) * a1.x
               + simd_shuffle(my_lut, (w>>20)&0xF) * a1.y
               + simd_shuffle(my_lut, (w>>24)&0xF) * a1.z
               + simd_shuffle(my_lut, (w>>28)&0xF) * a1.w; }
        { uint w = chunk.z; uint cb = base + 16;
          float4 a0 = *((device const float4*)(act + cb));
          float4 a1 = *((device const float4*)(act + cb + 4));
          sum += simd_shuffle(my_lut, (w>> 0)&0xF) * a0.x
               + simd_shuffle(my_lut, (w>> 4)&0xF) * a0.y
               + simd_shuffle(my_lut, (w>> 8)&0xF) * a0.z
               + simd_shuffle(my_lut, (w>>12)&0xF) * a0.w
               + simd_shuffle(my_lut, (w>>16)&0xF) * a1.x
               + simd_shuffle(my_lut, (w>>20)&0xF) * a1.y
               + simd_shuffle(my_lut, (w>>24)&0xF) * a1.z
               + simd_shuffle(my_lut, (w>>28)&0xF) * a1.w; }
        { uint w = chunk.w; uint cb = base + 24;
          float4 a0 = *((device const float4*)(act + cb));
          float4 a1 = *((device const float4*)(act + cb + 4));
          sum += simd_shuffle(my_lut, (w>> 0)&0xF) * a0.x
               + simd_shuffle(my_lut, (w>> 4)&0xF) * a0.y
               + simd_shuffle(my_lut, (w>> 8)&0xF) * a0.z
               + simd_shuffle(my_lut, (w>>12)&0xF) * a0.w
               + simd_shuffle(my_lut, (w>>16)&0xF) * a1.x
               + simd_shuffle(my_lut, (w>>20)&0xF) * a1.y
               + simd_shuffle(my_lut, (w>>24)&0xF) * a1.z
               + simd_shuffle(my_lut, (w>>28)&0xF) * a1.w; }
    }

    // Tail
    uint handled = n_uint4 * 32;
    for (uint k = handled + tid_in_row; k < K; k += 64) {
        uint byte_idx = k >> 1;
        uint shift = (k & 1) * 4;
        uint nibble = (packed[row * half_K + byte_idx] >> shift) & 0x0F;
        sum += simd_shuffle(my_lut, nibble) * act[k];
    }

    sum = simd_sum(sum);

    threadgroup float shmem[8];
    if (tiisg == 0) shmem[sgitg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg % 2 == 0 && tiisg == 0) {
        float total = shmem[sgitg] + shmem[sgitg + 1];

        if (oc > 0) {
            uint target = row * K;
            uint row_end = target + K;
            uint lo = 0, hi = oc;
            while (lo < hi) {
                uint mid = (lo + hi) >> 1;
                if (outlier_idx[mid] < target) lo = mid + 1;
                else hi = mid;
            }
            for (uint i = lo; i < oc && outlier_idx[i] < row_end; i++) {
                uint flat = outlier_idx[i];
                uint col = flat - target;
                uint byte_idx = row * half_K + (col >> 1);
                uint shift = (col & 1) * 4;
                uint nibble = (packed[byte_idx] >> shift) & 0x0F;
                total += (outlier_val[i] - simd_shuffle(my_lut, nibble)) * act[col];
            }
        }

        out[row] = total;
    }
}

// ============================================================================
// Branchless SQ4 matvec: register-only dequant via nested select.
// 3 selects for 8-entry band lookup + 1 select for sign = 4 ALU ops.
// Zero memory access for dequant. Same buffer layout as sq4_matvec.
// ============================================================================

kernel void sq4_matvec_branchless(
    device const float* act       [[buffer(0)]],
    device const uchar* packed    [[buffer(1)]],
    device const float* bands     [[buffer(2)]],
    device float* out             [[buffer(3)]],
    device const uint* p_K        [[buffer(4)]],
    device const uint* p_N        [[buffer(5)]],
    device const uint* outlier_idx  [[buffer(6)]],
    device const float* outlier_val [[buffer(7)]],
    device const uint* p_outlier_count [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    uint K = p_K[0], N = p_N[0];
    uint oc = p_outlier_count[0];

    uint row = tgid * 4 + sgitg / 2;
    if (row >= N) return;

    // Load 8 band means into registers (once, at kernel start)
    float b0 = bands[0], b1 = bands[1], b2 = bands[2], b3 = bands[3];
    float b4 = bands[4], b5 = bands[5], b6 = bands[6], b7 = bands[7];

    ushort half_sg = sgitg % 2;
    uint tid_in_row = half_sg * 32 + tiisg;
    uint half_K = K / 2;

    device const uint4* row_packed4 = (device const uint4*)(packed + row * half_K);
    uint n_uint4 = half_K >> 4;

    float sum = 0.0f;
    for (uint i = tid_in_row; i < n_uint4; i += 64) {
        uint4 chunk = row_packed4[i];
        uint base = i * 32;

        uint words[4] = {chunk.x, chunk.y, chunk.z, chunk.w};
        #pragma unroll
        for (int q = 0; q < 4; q++) {
            uint w = words[q];
            uint cb = base + q * 8;
            float4 a0 = *((device const float4*)(act + cb));
            float4 a1 = *((device const float4*)(act + cb + 4));
            float aa[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};

            #pragma unroll
            for (int j = 0; j < 8; j++) {
                uint nib = (w >> (j * 4)) & 0x0F;
                uint band = nib & 0x07;
                float lo2 = select(select(b0, b1, bool(band & 1)), select(b2, b3, bool(band & 1)), bool(band & 2));
                float hi2 = select(select(b4, b5, bool(band & 1)), select(b6, b7, bool(band & 1)), bool(band & 2));
                float val = select(lo2, hi2, bool(band & 4));
                val = select(val, -val, bool(nib & 8));
                sum += val * aa[j];
            }
        }
    }

    // Tail
    uint handled = n_uint4 * 32;
    for (uint k = handled + tid_in_row; k < K; k += 64) {
        uint byte_idx = k >> 1;
        uint shift = (k & 1) * 4;
        uint nib = (packed[row * half_K + byte_idx] >> shift) & 0x0F;
        uint band = nib & 0x07;
        float lo2 = select(select(b0, b1, bool(band & 1)), select(b2, b3, bool(band & 1)), bool(band & 2));
        float hi2 = select(select(b4, b5, bool(band & 1)), select(b6, b7, bool(band & 1)), bool(band & 2));
        float val = select(lo2, hi2, bool(band & 4));
        val = select(val, -val, bool(nib & 8));
        sum += val * act[k];
    }

    sum = simd_sum(sum);

    threadgroup float shmem[8];
    if (tiisg == 0) shmem[sgitg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg % 2 == 0 && tiisg == 0) {
        float total = shmem[sgitg] + shmem[sgitg + 1];

        if (oc > 0) {
            uint target = row * K;
            uint row_end = target + K;
            uint lo = 0, hi = oc;
            while (lo < hi) {
                uint mid = (lo + hi) >> 1;
                if (outlier_idx[mid] < target) lo = mid + 1;
                else hi = mid;
            }
            for (uint i = lo; i < oc && outlier_idx[i] < row_end; i++) {
                uint flat = outlier_idx[i];
                uint col = flat - target;
                uint byte_idx = row * half_K + (col >> 1);
                uint shift = (col & 1) * 4;
                uint nib = (packed[byte_idx] >> shift) & 0x0F;
                uint band = nib & 0x07;
                float lo2 = select(select(b0, b1, bool(band&1)), select(b2, b3, bool(band&1)), bool(band&2));
                float hi2 = select(select(b4, b5, bool(band&1)), select(b6, b7, bool(band&1)), bool(band&2));
                float bval = select(lo2, hi2, bool(band&4));
                bval = select(bval, -bval, bool(nib&8));
                total += (outlier_val[i] - bval) * act[col];
            }
        }

        out[row] = total;
    }
}

// ============================================================================
// Band-tiled SQ4 matvec: weights sorted by band within each row.
// No LUT — band value is a constant per band segment.
// Reads act[col] randomly but act fits in L1 cache (dim=896 → 3.5KB).
//
// Data layout per tensor (prepared at upload):
//   positions[N*K]:    uint16 column indices, band-sorted within each row
//   signs[N*K/8]:      1-bit sign per weight, matching positions[] order
//   band_offsets[9]:   cumulative counts — same for all rows (equal-count bands)
//
// Dispatch: same shape as sq4_matvec — 4 rows/tg, 2 simdgroups/row, 256 threads.
// ============================================================================

kernel void sq4_matvec_tiled(
    device const float* act         [[buffer(0)]],
    device const ushort* positions  [[buffer(1)]],
    device const uchar* signs       [[buffer(2)]],
    device float* out               [[buffer(3)]],
    device const uint* p_K          [[buffer(4)]],
    device const uint* p_N          [[buffer(5)]],
    device const uint* band_offsets [[buffer(6)]],
    device const float* bands       [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    uint K = p_K[0], N = p_N[0];
    uint row = tgid * 4 + sgitg / 2;
    if (row >= N) return;

    ushort half_sg = sgitg % 2;
    uint tid_in_row = half_sg * 32 + tiisg;

    uint row_off = row * K;
    device const ushort* rpos = positions + row_off;
    device const uchar* rsign = signs + row_off / 8;

    float sum = 0.0f;

    for (uint b = 0; b < 8; b++) {
        float bval = bands[b];
        uint bstart = band_offsets[b];
        uint bend = band_offsets[b + 1];

        for (uint i = bstart + tid_in_row; i < bend; i += 64) {
            ushort col = rpos[i];
            uint sign_bit = (rsign[i / 8] >> (i % 8)) & 1;
            float w = sign_bit ? -bval : bval;
            sum += w * act[col];
        }
    }

    sum = simd_sum(sum);
    threadgroup float shmem[8];
    if (tiisg == 0) shmem[sgitg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg % 2 == 0 && tiisg == 0) {
        out[row] = shmem[sgitg] + shmem[sgitg + 1];
    }
}


// ============================================================================
// Fused SQ4 gate+up matvec + SiLU·gate activation.
// Three dispatches → one. Both matvecs read same input, same LUT.
// Output: silu(gate_row · act) * (up_row · act) per row.
//
// Buffer layout: act(0), gate_packed(1), gate_bands(2), up_packed(3),
//   up_bands(4), out(5), p_K(6), p_N(7), gate_oidx(8), gate_oval(9),
//   gate_oc(10), up_oidx(11), up_oval(12), up_oc(13)
//
// Dispatch: same shape as sq4_matvec — 4 rows/tg, 2 simdgroups/row, 256 threads.
// ============================================================================

kernel void sq4_fused_gate_up_silu(
    device const float* act         [[buffer(0)]],
    device const uchar* gate_packed [[buffer(1)]],
    device const float* gate_bands  [[buffer(2)]],
    device const uchar* up_packed   [[buffer(3)]],
    device const float* up_bands    [[buffer(4)]],
    device float* out               [[buffer(5)]],
    device const uint* p_K          [[buffer(6)]],
    device const uint* p_N          [[buffer(7)]],
    device const uint* gate_oidx    [[buffer(8)]],
    device const float* gate_oval   [[buffer(9)]],
    device const uint* p_gate_oc    [[buffer(10)]],
    device const uint* up_oidx      [[buffer(11)]],
    device const float* up_oval     [[buffer(12)]],
    device const uint* p_up_oc      [[buffer(13)]],
    uint tgid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    uint K = p_K[0], N = p_N[0];
    uint gate_oc = p_gate_oc[0], up_oc = p_up_oc[0];

    // Two LUTs: one for gate, one for up
    threadgroup float g_table[16];
    threadgroup float u_table[16];
    if (sgitg == 0 && tiisg < 8) {
        g_table[tiisg] = gate_bands[tiisg];
        g_table[tiisg + 8] = -gate_bands[tiisg];
        u_table[tiisg] = up_bands[tiisg];
        u_table[tiisg + 8] = -up_bands[tiisg];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = tgid * 4 + sgitg / 2;
    if (row >= N) return;

    ushort half_sg = sgitg % 2;
    uint tid_in_row = half_sg * 32 + tiisg;
    uint half_K = K / 2;

    device const uint4* g_row = (device const uint4*)(gate_packed + row * half_K);
    device const uint4* u_row = (device const uint4*)(up_packed + row * half_K);
    uint n_uint4 = half_K >> 4;

    float g_sum = 0.0f, u_sum = 0.0f;
    for (uint i = tid_in_row; i < n_uint4; i += 64) {
        uint4 gc = g_row[i];
        uint4 uc = u_row[i];
        uint base = i * 32;

        // Process 4 uint32 words × 8 nibbles each — gate and up simultaneously
        #define FUSED8(gw, uw, cb) { \
            float4 a0 = *((device const float4*)(act + cb)); \
            float4 a1 = *((device const float4*)(act + cb + 4)); \
            g_sum += g_table[(gw>> 0)&0xF]*a0.x + g_table[(gw>> 4)&0xF]*a0.y \
                   + g_table[(gw>> 8)&0xF]*a0.z + g_table[(gw>>12)&0xF]*a0.w \
                   + g_table[(gw>>16)&0xF]*a1.x + g_table[(gw>>20)&0xF]*a1.y \
                   + g_table[(gw>>24)&0xF]*a1.z + g_table[(gw>>28)&0xF]*a1.w; \
            u_sum += u_table[(uw>> 0)&0xF]*a0.x + u_table[(uw>> 4)&0xF]*a0.y \
                   + u_table[(uw>> 8)&0xF]*a0.z + u_table[(uw>>12)&0xF]*a0.w \
                   + u_table[(uw>>16)&0xF]*a1.x + u_table[(uw>>20)&0xF]*a1.y \
                   + u_table[(uw>>24)&0xF]*a1.z + u_table[(uw>>28)&0xF]*a1.w; \
        }
        FUSED8(gc.x, uc.x, base);
        FUSED8(gc.y, uc.y, base + 8);
        FUSED8(gc.z, uc.z, base + 16);
        FUSED8(gc.w, uc.w, base + 24);
        #undef FUSED8
    }

    // Tail
    uint handled = n_uint4 * 32;
    for (uint k = handled + tid_in_row; k < K; k += 64) {
        uint byte_idx = k >> 1;
        uint shift = (k & 1) * 4;
        uchar g_nib = (gate_packed[row * half_K + byte_idx] >> shift) & 0x0F;
        uchar u_nib = (up_packed[row * half_K + byte_idx] >> shift) & 0x0F;
        float a = act[k];
        g_sum += g_table[g_nib] * a;
        u_sum += u_table[u_nib] * a;
    }

    // Reduce both sums
    g_sum = simd_sum(g_sum);
    u_sum = simd_sum(u_sum);

    threadgroup float g_shmem[8], u_shmem[8];
    if (tiisg == 0) { g_shmem[sgitg] = g_sum; u_shmem[sgitg] = u_sum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg % 2 == 0 && tiisg == 0) {
        float gate_val = g_shmem[sgitg] + g_shmem[sgitg + 1];
        float up_val = u_shmem[sgitg] + u_shmem[sgitg + 1];

        // Outlier correction for gate
        if (gate_oc > 0) {
            uint target = row * K, row_end = target + K;
            uint lo = 0, hi = gate_oc;
            while (lo < hi) { uint mid = (lo+hi)>>1; if (gate_oidx[mid] < target) lo=mid+1; else hi=mid; }
            for (uint i = lo; i < gate_oc && gate_oidx[i] < row_end; i++) {
                uint col = gate_oidx[i] - target;
                uint bi = row * half_K + (col>>1); uint sh = (col&1)*4;
                uchar nib = (gate_packed[bi] >> sh) & 0x0F;
                gate_val += (gate_oval[i] - g_table[nib]) * act[col];
            }
        }
        // Outlier correction for up
        if (up_oc > 0) {
            uint target = row * K, row_end = target + K;
            uint lo = 0, hi = up_oc;
            while (lo < hi) { uint mid = (lo+hi)>>1; if (up_oidx[mid] < target) lo=mid+1; else hi=mid; }
            for (uint i = lo; i < up_oc && up_oidx[i] < row_end; i++) {
                uint col = up_oidx[i] - target;
                uint bi = row * half_K + (col>>1); uint sh = (col&1)*4;
                uchar nib = (up_packed[bi] >> sh) & 0x0F;
                up_val += (up_oval[i] - u_table[nib]) * act[col];
            }
        }

        // SiLU(gate) * up — fused activation
        float silu_gate = gate_val / (1.0f + exp(-gate_val));
        out[row] = silu_gate * up_val;
    }
}

// ============================================================================
// Fused gate+up+SiLU with linear INT4 dequant (no LUT). MLX-style.
// Both matvecs use per-group scale+bias. Output = silu(gate·act) * (up·act).
// 3 dispatches → 1. No threadgroup LUT.
// ============================================================================

kernel void sq4_fused_gate_up_silu_linear(
    device const float* act           [[buffer(0)]],
    device const uchar* gate_packed   [[buffer(1)]],
    device const float* gate_scales   [[buffer(2)]],
    device const float* gate_biases   [[buffer(3)]],
    device const uchar* up_packed     [[buffer(4)]],
    device const float* up_scales     [[buffer(5)]],
    device const float* up_biases     [[buffer(6)]],
    device float* out                 [[buffer(7)]],
    device const uint* p_K            [[buffer(8)]],
    device const uint* p_N            [[buffer(9)]],
    device const uint* gate_oidx      [[buffer(10)]],
    device const float* gate_oval     [[buffer(11)]],
    device const uint* p_gate_oc      [[buffer(12)]],
    device const uint* up_oidx        [[buffer(13)]],
    device const float* up_oval       [[buffer(14)]],
    device const uint* p_up_oc        [[buffer(15)]],
    uint tgid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    uint K = p_K[0], N = p_N[0];
    uint gate_oc = p_gate_oc[0], up_oc = p_up_oc[0];

    uint row = tgid * 4 + sgitg / 2;
    if (row >= N) return;

    ushort half_sg = sgitg % 2;
    uint tid_in_row = half_sg * 32 + tiisg;
    uint half_K = K / 2;
    uint groups_per_row = K / LIN_GROUP_SIZE;

    device const uint16_t* g_row = (device const uint16_t*)(gate_packed + row * half_K);
    device const uint16_t* u_row = (device const uint16_t*)(up_packed + row * half_K);
    device const float* g_sc = gate_scales + row * groups_per_row;
    device const float* g_bi = gate_biases + row * groups_per_row;
    device const float* u_sc = up_scales + row * groups_per_row;
    device const float* u_bi = up_biases + row * groups_per_row;

    uint n_u16 = half_K / 2;

    float g_sum = 0.0f, u_sum = 0.0f;
    for (uint i = tid_in_row; i < n_u16; i += 64) {
        uint16_t gw = g_row[i];
        uint16_t uw = u_row[i];
        uint col_base = i * 4;
        uint group = col_base / LIN_GROUP_SIZE;

        float gs = g_sc[group], gb = g_bi[group];
        float us = u_sc[group], ub = u_bi[group];
        float4 a = *((device const float4*)(act + col_base));

        g_sum += (float(gw & 0x000F) * gs + gb) * a.x
               + (float((gw >> 4) & 0x0F) * gs + gb) * a.y
               + (float((gw >> 8) & 0x0F) * gs + gb) * a.z
               + (float((gw >> 12) & 0x0F) * gs + gb) * a.w;

        u_sum += (float(uw & 0x000F) * us + ub) * a.x
               + (float((uw >> 4) & 0x0F) * us + ub) * a.y
               + (float((uw >> 8) & 0x0F) * us + ub) * a.z
               + (float((uw >> 12) & 0x0F) * us + ub) * a.w;
    }

    g_sum = simd_sum(g_sum);
    u_sum = simd_sum(u_sum);

    threadgroup float g_shmem[8], u_shmem[8];
    if (tiisg == 0) { g_shmem[sgitg] = g_sum; u_shmem[sgitg] = u_sum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg % 2 == 0 && tiisg == 0) {
        float gate_val = g_shmem[sgitg] + g_shmem[sgitg + 1];
        float up_val = u_shmem[sgitg] + u_shmem[sgitg + 1];

        // Outlier corrections
        if (gate_oc > 0) {
            uint target = row * K, row_end = target + K;
            uint lo = 0, hi = gate_oc;
            while (lo < hi) { uint mid = (lo+hi)>>1; if (gate_oidx[mid] < target) lo=mid+1; else hi=mid; }
            for (uint i = lo; i < gate_oc && gate_oidx[i] < row_end; i++) {
                uint col = gate_oidx[i] - target;
                uint byte_idx = col >> 1; uint shift = (col & 1) * 4;
                uint nib = (gate_packed[row * half_K + byte_idx] >> shift) & 0x0F;
                uint group = col / LIN_GROUP_SIZE;
                float approx = float(nib) * g_sc[group] + g_bi[group];
                gate_val += (gate_oval[i] - approx) * act[col];
            }
        }
        if (up_oc > 0) {
            uint target = row * K, row_end = target + K;
            uint lo = 0, hi = up_oc;
            while (lo < hi) { uint mid = (lo+hi)>>1; if (up_oidx[mid] < target) lo=mid+1; else hi=mid; }
            for (uint i = lo; i < up_oc && up_oidx[i] < row_end; i++) {
                uint col = up_oidx[i] - target;
                uint byte_idx = col >> 1; uint shift = (col & 1) * 4;
                uint nib = (up_packed[row * half_K + byte_idx] >> shift) & 0x0F;
                uint group = col / LIN_GROUP_SIZE;
                float approx = float(nib) * u_sc[group] + u_bi[group];
                up_val += (up_oval[i] - approx) * act[col];
            }
        }

        float silu_gate = gate_val / (1.0f + exp(-gate_val));
        out[row] = silu_gate * up_val;
    }
}

// SQ4 outlier correction (for testing — packed nibble format)
kernel void sq4_outlier_correct(
    device const uint* outlier_idx    [[buffer(0)]],
    device const float* outlier_val   [[buffer(1)]],
    device const uchar* packed        [[buffer(2)]],
    device const float* bands         [[buffer(3)]],
    device const float* act           [[buffer(4)]],
    device atomic_float* out          [[buffer(5)]],
    device const uint* p_cols         [[buffer(6)]],
    device const uint* p_count        [[buffer(7)]],
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

// Argmax over logits buffer. Single threadgroup, parallel reduction.
// result[0] = argmax token ID.
kernel void argmax_sample(
    device const float* logits    [[buffer(0)]],
    device uint* result           [[buffer(1)]],
    device const uint* p_N        [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]])
{
    uint N = p_N[0];
    threadgroup float smax[256];
    threadgroup uint sidx[256];

    float best = -1e30f;
    uint bestIdx = 0;
    for (uint i = tid; i < N; i += tpg) {
        float v = logits[i];
        if (v > best) { best = v; bestIdx = i; }
    }
    smax[tid] = best;
    sidx[tid] = bestIdx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (tid < s && smax[tid + s] > smax[tid]) {
            smax[tid] = smax[tid + s];
            sidx[tid] = sidx[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) result[0] = sidx[0];
}
