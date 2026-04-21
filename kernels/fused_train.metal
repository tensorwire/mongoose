#include <metal_stdlib>
using namespace metal;

// Fused training kernels — forward + backward in minimal dispatches.
// Pattern from tree_train_full.metal: struct constants, shared memory,
// one threadgroup per position, dim threads per threadgroup.

struct TrainLayerConstants {
    uint dim;
    uint kvDim;
    uint headDim;
    uint nHeads;
    uint nKVHeads;
    uint ffnDim;
    uint seqLen;
    float ropeTheta;
    float eps;
};

// ============================================================
// Fused pre-attention: RMSNorm → Q/K/V GEMM → RoPE
// One threadgroup per position, dim threads.
// Reads hidden[seqLen, dim], writes Q[seqLen, dim], K[seqLen, kvDim], V[seqLen, kvDim].
// Also saves normed[seqLen, dim] and rmsScale[seqLen] for backward.
// ============================================================
kernel void fused_pre_attn(
    device float* hidden          [[buffer(0)]],   // [seqLen, dim] — read (not modified)
    device const float* normW     [[buffer(1)]],   // [dim] RMSNorm weight
    device const float* wq        [[buffer(2)]],   // [dim, dim] row-major, transposed access
    device const float* wk        [[buffer(3)]],   // [kvDim, dim]
    device const float* wv        [[buffer(4)]],   // [kvDim, dim]
    device float* Q               [[buffer(5)]],   // [seqLen, dim] output
    device float* K               [[buffer(6)]],   // [seqLen, kvDim] output
    device float* V               [[buffer(7)]],   // [seqLen, kvDim] output
    device float* normedOut       [[buffer(8)]],   // [seqLen, dim] saved for backward
    device float* rmsScale        [[buffer(9)]],   // [seqLen] saved for backward
    device float* xIn             [[buffer(10)]],  // [seqLen, dim] save pre-norm hidden for backward
    constant TrainLayerConstants& C [[buffer(11)]],
    uint pos [[threadgroup_position_in_grid]],
    uint col [[thread_index_in_threadgroup]])
{
    if (pos >= C.seqLen || col >= C.dim) return;

    uint dim = C.dim;
    uint kvDim = C.kvDim;
    uint headDim = C.headDim;
    uint hOff = pos * dim;

    // Save pre-norm hidden
    float x = hidden[hOff + col];
    xIn[hOff + col] = x;

    // === RMSNorm ===
    // Sum of squares (cooperative reduction via shared memory)
    threadgroup float shared[1024];
    shared[col] = x * x;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction
    for (uint s = dim / 2; s > 0; s >>= 1) {
        if (col < s) shared[col] += shared[col + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float scale = 0.0f;
    if (col == 0) {
        scale = rsqrt(shared[0] / float(dim) + C.eps);
        rmsScale[pos] = scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float normed = x * scale * normW[col];
    normedOut[hOff + col] = normed;

    // Restore hidden (RMSNorm was in-place in the per-op version, here we don't modify hidden)
    // hidden stays unchanged; the per-op version modified it and then copied back.
    // In fused mode, we read hidden, compute normed separately.

    // === Q GEMM: Q[pos, col] = sum_k(normed[pos, k] * wq[col, k]) ===
    // wq is [dim, dim] row-major. wq[col, k] = wq[col * dim + k].
    // We need all normed values — load into shared memory.
    shared[col] = normed;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (col < dim) {
        float dot = 0.0f;
        for (uint k = 0; k < dim; k++) {
            dot += shared[k] * wq[col * dim + k];
        }
        Q[hOff + col] = dot;
    }

    // === K GEMM: K[pos, col] = sum_k(normed[pos, k] * wk[col, k]) for col < kvDim ===
    if (col < kvDim) {
        float dot = 0.0f;
        for (uint k = 0; k < dim; k++) {
            dot += shared[k] * wk[col * dim + k];
        }
        K[pos * kvDim + col] = dot;
    }

    // === V GEMM: same pattern ===
    if (col < kvDim) {
        float dot = 0.0f;
        for (uint k = 0; k < dim; k++) {
            dot += shared[k] * wv[col * dim + k];
        }
        V[pos * kvDim + col] = dot;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // === RoPE on Q ===
    // Q[pos, h*headDim + 2j]   = Q*cos - Q_next*sin
    // Q[pos, h*headDim + 2j+1] = Q*sin + Q_next*cos
    if (col < dim) {
        uint h = col / headDim;
        uint j = col % headDim;
        if (j % 2 == 0 && j + 1 < headDim) {
            float freq = 1.0f / pow(C.ropeTheta, float(j) / float(headDim));
            float angle = float(pos) * freq;
            float cosA = cos(angle), sinA = sin(angle);
            float q0 = Q[hOff + h * headDim + j];
            float q1 = Q[hOff + h * headDim + j + 1];
            Q[hOff + h * headDim + j]     = q0 * cosA - q1 * sinA;
            Q[hOff + h * headDim + j + 1] = q0 * sinA + q1 * cosA;
        }
    }

    // === RoPE on K ===
    if (col < kvDim) {
        uint nKVH = C.nKVHeads;
        uint h = col / headDim;
        uint j = col % headDim;
        if (j % 2 == 0 && j + 1 < headDim && h < nKVH) {
            float freq = 1.0f / pow(C.ropeTheta, float(j) / float(headDim));
            float angle = float(pos) * freq;
            float cosA = cos(angle), sinA = sin(angle);
            float k0 = K[pos * kvDim + h * headDim + j];
            float k1 = K[pos * kvDim + h * headDim + j + 1];
            K[pos * kvDim + h * headDim + j]     = k0 * cosA - k1 * sinA;
            K[pos * kvDim + h * headDim + j + 1] = k0 * sinA + k1 * cosA;
        }
    }
}

// ============================================================
// Fused post-attention: WO GEMM → Residual → RMSNorm → Gate/Up GEMM → SiLU → Down GEMM → Residual
// One threadgroup per position, max(dim, ffnDim) threads.
// ============================================================
kernel void fused_post_attn(
    device float* hidden          [[buffer(0)]],   // [seqLen, dim] — modified in place (residual adds)
    device const float* attnOut   [[buffer(1)]],   // [seqLen, dim] from attention
    device const float* wo        [[buffer(2)]],   // [dim, dim]
    device const float* normW2    [[buffer(3)]],   // [dim] post-attn RMSNorm weight
    device const float* gate      [[buffer(4)]],   // [ffnDim, dim]
    device const float* up        [[buffer(5)]],   // [ffnDim, dim]
    device const float* down      [[buffer(6)]],   // [dim, ffnDim]
    device float* xMid            [[buffer(7)]],   // [seqLen, dim] save for backward
    device float* normed2         [[buffer(8)]],   // [seqLen, dim] save for backward
    device float* rmsScale2       [[buffer(9)]],   // [seqLen] save for backward
    device float* gatePre         [[buffer(10)]],  // [seqLen, ffnDim] save for backward
    device float* upOut           [[buffer(11)]],  // [seqLen, ffnDim] save for backward
    device float* ffnMid          [[buffer(12)]],  // [seqLen, ffnDim] save for backward
    constant TrainLayerConstants& C [[buffer(13)]],
    uint pos [[threadgroup_position_in_grid]],
    uint col [[thread_index_in_threadgroup]])
{
    uint dim = C.dim;
    uint ffnDim = C.ffnDim;
    if (pos >= C.seqLen) return;

    uint hOff = pos * dim;
    uint fOff = pos * ffnDim;

    // === WO GEMM + Residual ===
    // attnOut[pos] is ready. Compute wo_out[pos, col] = sum_k(attnOut[pos,k] * wo[col,k])
    threadgroup float shared[1024];

    // Load attnOut into shared for the matmul
    if (col < dim) shared[col] = attnOut[hOff + col];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (col < dim) {
        float dot = 0.0f;
        for (uint k = 0; k < dim; k++) {
            dot += shared[k] * wo[col * dim + k];
        }
        hidden[hOff + col] += dot;  // residual add
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Save xMid (pre-norm hidden for backward)
    if (col < dim) xMid[hOff + col] = hidden[hOff + col];

    // === RMSNorm 2 ===
    if (col < dim) shared[col] = hidden[hOff + col] * hidden[hOff + col];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = dim / 2; s > 0; s >>= 1) {
        if (col < s) shared[col] += shared[col + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float scale2 = 0.0f;
    if (col == 0) {
        scale2 = rsqrt(shared[0] / float(dim) + C.eps);
        rmsScale2[pos] = scale2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float n2 = 0.0f;
    if (col < dim) {
        n2 = hidden[hOff + col] * scale2 * normW2[col];
        normed2[hOff + col] = n2;
    }

    // Restore hidden to pre-norm state for backward
    // (In the per-op path, hidden was modified by RMSNorm then restored from xMid.
    //  Here we save xMid above and don't modify hidden — we compute n2 separately.
    //  But we DID modify hidden with the residual add above. hidden now = xMid.)

    // === Gate GEMM: gatePre[pos, col] = sum_k(n2[k] * gate[col, k]) for col < ffnDim ===
    // Need normed2 in shared memory
    if (col < dim) shared[col] = n2;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (col < ffnDim) {
        float dot = 0.0f;
        for (uint k = 0; k < dim; k++) {
            dot += shared[k] * gate[col * dim + k];
        }
        gatePre[fOff + col] = dot;
    }

    // === Up GEMM ===
    if (col < ffnDim) {
        float dot = 0.0f;
        for (uint k = 0; k < dim; k++) {
            dot += shared[k] * up[col * dim + k];
        }
        upOut[fOff + col] = dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // === SiLU gate mul ===
    if (col < ffnDim) {
        float g = gatePre[fOff + col];
        float silu = g / (1.0f + exp(-g));
        ffnMid[fOff + col] = silu * upOut[fOff + col];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // === Down GEMM + Residual ===
    // down is [dim, ffnDim]. out[col] = sum_k(ffnMid[k] * down[col, k])
    // Need ffnMid in shared memory — but ffnDim might be > dim. Reuse shared.
    if (col < ffnDim) shared[col] = ffnMid[fOff + col];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (col < dim) {
        float dot = 0.0f;
        for (uint k = 0; k < ffnDim; k++) {
            dot += shared[k] * down[col * ffnDim + k];
        }
        hidden[hOff + col] += dot;  // residual add
    }
}
