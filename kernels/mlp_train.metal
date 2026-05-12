#include <metal_stdlib>
using namespace metal;

// MLP training kernels — self-contained, no dependencies on the main mongoose kernel library.
// Includes matmul, ReLU, AdamW, BatchNorm, BCE loss, dropout, and utility ops.
// Designed for multi-dispatch pattern: one command buffer, many dispatches.

struct MLPConstants {
    uint B;       // batch size
    uint D;       // feature dimension for this layer
    float momentum;
    float eps;
    float dropoutP;
    uint  dropSeed;
    uint  dropCounter;
};

// ============================================================
// Bias add: out[b*D + d] += bias[d]
// Dispatch: threads = B * D
// ============================================================
kernel void mlp_bias_add(
    device float* out        [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    constant uint& B         [[buffer(2)]],
    constant uint& D         [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= B * D) return;
    uint d = id % D;
    out[id] += bias[d];
}

// ============================================================
// BatchNorm forward — pass 1: compute per-feature sum
// One threadgroup per feature, threads iterate over batch.
// Output: sum[D] (partial sums accumulated via threadgroup reduction)
// Dispatch: threadgroups = D, threads_per_threadgroup = min(B, 1024)
// ============================================================
kernel void mlp_bn_sum(
    device const float* x     [[buffer(0)]],  // [B, D]
    device float* sumOut      [[buffer(1)]],   // [D]
    constant uint& B          [[buffer(2)]],
    constant uint& D          [[buffer(3)]],
    uint d   [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]])
{
    if (d >= D) return;
    float sum = 0.0f;
    for (uint b = tid; b < B; b += tpg) {
        sum += x[b * D + d];
    }
    sum = simd_sum(sum);
    threadgroup float shared[32];
    uint lane = tid % 32;
    uint warp = tid / 32;
    if (lane == 0) shared[warp] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (warp == 0) {
        sum = (lane < (tpg + 31) / 32) ? shared[lane] : 0.0f;
        sum = simd_sum(sum);
    }
    if (tid == 0) {
        sumOut[d] = sum / float(B);
    }
}

// ============================================================
// BatchNorm forward — pass 2: compute per-feature variance
// Reads mean from sumOut (computed by pass 1).
// Output: varOut[D]
// Dispatch: threadgroups = D, threads_per_threadgroup = min(B, 1024)
// ============================================================
kernel void mlp_bn_var(
    device const float* x      [[buffer(0)]],  // [B, D]
    device const float* mean   [[buffer(1)]],   // [D] (from pass 1)
    device float* varOut       [[buffer(2)]],   // [D]
    constant uint& B           [[buffer(3)]],
    constant uint& D           [[buffer(4)]],
    uint d   [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]])
{
    if (d >= D) return;
    float m = mean[d];
    float sumSq = 0.0f;
    for (uint b = tid; b < B; b += tpg) {
        float diff = x[b * D + d] - m;
        sumSq += diff * diff;
    }
    sumSq = simd_sum(sumSq);
    threadgroup float shared[32];
    uint lane = tid % 32;
    uint warp = tid / 32;
    if (lane == 0) shared[warp] = sumSq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (warp == 0) {
        sumSq = (lane < (tpg + 31) / 32) ? shared[lane] : 0.0f;
        sumSq = simd_sum(sumSq);
    }
    if (tid == 0) {
        varOut[d] = sumSq / float(B);
    }
}

// ============================================================
// BatchNorm forward — pass 3: normalize + scale + shift + update running stats
// Dispatch: threads = B * D
// ============================================================
kernel void mlp_bn_norm(
    device float* x              [[buffer(0)]],   // [B, D] — modified in-place
    device const float* mean     [[buffer(1)]],   // [D]
    device const float* var      [[buffer(2)]],   // [D]
    device const float* gamma    [[buffer(3)]],   // [D]
    device const float* beta     [[buffer(4)]],   // [D]
    device float* runMean        [[buffer(5)]],   // [D] — updated
    device float* runVar         [[buffer(6)]],   // [D] — updated
    constant uint& B             [[buffer(7)]],
    constant uint& D             [[buffer(8)]],
    constant float& momentum     [[buffer(9)]],
    constant float& eps          [[buffer(10)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= B * D) return;
    uint d = id % D;
    uint b = id / D;
    float m = mean[d];
    float v = var[d];
    float invstd = rsqrt(v + eps);
    float xhat = (x[id] - m) * invstd;
    x[id] = gamma[d] * xhat + beta[d];
    // Update running stats (only first batch row does this to avoid races)
    if (b == 0) {
        runMean[d] = (1.0f - momentum) * runMean[d] + momentum * m;
        runVar[d] = (1.0f - momentum) * runVar[d] + momentum * v;
    }
}

// ============================================================
// BatchNorm backward — pass 1: compute dGamma and dBeta
// One threadgroup per feature.
// Dispatch: threadgroups = D, threads_per_threadgroup = min(B, 1024)
// ============================================================
kernel void mlp_bn_bwd_stats(
    device const float* dOut     [[buffer(0)]],  // [B, D]
    device const float* x        [[buffer(1)]],  // [B, D] pre-BN input
    device const float* mean     [[buffer(2)]],  // [D]
    device const float* var      [[buffer(3)]],  // [D]
    device float* dGamma         [[buffer(4)]],  // [D]
    device float* dBeta          [[buffer(5)]],  // [D]
    constant uint& B             [[buffer(6)]],
    constant uint& D             [[buffer(7)]],
    uint d   [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]])
{
    if (d >= D) return;
    float m = mean[d];
    float v = var[d];
    float invstd = rsqrt(v + 1e-5f);
    float dgSum = 0.0f, dbSum = 0.0f;
    for (uint b = tid; b < B; b += tpg) {
        float xhat = (x[b * D + d] - m) * invstd;
        dgSum += dOut[b * D + d] * xhat;
        dbSum += dOut[b * D + d];
    }
    dgSum = simd_sum(dgSum);
    dbSum = simd_sum(dbSum);
    threadgroup float sg[32], sb[32];
    uint lane = tid % 32;
    uint warp = tid / 32;
    if (lane == 0) { sg[warp] = dgSum; sb[warp] = dbSum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (warp == 0) {
        dgSum = (lane < (tpg + 31) / 32) ? sg[lane] : 0.0f;
        dbSum = (lane < (tpg + 31) / 32) ? sb[lane] : 0.0f;
        dgSum = simd_sum(dgSum);
        dbSum = simd_sum(dbSum);
    }
    if (tid == 0) {
        dGamma[d] = dgSum / float(B);
        dBeta[d] = dbSum / float(B);
    }
}

// ============================================================
// BatchNorm backward — pass 2: compute dx
// Dispatch: threads = B * D
// ============================================================
kernel void mlp_bn_bwd_dx(
    device float* dOut           [[buffer(0)]],  // [B, D] — modified in-place to dx
    device const float* x        [[buffer(1)]],  // [B, D] pre-BN input
    device const float* mean     [[buffer(2)]],  // [D]
    device const float* var      [[buffer(3)]],  // [D]
    device const float* gamma    [[buffer(4)]],  // [D]
    device const float* dGamma   [[buffer(5)]],  // [D]
    device const float* dBeta    [[buffer(6)]],  // [D]
    constant uint& B             [[buffer(7)]],
    constant uint& D             [[buffer(8)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= B * D) return;
    uint d = id % D;
    float m = mean[d];
    float v = var[d];
    float invstd = rsqrt(v + 1e-5f);
    float g = gamma[d];
    float dg = dGamma[d] * float(B);  // undo /B to get raw sum
    float db = dBeta[d] * float(B);
    float invB = 1.0f / float(B);
    float xhat = (x[id] - m) * invstd;
    float dxhat = g * dOut[id];
    dOut[id] = invstd * (dxhat - invB * (db * g + dg * g * xhat));
}

// ============================================================
// BCE with logits loss + gradient
// loss_i = max(x,0) - x*t + log(1 + exp(-|x|))
// grad_i = (sigmoid(x) - t) / B
// Dispatch: threads = B
// ============================================================
kernel void mlp_bce_with_logits(
    device const float* logits   [[buffer(0)]],  // [B]
    device const float* targets  [[buffer(1)]],  // [B]
    device float* grad           [[buffer(2)]],  // [B]
    device float* lossBuf        [[buffer(3)]],  // [B] per-sample loss
    constant uint& B             [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= B) return;
    float x = logits[id];
    float t = targets[id];
    float absX = abs(x);
    lossBuf[id] = max(x, 0.0f) - x * t + log(1.0f + exp(-absX));
    float sig = 1.0f / (1.0f + exp(-x));
    grad[id] = (sig - t) / float(B);
}

// ============================================================
// Reduce mean: out[0] = sum(in[0..n-1]) / n
// Single threadgroup reduction.
// Dispatch: threadgroups = 1, threads = min(n, 1024)
// ============================================================
kernel void mlp_reduce_mean(
    device const float* in  [[buffer(0)]],
    device float* out       [[buffer(1)]],
    constant uint& n        [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]])
{
    float sum = 0.0f;
    for (uint i = tid; i < n; i += tpg) {
        sum += in[i];
    }
    sum = simd_sum(sum);
    threadgroup float shared[32];
    uint lane = tid % 32;
    uint warp = tid / 32;
    if (lane == 0) shared[warp] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (warp == 0) {
        sum = (lane < (tpg + 31) / 32) ? shared[lane] : 0.0f;
        sum = simd_sum(sum);
    }
    if (tid == 0) {
        out[0] = sum / float(n);
    }
}

// ============================================================
// Reduce mean rows: out[d] = sum(in[b*D + d], b=0..B-1) / B
// For bias gradient.
// Dispatch: threadgroups = D, threads = min(B, 1024)
// ============================================================
kernel void mlp_reduce_mean_rows(
    device const float* in  [[buffer(0)]],  // [B, D]
    device float* out       [[buffer(1)]],  // [D]
    constant uint& B        [[buffer(2)]],
    constant uint& D        [[buffer(3)]],
    uint d   [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]])
{
    if (d >= D) return;
    float sum = 0.0f;
    for (uint b = tid; b < B; b += tpg) {
        sum += in[b * D + d];
    }
    sum = simd_sum(sum);
    threadgroup float shared[32];
    uint lane = tid % 32;
    uint warp = tid / 32;
    if (lane == 0) shared[warp] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (warp == 0) {
        sum = (lane < (tpg + 31) / 32) ? shared[lane] : 0.0f;
        sum = simd_sum(sum);
    }
    if (tid == 0) {
        out[d] = sum / float(B);
    }
}

// ============================================================
// Dropout forward (Philox-like via metal hash)
// In-place: x[i] = (rand > p) ? x[i] * scale : 0
// mask[i] = (rand > p) ? scale : 0
// Dispatch: threads = n
// ============================================================
kernel void mlp_dropout_fwd(
    device float* x          [[buffer(0)]],  // [n] — modified in-place
    device float* mask       [[buffer(1)]],  // [n] — saved for backward
    constant uint& n         [[buffer(2)]],
    constant float& p        [[buffer(3)]],  // dropout probability
    constant uint& seed      [[buffer(4)]],
    constant uint& counter   [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= n) return;
    // Hash-based PRNG (Wang hash, deterministic given id + counter + seed)
    uint h = id ^ seed;
    h ^= counter;
    h = (h ^ 61u) ^ (h >> 16u);
    h *= 9u;
    h ^= h >> 4u;
    h *= 0x27d4eb2du;
    h ^= h >> 15u;
    float r = float(h) / float(0xFFFFFFFFu);
    float scale = 1.0f / (1.0f - p);
    if (r < p) {
        x[id] = 0.0f;
        mask[id] = 0.0f;
    } else {
        x[id] *= scale;
        mask[id] = scale;
    }
}

// ============================================================
// Dropout backward: dx[i] *= mask[i]
// Dispatch: threads = n
// ============================================================
kernel void mlp_dropout_bwd(
    device float* dx             [[buffer(0)]],
    device const float* mask     [[buffer(1)]],
    constant uint& n             [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= n) return;
    dx[id] *= mask[id];
}

// ============================================================
// Copy buffer: dst[i] = src[i]
// Dispatch: threads = n
// ============================================================
kernel void mlp_copy(
    device float* dst            [[buffer(0)]],
    device const float* src      [[buffer(1)]],
    constant uint& n             [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= n) return;
    dst[id] = src[id];
}

// ============================================================
// Zero memory
// ============================================================
kernel void mlp_zero(
    device float* ptr [[buffer(0)]],
    uint id [[thread_position_in_grid]])
{
    ptr[id] = 0.0f;
}

// ============================================================
// Scale in-place: a[i] *= s[0]
// ============================================================
kernel void mlp_scale_inplace(
    device float* a [[buffer(0)]],
    device const float* s [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    a[id] *= s[0];
}

// ============================================================
// ReLU in-place
// ============================================================
kernel void mlp_relu_inplace(
    device float* x [[buffer(0)]],
    uint id [[thread_position_in_grid]])
{
    x[id] = max(x[id], 0.0f);
}

// ============================================================
// ReLU backward: result[i] = (fwd[i] > 0) ? dOut[i] : 0
// ============================================================
kernel void mlp_relu_backward(
    device float* result         [[buffer(0)]],
    device const float* dOut     [[buffer(1)]],
    device const float* fwd      [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    result[id] = (fwd[id] > 0.0f) ? dOut[id] : 0.0f;
}

// ============================================================
// AdamW optimizer — one thread per parameter
// ============================================================
kernel void mlp_adamw(
    device float* param          [[buffer(0)]],
    device const float* grad     [[buffer(1)]],
    device float* m              [[buffer(2)]],
    device float* v              [[buffer(3)]],
    constant float& lr           [[buffer(4)]],
    constant float& beta1        [[buffer(5)]],
    constant float& beta2        [[buffer(6)]],
    constant float& bc1          [[buffer(7)]],
    constant float& bc2          [[buffer(8)]],
    constant float& eps          [[buffer(9)]],
    constant float& wd           [[buffer(10)]],
    uint id [[thread_position_in_grid]])
{
    float g = grad[id];
    float m_new = beta1 * m[id] + (1.0f - beta1) * g;
    float v_new = beta2 * v[id] + (1.0f - beta2) * g * g;
    m[id] = m_new;
    v[id] = v_new;
    float mh = m_new / bc1;
    float vh = v_new / bc2;
    param[id] -= lr * (mh / (sqrt(vh) + eps) + wd * param[id]);
}

// ============================================================
// GEMM: C = A @ B^T — tiled 32x32
// A[M, K], B[N, K] (B stored row-major, transposed access)
// ============================================================
kernel void mlp_gemm_bt(
    device const float* A   [[buffer(0)]],
    device const float* B   [[buffer(1)]],
    device float* C         [[buffer(2)]],
    constant uint& M        [[buffer(3)]],
    constant uint& K        [[buffer(4)]],
    constant uint& N        [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid  [[thread_position_in_threadgroup]])
{
    const uint TILE = 32;
    uint row = tgid.y * TILE + tid.y;
    uint col = tgid.x * TILE + tid.x;

    threadgroup float tileA[32][32];
    threadgroup float tileB[32][32];

    float sum = 0.0f;
    for (uint t = 0; t < (K + TILE - 1) / TILE; t++) {
        uint ak = t * TILE + tid.x;
        uint bk = t * TILE + tid.y;
        tileA[tid.y][tid.x] = (row < M && ak < K) ? A[row * K + ak] : 0.0f;
        tileB[tid.y][tid.x] = (col < N && bk < K) ? B[col * K + bk] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < TILE; k++) {
            sum += tileA[tid.y][k] * tileB[k][tid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================
// GEMM: C = A^T @ B — for weight gradient dW = dOut^T @ input
// A[M, K] (transposed to [K, M]), B[M, N] → C[K, N]
// ============================================================
kernel void mlp_gemm_tn(
    device const float* A   [[buffer(0)]],
    device const float* B   [[buffer(1)]],
    device float* C         [[buffer(2)]],
    constant uint& M        [[buffer(3)]],
    constant uint& K        [[buffer(4)]],
    constant uint& N        [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid  [[thread_position_in_threadgroup]])
{
    const uint TILE = 32;
    uint row = tgid.y * TILE + tid.y;  // row in C = column in A
    uint col = tgid.x * TILE + tid.x;  // col in C = col in B

    threadgroup float tileA[32][32];
    threadgroup float tileB[32][32];

    float sum = 0.0f;
    for (uint t = 0; t < (M + TILE - 1) / TILE; t++) {
        uint am = t * TILE + tid.x;  // index into M dimension
        uint bm = t * TILE + tid.y;
        // A^T[row, am] = A[am, row]
        tileA[tid.y][tid.x] = (row < K && am < M) ? A[am * K + row] : 0.0f;
        tileB[tid.y][tid.x] = (col < N && bm < M) ? B[bm * N + col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < TILE; k++) {
            sum += tileA[tid.y][k] * tileB[k][tid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < K && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================
// GEMM: C = A @ B — for input gradient dInput = dOut @ W
// A[M, K], B[K, N] → C[M, N]
// ============================================================
kernel void mlp_gemm_nn(
    device const float* A   [[buffer(0)]],
    device const float* B   [[buffer(1)]],
    device float* C         [[buffer(2)]],
    constant uint& M        [[buffer(3)]],
    constant uint& K        [[buffer(4)]],
    constant uint& N        [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid  [[thread_position_in_threadgroup]])
{
    const uint TILE = 32;
    uint row = tgid.y * TILE + tid.y;
    uint col = tgid.x * TILE + tid.x;

    threadgroup float tileA[32][32];
    threadgroup float tileB[32][32];

    float sum = 0.0f;
    for (uint t = 0; t < (K + TILE - 1) / TILE; t++) {
        uint ak = t * TILE + tid.x;
        uint bk = t * TILE + tid.y;
        tileA[tid.y][tid.x] = (row < M && ak < K) ? A[row * K + ak] : 0.0f;
        tileB[tid.y][tid.x] = (bk < K && col < N) ? B[bk * N + col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < TILE; k++) {
            sum += tileA[tid.y][k] * tileB[k][tid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
