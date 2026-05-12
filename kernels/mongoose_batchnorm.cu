// mongoose_batchnorm.cu — BatchNorm forward + backward for MLP training.
// One block per feature dimension, parallel reduction over batch.

// Forward: compute batch mean/var, normalize, scale+shift, update running stats.
// Grid: D blocks, blockDim: min(B, 1024)
__global__ void batchnorm_fwd_kernel(
    const float* __restrict__ x, float* __restrict__ out,
    const float* __restrict__ gamma, const float* __restrict__ beta,
    float* __restrict__ running_mean, float* __restrict__ running_var,
    float* __restrict__ save_mean, float* __restrict__ save_invstd,
    int B, int D, float momentum, float eps
) {
    int d = blockIdx.x;
    if (d >= D) return;

    // Compute mean for this feature
    float sum = 0.0f;
    for (int b = threadIdx.x; b < B; b += blockDim.x) {
        sum += x[b * D + d];
    }
    // Warp reduce
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    if (lane == 0) shared[wid] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        int nWarps = (blockDim.x + warpSize - 1) / warpSize;
        float total = 0.0f;
        for (int i = 0; i < nWarps; i++) total += shared[i];
        shared[0] = total;
    }
    __syncthreads();
    float mean = shared[0] / (float)B;

    // Compute variance
    float var_sum = 0.0f;
    for (int b = threadIdx.x; b < B; b += blockDim.x) {
        float diff = x[b * D + d] - mean;
        var_sum += diff * diff;
    }
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    if (lane == 0) shared[wid] = var_sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        int nWarps = (blockDim.x + warpSize - 1) / warpSize;
        float total = 0.0f;
        for (int i = 0; i < nWarps; i++) total += shared[i];
        shared[0] = total;
    }
    __syncthreads();
    float variance = shared[0] / (float)B;

    float invstd = rsqrtf(variance + eps);

    // Save for backward
    if (threadIdx.x == 0) {
        save_mean[d] = mean;
        save_invstd[d] = invstd;
        running_mean[d] = (1.0f - momentum) * running_mean[d] + momentum * mean;
        running_var[d] = (1.0f - momentum) * running_var[d] + momentum * variance;
    }
    __syncthreads();

    // Normalize + scale + shift
    float g = gamma[d];
    float b_val = beta[d];
    for (int b = threadIdx.x; b < B; b += blockDim.x) {
        float xhat = (x[b * D + d] - mean) * invstd;
        out[b * D + d] = g * xhat + b_val;
    }
}

void mongoose_batchnorm_fwd(
    const float* x, float* out,
    const float* gamma, const float* beta,
    float* running_mean, float* running_var,
    float* save_mean, float* save_invstd,
    int B, int D, float momentum, float eps, cudaStream_t stream
) {
    int threads = B < 1024 ? ((B + 31) / 32) * 32 : 1024;
    if (threads < 32) threads = 32;
    batchnorm_fwd_kernel<<<D, threads, 0, stream>>>(
        x, out, gamma, beta, running_mean, running_var,
        save_mean, save_invstd, B, D, momentum, eps);
}

// Backward: compute dx, dGamma, dBeta
// Grid: D blocks, blockDim: min(B, 1024)
__global__ void batchnorm_bwd_kernel(
    const float* __restrict__ dOut, const float* __restrict__ x,
    const float* __restrict__ save_mean, const float* __restrict__ save_invstd,
    const float* __restrict__ gamma,
    float* __restrict__ dx, float* __restrict__ dGamma, float* __restrict__ dBeta,
    int B, int D
) {
    int d = blockIdx.x;
    if (d >= D) return;

    float mean = save_mean[d];
    float invstd = save_invstd[d];
    float g = gamma[d];

    // Compute dGamma = sum(dOut * xhat) and dBeta = sum(dOut)
    float dgamma_sum = 0.0f;
    float dbeta_sum = 0.0f;
    for (int b = threadIdx.x; b < B; b += blockDim.x) {
        float xhat = (x[b * D + d] - mean) * invstd;
        dgamma_sum += dOut[b * D + d] * xhat;
        dbeta_sum += dOut[b * D + d];
    }

    // Reduce
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        dgamma_sum += __shfl_down_sync(0xffffffff, dgamma_sum, offset);
        dbeta_sum += __shfl_down_sync(0xffffffff, dbeta_sum, offset);
    }
    __shared__ float s_dgamma[32], s_dbeta[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    if (lane == 0) { s_dgamma[wid] = dgamma_sum; s_dbeta[wid] = dbeta_sum; }
    __syncthreads();
    if (threadIdx.x == 0) {
        int nWarps = (blockDim.x + warpSize - 1) / warpSize;
        float tg = 0, tb = 0;
        for (int i = 0; i < nWarps; i++) { tg += s_dgamma[i]; tb += s_dbeta[i]; }
        s_dgamma[0] = tg;
        s_dbeta[0] = tb;
        dGamma[d] = tg / (float)B;
        dBeta[d] = tb / (float)B;
    }
    __syncthreads();

    float dg = s_dgamma[0];
    float db = s_dbeta[0];
    float invB = 1.0f / (float)B;

    // dx = invstd * (gamma * dOut - invB * (dBeta + xhat * dGamma))
    for (int b = threadIdx.x; b < B; b += blockDim.x) {
        float xhat = (x[b * D + d] - mean) * invstd;
        float dxhat = g * dOut[b * D + d];
        dx[b * D + d] = invstd * (dxhat - invB * (db * g + dg * g * xhat));
    }
}

void mongoose_batchnorm_bwd(
    const float* dOut, const float* x,
    const float* save_mean, const float* save_invstd,
    const float* gamma,
    float* dx, float* dGamma, float* dBeta,
    int B, int D, cudaStream_t stream
) {
    int threads = B < 1024 ? ((B + 31) / 32) * 32 : 1024;
    if (threads < 32) threads = 32;
    batchnorm_bwd_kernel<<<D, threads, 0, stream>>>(
        dOut, x, save_mean, save_invstd, gamma, dx, dGamma, dBeta, B, D);
}
