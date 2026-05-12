// mongoose_dropout.cu — Philox counter-based dropout (CUDA graph safe).
// Uses curand_kernel.h device API — no host-side state, deterministic given (seed, counter, tid).
// NOTE: curand_kernel.h must be included BEFORE extern "C" in the parent .cu file.

// Forward: apply inverted dropout.
// out[i] = (rand > p) ? x[i] / (1-p) : 0
// mask[i] = (rand > p) ? 1/(1-p) : 0  (saved for backward)
__global__ void dropout_fwd_kernel(
    float* __restrict__ x, float* __restrict__ mask,
    unsigned long long seed, unsigned long long counter,
    float p, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, i, counter, &state);
    float r = curand_uniform(&state);

    if (r < p) {
        x[i] = 0.0f;
        mask[i] = 0.0f;
    } else {
        float scale = 1.0f / (1.0f - p);
        x[i] *= scale;
        mask[i] = scale;
    }
}

void mongoose_dropout_fwd(
    float* x, float* mask,
    unsigned long long seed, unsigned long long counter,
    float p, int n, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    dropout_fwd_kernel<<<blocks, threads, 0, stream>>>(x, mask, seed, counter, p, n);
}

// Backward: dx[i] = dOut[i] * mask[i]
// (element-wise multiply — could use existing KScaleByWeight but this avoids parameter confusion)
__global__ void dropout_bwd_kernel(
    float* __restrict__ dx, const float* __restrict__ mask, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dx[i] *= mask[i];
}

void mongoose_dropout_bwd(float* dx, const float* mask, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    dropout_bwd_kernel<<<blocks, threads, 0, stream>>>(dx, mask, n);
}
