// mongoose_adamw.cu — adamw kernels
// Auto-extracted from mongoose.cu. Do not edit directly.

// === AdamW on GPU — no CPU round-trip for weight updates ===
// param -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * param)
// m = beta1 * m + (1-beta1) * grad
// v = beta2 * v + (1-beta2) * grad^2
__global__ void adamw_kernel(
    float* param, const float* grad, float* m, float* v,
    float lr, float wd, float beta1, float beta2, float bc1, float bc2, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float g = grad[i];
    float mi = beta1 * m[i] + (1.0f - beta1) * g;
    float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = mi;
    v[i] = vi;

    float mhat = mi / bc1;
    float vhat = vi / bc2;
    param[i] -= lr * (mhat / (sqrtf(vhat) + 1e-8f) + wd * param[i]);
}

void mongoose_adamw(
    float* param, const float* grad, float* m, float* v,
    float lr, float wd, float beta1, float beta2, float bc1, float bc2,
    int n, cudaStream_t stream
) {
    adamw_kernel<<<(n+255)/256, 256, 0, stream>>>(param, grad, m, v, lr, wd, beta1, beta2, bc1, bc2, n);
}

// === Gradient clipping (GPU-only) ===
// Pass 1: accumulate sum-of-squares into a single float via atomicAdd
__global__ void grad_sumsq_kernel(const float* grad, float* sumsq, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = grad[i];
    atomicAdd(sumsq, g * g);
}

// Pass 2: scale all elements by factor if needed
__global__ void grad_scale_kernel(float* grad, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    grad[i] *= scale;
}

void mongoose_grad_sumsq(const float* grad, float* sumsq, int n, cudaStream_t stream) {
    grad_sumsq_kernel<<<(n+255)/256, 256, 0, stream>>>(grad, sumsq, n);
}

void mongoose_grad_scale(float* grad, float scale, int n, cudaStream_t stream) {
    grad_scale_kernel<<<(n+255)/256, 256, 0, stream>>>(grad, scale, n);
}

