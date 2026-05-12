// mongoose_bce_loss.cu — numerically stable BCEWithLogitsLoss.
// loss_i = max(x,0) - x*t + log(1 + exp(-|x|))
// grad_i = (sigmoid(x) - t) / B

__global__ void bce_with_logits_kernel(
    const float* logits, const float* targets,
    float* grad, float* loss_buf, int B
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;

    float x = logits[i];
    float t = targets[i];

    float abs_x = (x >= 0) ? x : -x;
    float loss = fmaxf(x, 0.0f) - x * t + logf(1.0f + expf(-abs_x));
    loss_buf[i] = loss;

    float sig = 1.0f / (1.0f + expf(-x));
    grad[i] = (sig - t) / (float)B;
}

void mongoose_bce_with_logits(
    const float* logits, const float* targets,
    float* grad, float* loss_buf, int B, cudaStream_t stream
) {
    int threads = B < 256 ? B : 256;
    int blocks = (B + threads - 1) / threads;
    bce_with_logits_kernel<<<blocks, threads, 0, stream>>>(
        logits, targets, grad, loss_buf, B);
}

// Parallel reduction: out[0] = sum(in[0..n-1]) / n
__global__ void reduce_mean_kernel(const float* in, float* out, int n) {
    __shared__ float shared[256];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        sum += in[i];
    }
    shared[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[0] = shared[0] / (float)n;
}

void mongoose_reduce_mean(const float* in, float* out, int n, cudaStream_t stream) {
    reduce_mean_kernel<<<1, 256, 0, stream>>>(in, out, n);
}
