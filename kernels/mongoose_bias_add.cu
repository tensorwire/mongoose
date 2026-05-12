// mongoose_bias_add.cu — broadcast bias addition over batch dimension.
// out[b * D + d] += bias[d] for all b in [0, B)

__global__ void bias_add_kernel(float* out, const float* bias, int B, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * D) return;
    int d = idx % D;
    out[idx] += bias[d];
}

void mongoose_bias_add(float* out, const float* bias, int B, int D, cudaStream_t stream) {
    int n = B * D;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    bias_add_kernel<<<blocks, threads, 0, stream>>>(out, bias, B, D);
}

// reduce_mean_rows: out[d] = sum(in[b*D + d], b=0..B-1) / B
// Used for bias gradient: dB = mean(dOut, axis=0)
__global__ void reduce_mean_rows_kernel(const float* in, float* out, int B, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float sum = 0.0f;
    for (int b = 0; b < B; b++) {
        sum += in[b * D + d];
    }
    out[d] = sum / (float)B;
}

void mongoose_reduce_mean_rows(const float* in, float* out, int B, int D, cudaStream_t stream) {
    int threads = D < 256 ? D : 256;
    int blocks = (D + threads - 1) / threads;
    reduce_mean_rows_kernel<<<blocks, threads, 0, stream>>>(in, out, B, D);
}
