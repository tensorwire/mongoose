// mongoose_kv.cu — kv kernels
// Auto-extracted from mongoose.cu. Do not edit directly.

// === KV cache write: cache[pos*kvDim .. (pos+1)*kvDim] = src ===
__global__ void kv_cache_write_kernel(float* cache, const float* src, int pos, int kvDim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kvDim) cache[pos * kvDim + i] = src[i];
}

void mongoose_kv_cache_write(float* cache, const float* src, int pos, int kvDim, cudaStream_t stream) {
    int threads = kvDim < 256 ? kvDim : 256;
    int blocks = (kvDim + threads - 1) / threads;
    kv_cache_write_kernel<<<blocks, threads, 0, stream>>>(cache, src, pos, kvDim);
}

