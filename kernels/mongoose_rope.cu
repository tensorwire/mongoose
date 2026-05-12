// mongoose_rope.cu — rope kernels
// Auto-extracted from mongoose.cu. Do not edit directly.

// === RoPE forward on GPU ===
// Apply rotary position embeddings in-place.
// x[seqLen, dim], dim = nHeads * headDim. Operates on pairs (x[2j], x[2j+1]).
// cos_table[pos * halfHead + j], sin_table[pos * halfHead + j] are precomputed.
__global__ void rope_kernel(float* x, const float* cos_tab, const float* sin_tab,
                            int dim, int headDim, int nHeads, int halfHead) {
    int pos = blockIdx.x;
    // HuggingFace rotate_half convention: pair (x[j], x[j+halfHead])
    for (int h = 0; h < nHeads; h++) {
        int base = pos * dim + h * headDim;
        for (int j = threadIdx.x; j < halfHead; j += blockDim.x) {
            float c = cos_tab[pos * halfHead + j];
            float s = sin_tab[pos * halfHead + j];
            float x0 = x[base + j];
            float x1 = x[base + halfHead + j];
            x[base + j]            = x0 * c - x1 * s;
            x[base + halfHead + j] = x0 * s + x1 * c;
        }
    }
}

void mongoose_rope(float* x, const float* cos_tab, const float* sin_tab,
                   int seqLen, int dim, int headDim, int nHeads, cudaStream_t stream) {
    int halfHead = headDim / 2;
    int threads = halfHead < 256 ? halfHead : 256;
    rope_kernel<<<seqLen, threads, 0, stream>>>(x, cos_tab, sin_tab, dim, headDim, nHeads, halfHead);
}

// === RoPE backward (same as forward but negate sin) ===
__global__ void rope_backward_kernel(float* dx, const float* cos_tab, const float* sin_tab,
                                      int dim, int headDim, int nHeads, int halfHead) {
    int pos = blockIdx.x;
    // HuggingFace rotate_half convention backward
    for (int h = 0; h < nHeads; h++) {
        int base = pos * dim + h * headDim;
        for (int j = threadIdx.x; j < halfHead; j += blockDim.x) {
            float c = cos_tab[pos * halfHead + j];
            float s = sin_tab[pos * halfHead + j];
            float x0 = dx[base + j];
            float x1 = dx[base + halfHead + j];
            dx[base + j]            =  x0 * c + x1 * s;
            dx[base + halfHead + j] = -x0 * s + x1 * c;
        }
    }
}

void mongoose_rope_backward(float* dx, const float* cos_tab, const float* sin_tab,
                             int seqLen, int dim, int headDim, int nHeads, cudaStream_t stream) {
    int halfHead = headDim / 2;
    int threads = halfHead < 256 ? halfHead : 256;
    rope_backward_kernel<<<seqLen, threads, 0, stream>>>(dx, cos_tab, sin_tab, dim, headDim, nHeads, halfHead);
}

__global__ void rope_fp16_kernel(__half* x, const float* cos_tab, const float* sin_tab,
                                  int dim, int headDim, int nHeads, int halfHead) {
    int pos = blockIdx.x;
    for (int h = 0; h < nHeads; h++) {
        int base = pos * dim + h * headDim;
        for (int j = threadIdx.x; j < halfHead; j += blockDim.x) {
            float c = cos_tab[pos * halfHead + j];
            float s = sin_tab[pos * halfHead + j];
            float x0 = __half2float(x[base + j]);
            float x1 = __half2float(x[base + halfHead + j]);
            x[base + j]            = __float2half(x0 * c - x1 * s);
            x[base + halfHead + j] = __float2half(x0 * s + x1 * c);
        }
    }
}

void mongoose_rope_fp16(void* x, const float* cos_tab, const float* sin_tab,
                         int seqLen, int dim, int headDim, int nHeads, cudaStream_t stream) {
    int halfHead = headDim / 2;
    int threads = halfHead < 256 ? halfHead : 256;
    rope_fp16_kernel<<<seqLen, threads, 0, stream>>>((__half*)x, cos_tab, sin_tab, dim, headDim, nHeads, halfHead);
}

__global__ void rope_backward_fp16_kernel(__half* dx, const float* cos_tab, const float* sin_tab,
                                           int dim, int headDim, int nHeads, int halfHead) {
    int pos = blockIdx.x;
    for (int h = 0; h < nHeads; h++) {
        int base = pos * dim + h * headDim;
        for (int j = threadIdx.x; j < halfHead; j += blockDim.x) {
            float c = cos_tab[pos * halfHead + j];
            float s = sin_tab[pos * halfHead + j];
            float x0 = __half2float(dx[base + j]);
            float x1 = __half2float(dx[base + halfHead + j]);
            dx[base + j]            = __float2half( x0 * c + x1 * s);
            dx[base + halfHead + j] = __float2half(-x0 * s + x1 * c);
        }
    }
}

void mongoose_rope_backward_fp16(void* dx, const float* cos_tab, const float* sin_tab,
                                  int seqLen, int dim, int headDim, int nHeads, cudaStream_t stream) {
    int halfHead = headDim / 2;
    int threads = halfHead < 256 ? halfHead : 256;
    rope_backward_fp16_kernel<<<seqLen, threads, 0, stream>>>((__half*)dx, cos_tab, sin_tab, dim, headDim, nHeads, halfHead);
}

