// mongoose_elementwise.cu — elementwise kernels
// Auto-extracted from mongoose.cu. Do not edit directly.

// === ReLU out-of-place: out = relu(input) ===
__global__ void relu_out_kernel(const float* input, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = input[i] > 0 ? input[i] : 0;
}

void mongoose_relu_out(const float* input, float* out, int n, cudaStream_t stream) {
    relu_out_kernel<<<(n+255)/256, 256, 0, stream>>>(input, out, n);
}

// === ReLU ===
__global__ void relu_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && x[i] < 0) x[i] = 0;
}

void mongoose_relu(float* x, int n, cudaStream_t stream) {
    relu_kernel<<<(n+255)/256, 256, 0, stream>>>(x, n);
}

// === ReLU backward: out = dOut * (input > 0) ===
__global__ void relu_backward_kernel(float* out, const float* dOut, const float* input, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = input[i] > 0 ? dOut[i] : 0;
}

void mongoose_relu_backward(float* out, const float* dOut, const float* input, int n, cudaStream_t stream) {
    relu_backward_kernel<<<(n+255)/256, 256, 0, stream>>>(out, dOut, input, n);
}

// === Element-wise add: a += b ===
__global__ void add_inplace_kernel(float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

void mongoose_add_inplace(float* a, const float* b, int n, cudaStream_t stream) {
    add_inplace_kernel<<<(n+255)/256, 256, 0, stream>>>(a, b, n);
}

// === Scale by norm weight: x[i*dim+j] *= weight[j] ===
__global__ void scale_by_weight_kernel(float* x, const float* weight, int dim, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) x[i] *= weight[i % dim];
}

void mongoose_scale_by_weight(float* x, const float* weight, int seqLen, int dim, cudaStream_t stream) {
    int n = seqLen * dim;
    scale_by_weight_kernel<<<(n+255)/256, 256, 0, stream>>>(x, weight, dim, n);
}

// === Embedding gather: out[i] = tokEmb[tokens[i]] + posEmb[i] ===
__global__ void embedding_gather_kernel(float* out, const float* tokEmb, const float* posEmb,
                                         const int* tokens, int dim) {
    int pos = blockIdx.x;
    int tok = tokens[pos];
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        out[pos*dim + j] = tokEmb[tok*dim + j] + posEmb[pos*dim + j];
    }
}

void mongoose_embedding_gather(float* out, const float* tokEmb, const float* posEmb,
                                const int* tokens, int seqLen, int dim, cudaStream_t stream) {
    int threads = dim < 1024 ? dim : 1024;
    embedding_gather_kernel<<<seqLen, threads, 0, stream>>>(out, tokEmb, posEmb, tokens, dim);
}

// === Copy device to device ===
void mongoose_copy(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream);
}

// === Memset zero ===
void mongoose_zero(void* ptr, size_t bytes, cudaStream_t stream) {
    cudaMemsetAsync(ptr, 0, bytes, stream);
}

// === Sync ===
void mongoose_sync(cudaStream_t stream) {
    cudaStreamSynchronize(stream);
}

// === Fused residual add + RMSNorm: out = rmsnorm(a + b, weight) ===
// Eliminates 2 kernel launches (add + rmsnorm) → 1 dispatch.
__global__ void fused_add_rmsnorm_kernel(const float* a, const float* b, float* out,
                                          const float* weight, int dim) {
    int row = blockIdx.x;
    const float* ar = a + row * dim;
    const float* br = b + row * dim;
    float* outr = out + row * dim;

    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = ar[i] + br[i];
        outr[i] = v;
        ss += v * v;
    }
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        ss += __shfl_down_sync(0xffffffff, ss, offset);
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    if (lane == 0) shared[wid] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        int nWarps = (blockDim.x + warpSize - 1) / warpSize;
        float total = 0.0f;
        for (int i = 0; i < nWarps; i++) total += shared[i];
        shared[0] = total;
    }
    __syncthreads();

    float scale = rsqrtf(shared[0] / dim + 1e-6f);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        outr[i] = outr[i] * scale * weight[i];
    }
}

void mongoose_fused_add_rmsnorm(const float* a, const float* b, float* out,
                                 const float* weight, int seqLen, int dim, cudaStream_t stream) {
    int threads = dim < 1024 ? dim : 1024;
    fused_add_rmsnorm_kernel<<<seqLen, threads, 0, stream>>>(a, b, out, weight, dim);
}

// === Fused residual add out-of-place: out = a + b ===
__global__ void add_out_kernel(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

void mongoose_add_out(const float* a, const float* b, float* out, int n, cudaStream_t stream) {
    add_out_kernel<<<(n+255)/256, 256, 0, stream>>>(a, b, out, n);
}

// === Scale: out[i] = x[i] * alpha (for gradient scaling) ===
__global__ void scale_kernel(const float* x, float* out, float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] * alpha;
}

void mongoose_scale(const float* x, float* out, float alpha, int n, cudaStream_t stream) {
    scale_kernel<<<(n+255)/256, 256, 0, stream>>>(x, out, alpha, n);
}

// === Embedding gather (no position embedding, just token) ===
// out[pos*dim..] = embed[token[pos]*dim..]
__global__ void embed_gather_kernel(float* out, const float* embed, const int* tokens, int dim) {
    int pos = blockIdx.x;
    int tok = tokens[pos];
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        out[pos*dim + j] = embed[tok*dim + j];
    }
}

void mongoose_embed_gather(float* out, const float* embed, const int* tokens,
                           int seqLen, int dim, cudaStream_t stream) {
    int threads = dim < 1024 ? dim : 1024;
    embed_gather_kernel<<<seqLen, threads, 0, stream>>>(out, embed, tokens, dim);
}

// === FP16 Matmul with Transpose B ===
// C[m,n] = A[m,k] @ B[n,k]^T, where A and B are FP16, C is FP32.
// This is the mixed-precision path for Q8 LoRA forward: INT8→FP16 dequant + FP16 matmul.

// === FP32 ↔ FP16 Conversion ===
// Convert FP32 tensor to FP16 in-place on GPU. For mixed-precision matmul input conversion.
__global__ void fp32_to_fp16_kernel(const float* in, __half* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

void mongoose_fp32_to_fp16(const float* in, void* out, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    fp32_to_fp16_kernel<<<blocks, threads, 0, stream>>>(in, (__half*)out, n);
}

// Convert FP16 tensor to FP32 on GPU.
__global__ void fp16_to_fp32_kernel(const __half* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
}

void mongoose_fp16_to_fp32(const void* in, float* out, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    fp16_to_fp32_kernel<<<blocks, threads, 0, stream>>>((__half*)in, out, n);
}

// === FP16 Utility Kernels ===

__global__ void fp16_add_inplace_kernel(__half* a, const __half* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = __float2half(__half2float(a[i]) + __half2float(b[i]));
}

void mongoose_fp16_add_inplace(void* a, const void* b, int n, cudaStream_t stream) {
    fp16_add_inplace_kernel<<<(n+255)/256, 256, 0, stream>>>((__half*)a, (const __half*)b, n);
}

__global__ void fp32_add_fp16_kernel(float* a, const __half* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += __half2float(b[i]);
}

void mongoose_fp32_add_fp16(float* a, const void* b, int n, cudaStream_t stream) {
    fp32_add_fp16_kernel<<<(n+255)/256, 256, 0, stream>>>(a, (const __half*)b, n);
}


// === Argmax: find index of max element ===
__global__ void argmax_kernel(const float* data, unsigned int* result, int n) {
    __shared__ float smax[256];
    __shared__ unsigned int sidx[256];

    float best = -1e30f;
    unsigned int bestIdx = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = data[i];
        if (v > best) { best = v; bestIdx = i; }
    }
    smax[threadIdx.x] = best;
    sidx[threadIdx.x] = bestIdx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && smax[threadIdx.x + s] > smax[threadIdx.x]) {
            smax[threadIdx.x] = smax[threadIdx.x + s];
            sidx[threadIdx.x] = sidx[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) result[0] = sidx[0];
}

void mongoose_argmax(const float* data, unsigned int* result, int n, cudaStream_t stream) {
    argmax_kernel<<<1, 256, 0, stream>>>(data, result, n);
}

// === Embed gather single token: out[0..dim-1] = embed[tokenID*dim .. (tokenID+1)*dim-1] ===
__global__ void embed_gather_single_kernel(const float* embed, float* out, int tokenID, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) out[i] = embed[tokenID * dim + i];
}

void mongoose_embed_gather_single(const float* embed, float* out, int tokenID, int dim, cudaStream_t stream) {
    embed_gather_single_kernel<<<(dim+255)/256, 256, 0, stream>>>(embed, out, tokenID, dim);
}
