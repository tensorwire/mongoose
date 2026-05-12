// mongoose_rmsnorm.cu — rmsnorm kernels
// Auto-extracted from mongoose.cu. Do not edit directly.

// === RMSNorm ===
// x[seqLen, dim] normalized in-place, weight[dim]
__global__ void rmsnorm_kernel(float* x, const float* weight, int dim) {
    int row = blockIdx.x;
    float* xr = x + row * dim;

    // Sum of squares
    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        ss += xr[i] * xr[i];
    }

    // Warp reduce
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        ss += __shfl_down_sync(0xffffffff, ss, offset);

    // Block reduce via shared memory
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    if (lane == 0) shared[wid] = ss;
    __syncthreads();

    // Final reduce across warps — only thread 0 does it sequentially
    // (avoids __shfl_down_sync mask deadlock when nWarps < 32)
    if (threadIdx.x == 0) {
        int nWarps = (blockDim.x + warpSize - 1) / warpSize;
        float total = 0.0f;
        for (int i = 0; i < nWarps; i++) total += shared[i];
        shared[0] = total;
    }
    __syncthreads();

    float scale = rsqrtf(shared[0] / dim + 1e-6f);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        xr[i] = xr[i] * scale * weight[i];
    }
}

void mongoose_rmsnorm(float* x, const float* weight, int seqLen, int dim, cudaStream_t stream) {
    int threads = dim < 1024 ? dim : 1024;
    rmsnorm_kernel<<<seqLen, threads, 0, stream>>>(x, weight, dim);
}

// === RMSNorm out-of-place + save scale for backward ===
// out = rmsnorm(input, weight), rmsScales[row] = scale value for backward.
__global__ void rmsnorm_out_save_kernel(const float* input, float* out, const float* weight,
                                         float* rmsScales, int dim) {
    int row = blockIdx.x;
    const float* inr = input + row * dim;
    float* outr = out + row * dim;

    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        ss += inr[i] * inr[i];
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
    if (threadIdx.x == 0) rmsScales[row] = scale;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        outr[i] = inr[i] * scale * weight[i];
    }
}

void mongoose_rmsnorm_out_save(const float* input, float* out, const float* weight,
                                float* rmsScales, int seqLen, int dim, cudaStream_t stream) {
    int threads = dim < 1024 ? dim : 1024;
    rmsnorm_out_save_kernel<<<seqLen, threads, 0, stream>>>(input, out, weight, rmsScales, dim);
}

// === RMSNorm backward on GPU ===
// dOut[seqLen,dim], xIn[seqLen,dim] (pre-norm input), weight[dim], rmsScales[seqLen]
// dx[seqLen,dim] = gradient w.r.t. xIn
// dx[i] = (dOut[i]*weight[i] - xIn[i] * scale^2 * dot(dOut*weight, xIn) / dim) * scale
__global__ void rmsnorm_backward_kernel(const float* dOut, const float* xIn, const float* weight,
                                         const float* rmsScales, float* dx, int dim) {
    int row = blockIdx.x;
    float scale = rmsScales[row];
    const float* dO = dOut + row * dim;
    const float* x = xIn + row * dim;
    float* dxr = dx + row * dim;

    // Compute dot(dOut*weight, xIn) for this row
    float dot = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        dot += dO[i] * weight[i] * x[i];
    }
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        dot += __shfl_down_sync(0xffffffff, dot, offset);
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    if (lane == 0) shared[wid] = dot;
    __syncthreads();
    if (threadIdx.x == 0) {
        int nWarps = (blockDim.x + warpSize - 1) / warpSize;
        float total = 0.0f;
        for (int i = 0; i < nWarps; i++) total += shared[i];
        shared[0] = total;
    }
    __syncthreads();
    dot = shared[0];

    float coeff = scale * scale * scale * dot / dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        dxr[i] = (dO[i] * weight[i] - x[i] * coeff) * scale;
    }
}

void mongoose_rmsnorm_backward(const float* dOut, const float* xIn, const float* weight,
                                const float* rmsScales, float* dx, int seqLen, int dim, cudaStream_t stream) {
    int threads = dim < 1024 ? dim : 1024;
    rmsnorm_backward_kernel<<<seqLen, threads, 0, stream>>>(dOut, xIn, weight, rmsScales, dx, dim);
}

// === RMSNorm out-of-place: out = rmsnorm(input, weight) ===
// Input is NOT modified. Result written to out.
__global__ void rmsnorm_out_kernel(const float* input, float* out, const float* weight, int dim) {
    int row = blockIdx.x;
    const float* inr = input + row * dim;
    float* outr = out + row * dim;

    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        ss += inr[i] * inr[i];
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
        outr[i] = inr[i] * scale * weight[i];
    }
}

void mongoose_rmsnorm_out(const float* input, float* out, const float* weight, int seqLen, int dim, cudaStream_t stream) {
    int threads = dim < 1024 ? dim : 1024;
    rmsnorm_out_kernel<<<seqLen, threads, 0, stream>>>(input, out, weight, dim);
}

// === RMSNorm weight gradient ===
// dW[d] = sum_pos(dOut[pos,d] * normed[pos,d])
// normed = x / rms (the output of RMSNormOutSave before weight multiply)
// One thread per dim element, reduces across sequence positions.
__global__ void rmsnorm_wgrad_kernel(
    const float* __restrict__ dOut,     // [nPos, dim]
    const float* __restrict__ normed,   // [nPos, dim] — x/rms (pre-weight)
    const float* __restrict__ weight,   // [dim] — current norm weights
    float* __restrict__ dW,             // [dim] — output gradient
    int nPos, int dim
) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    float sum = 0.0f;
    float w = weight[d];
    for (int p = 0; p < nPos; p++) {
        // normed stored by KRMSNormOutSave is (x/rms)*w, so x/rms = normed/w
        float xnorm = (w != 0.0f) ? normed[p * dim + d] / w : 0.0f;
        sum += dOut[p * dim + d] * xnorm;
    }
    dW[d] = sum;
}

void mongoose_rmsnorm_wgrad(
    const float* dOut, const float* normed, const float* weight, float* dW,
    int nPos, int dim, cudaStream_t stream
) {
    rmsnorm_wgrad_kernel<<<(dim+255)/256, 256, 0, stream>>>(dOut, normed, weight, dW, nPos, dim);
}

// === FP16 Element-wise Kernels for Native FP16 Training ===
// All inputs/outputs are __half. Internal accumulation uses float for stability.
// These eliminate the FP32↔FP16 conversion overhead between GEMMs.
// RoPE cos/sin tables stay FP32 (tiny, shared across all positions).

__global__ void rmsnorm_out_save_fp16_kernel(const __half* input, __half* out, const __half* weight,
                                              float* rmsScales, int dim) {
    int row = blockIdx.x;
    const __half* inr = input + row * dim;
    __half* outr = out + row * dim;

    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = __half2float(inr[i]);
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
    if (threadIdx.x == 0) rmsScales[row] = scale;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        outr[i] = __float2half(__half2float(inr[i]) * scale * __half2float(weight[i]));
    }
}

void mongoose_rmsnorm_out_save_fp16(const void* input, void* out, const void* weight,
                                     float* rmsScales, int seqLen, int dim, cudaStream_t stream) {
    int threads = dim < 1024 ? dim : 1024;
    rmsnorm_out_save_fp16_kernel<<<seqLen, threads, 0, stream>>>(
        (const __half*)input, (__half*)out, (const __half*)weight, rmsScales, dim);
}

__global__ void rmsnorm_backward_fp16_kernel(const __half* dOut, const __half* xIn, const __half* weight,
                                              const float* rmsScales, __half* dx, int dim) {
    int row = blockIdx.x;
    float scale = rmsScales[row];
    const __half* dO = dOut + row * dim;
    const __half* x = xIn + row * dim;
    __half* dxr = dx + row * dim;

    float dot = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        dot += __half2float(dO[i]) * __half2float(weight[i]) * __half2float(x[i]);
    }
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        dot += __shfl_down_sync(0xffffffff, dot, offset);
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    if (lane == 0) shared[wid] = dot;
    __syncthreads();
    if (threadIdx.x == 0) {
        int nWarps = (blockDim.x + warpSize - 1) / warpSize;
        float total = 0.0f;
        for (int i = 0; i < nWarps; i++) total += shared[i];
        shared[0] = total;
    }
    __syncthreads();
    dot = shared[0];

    float coeff = scale * scale * scale * dot / dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        dxr[i] = __float2half((__half2float(dO[i]) * __half2float(weight[i]) - __half2float(x[i]) * coeff) * scale);
    }
}

void mongoose_rmsnorm_backward_fp16(const void* dOut, const void* xIn, const void* weight,
                                     const float* rmsScales, void* dx, int seqLen, int dim, cudaStream_t stream) {
    int threads = dim < 1024 ? dim : 1024;
    rmsnorm_backward_fp16_kernel<<<seqLen, threads, 0, stream>>>(
        (const __half*)dOut, (const __half*)xIn, (const __half*)weight, rmsScales, (__half*)dx, dim);
}

