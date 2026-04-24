// mongoose CUDA kernels — compiled once with nvcc, loaded at runtime by Go.
// nvcc -shared -o libmongoose_kernels.so mongoose.cu -Xcompiler -fPIC
//
// These are the element-wise ops that keep data on GPU between cuBLAS matmuls.
// No Python. No CGo for these. Just dlopen at runtime.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

extern "C" {

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

// === Causal Multi-Head Self-Attention (GQA-aware) ===
// One block per (position, head). GQA: multiple Q heads share K/V heads.
// Q[seqLen, dim], K[seqLen, kvDim], V[seqLen, kvDim], out[seqLen, dim].
__global__ void causal_attention_kernel(
    const float* Q, const float* K, const float* V, float* out,
    int seqLen, int dim, int kvDim, int numHeads, int numKVHeads, int headDim
) {
    int pos = blockIdx.x;
    int head = blockIdx.y;
    int hOff = head * headDim;
    int kvHead = head / (numHeads / numKVHeads);
    int kvOff = kvHead * headDim;

    extern __shared__ float shared[];
    float* scores = shared;

    float scale = rsqrtf((float)headDim);

    for (int j = threadIdx.x; j <= pos; j += blockDim.x) {
        float dot = 0.0f;
        for (int d = 0; d < headDim; d++) {
            dot += Q[pos * dim + hOff + d] * K[j * kvDim + kvOff + d];
        }
        scores[j] = dot * scale;
    }
    __syncthreads();

    // Softmax: find max
    float maxVal = -1e30f;
    for (int j = threadIdx.x; j <= pos; j += blockDim.x) {
        if (scores[j] > maxVal) maxVal = scores[j];
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        maxVal = fmaxf(maxVal, __shfl_down_sync(0xffffffff, maxVal, offset));
    __shared__ float blockMax[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    if (lane == 0) blockMax[wid] = maxVal;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = blockMax[0];
        for (int i = 1; i < (blockDim.x + warpSize - 1) / warpSize; i++)
            m = fmaxf(m, blockMax[i]);
        blockMax[0] = m;
    }
    __syncthreads();
    maxVal = blockMax[0];

    float sumExp = 0.0f;
    for (int j = threadIdx.x; j <= pos; j += blockDim.x) {
        scores[j] = expf(scores[j] - maxVal);
        sumExp += scores[j];
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        sumExp += __shfl_down_sync(0xffffffff, sumExp, offset);
    __shared__ float blockSum[32];
    if (lane == 0) blockSum[wid] = sumExp;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0;
        for (int i = 0; i < (blockDim.x + warpSize - 1) / warpSize; i++)
            s += blockSum[i];
        blockSum[0] = s;
    }
    __syncthreads();
    sumExp = blockSum[0];

    for (int j = threadIdx.x; j <= pos; j += blockDim.x) {
        scores[j] /= sumExp;
    }
    __syncthreads();

    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        float val = 0.0f;
        for (int j = 0; j <= pos; j++) {
            val += scores[j] * V[j * kvDim + kvOff + d];
        }
        out[pos * dim + hOff + d] = val;
    }
}

void mongoose_causal_attention(
    const float* Q, const float* K, const float* V, float* out,
    int seqLen, int dim, int numHeads, cudaStream_t stream
) {
    // Legacy MHA signature — kvDim=dim, numKVHeads=numHeads
    int headDim = dim / numHeads;
    int threads = headDim < 256 ? headDim : 256;
    size_t sharedBytes = seqLen * sizeof(float);
    dim3 grid(seqLen, numHeads);
    causal_attention_kernel<<<grid, threads, sharedBytes, stream>>>(
        Q, K, V, out, seqLen, dim, dim, numHeads, numHeads, headDim);
}

// GQA-aware version with explicit kvDim and numKVHeads
void mongoose_causal_attention_gqa(
    const float* Q, const float* K, const float* V, float* out,
    int seqLen, int dim, int kvDim, int numHeads, int numKVHeads, cudaStream_t stream
) {
    int headDim = dim / numHeads;
    int threads = headDim < 256 ? headDim : 256;
    size_t sharedBytes = seqLen * sizeof(float);
    dim3 grid(seqLen, numHeads);
    causal_attention_kernel<<<grid, threads, sharedBytes, stream>>>(
        Q, K, V, out, seqLen, dim, kvDim, numHeads, numKVHeads, headDim);
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

// === Sparse FFN Kernels ===
// CPU-predicted sparse dispatch: skip zero columns in matmul after ReLU.

// relu_and_index: Apply ReLU in-place AND build a compact index of non-zero dimensions.
// x[n] is modified in-place (ReLU applied).
// activeIdx[n] is filled with indices of non-zero elements.
// activeCount[1] is set to the number of non-zero elements.
// This fuses the ReLU + sparsity scan into one kernel — no CPU round-trip needed.
__global__ void relu_and_index_kernel(float* x, int* activeIdx, int* activeCount, int n) {
    __shared__ int blockCount;
    if (threadIdx.x == 0) blockCount = 0;
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int isActive = 0;

    if (i < n) {
        if (x[i] <= 0) {
            x[i] = 0;
        } else {
            isActive = 1;
        }
    }

    // Count actives in this block via atomicAdd to shared memory
    int localIdx = -1;
    if (isActive) {
        localIdx = atomicAdd(&blockCount, 1);
    }
    __syncthreads();

    // One thread per block reserves a range in the global activeIdx
    __shared__ int globalOffset;
    if (threadIdx.x == 0 && blockCount > 0) {
        globalOffset = atomicAdd(activeCount, blockCount);
    }
    __syncthreads();

    // Write active indices to global memory
    if (isActive && localIdx >= 0) {
        activeIdx[globalOffset + localIdx] = i;
    }
}

void mongoose_relu_and_index(float* x, int* activeIdx, int* activeCount, int n, cudaStream_t stream) {
    // Zero the count
    cudaMemsetAsync(activeCount, 0, sizeof(int), stream);
    relu_and_index_kernel<<<(n+255)/256, 256, 0, stream>>>(x, activeIdx, activeCount, n);
}

// sparse_matmul: out[i] = sum_j(WT[activeIdx[j]*rows + i] * x[activeIdx[j]])
// WT is the TRANSPOSED weight matrix stored column-major: WT[col*rows + row] = W[row*cols + col]
// This allows sequential memory access per active column.
//
// Grid: one block per output row (or group of rows).
// Each block processes all active columns for its assigned rows.
//
// For inference (rows=HiddenDim ~1024-4096, activeCols ~500-2000):
// Launch rows blocks, each thread handles a subset of active columns.
__global__ void sparse_matmul_kernel(
    float* out,           // [rows] output vector
    const float* WT,      // [cols * rows] transposed weight matrix
    const float* x,       // [cols] input vector (post-ReLU, sparse)
    const int* activeIdx, // [activeCount] indices of non-zero elements
    int activeCount,       // number of active columns
    int rows,              // output dimension (HiddenDim)
    int cols               // input dimension (FFNDim) — for WT layout
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    float sum = 0.0f;
    // Each thread processes a stride of active columns
    for (int a = threadIdx.x; a < activeCount; a += blockDim.x) {
        int j = activeIdx[a];
        sum += WT[j * rows + row] * x[j];
    }

    // Warp reduce
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    // Block reduce via shared memory
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    if (lane == 0) shared[wid] = sum;
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = 0;
        int nWarps = (blockDim.x + warpSize - 1) / warpSize;
        for (int i = 0; i < nWarps; i++)
            total += shared[i];
        out[row] = total;
    }
}

void mongoose_sparse_matmul(
    float* out, const float* WT, const float* x,
    const int* activeIdx, int activeCount,
    int rows, int cols, cudaStream_t stream
) {
    int threads = 256;
    if (activeCount < 256) threads = ((activeCount + 31) / 32) * 32; // round up to warp
    if (threads < 32) threads = 32;
    sparse_matmul_kernel<<<rows, threads, 0, stream>>>(
        out, WT, x, activeIdx, activeCount, rows, cols);
}

// === FP16 Sparse FFN Kernels ===
// Same as FP32 sparse kernels but with half-precision inputs for 2x bandwidth reduction.
// FP32 accumulation for numerical stability — same as cuBLAS mixed-precision.

// relu_and_index for FP16: apply ReLU in-place on half* and build active index.
__global__ void relu_and_index_fp16_kernel(__half* x, int* activeIdx, int* activeCount, int n) {
    __shared__ int blockCount;
    if (threadIdx.x == 0) blockCount = 0;
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int isActive = 0;

    if (i < n) {
        float val = __half2float(x[i]);
        if (val <= 0.0f) {
            x[i] = __float2half(0.0f);
        } else {
            isActive = 1;
        }
    }

    int localIdx = -1;
    if (isActive) {
        localIdx = atomicAdd(&blockCount, 1);
    }
    __syncthreads();

    __shared__ int globalOffset;
    if (threadIdx.x == 0 && blockCount > 0) {
        globalOffset = atomicAdd(activeCount, blockCount);
    }
    __syncthreads();

    if (isActive && localIdx >= 0) {
        activeIdx[globalOffset + localIdx] = i;
    }
}

void mongoose_relu_and_index_fp16(void* x, int* activeIdx, int* activeCount, int n, cudaStream_t stream) {
    cudaMemsetAsync(activeCount, 0, sizeof(int), stream);
    relu_and_index_fp16_kernel<<<(n+255)/256, 256, 0, stream>>>((__half*)x, activeIdx, activeCount, n);
}

// sparse_matmul FP16: WT is half*, x is half*, output is half*.
// FP32 accumulation internally. Reads half the bytes per element vs FP32.
__global__ void sparse_matmul_fp16_kernel(
    __half* out,
    const __half* WT,     // [cols * rows] transposed, FP16
    const __half* x,      // [cols] sparse activation, FP16
    const int* activeIdx,
    int activeCount,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    float sum = 0.0f;
    for (int a = threadIdx.x; a < activeCount; a += blockDim.x) {
        int j = activeIdx[a];
        sum += __half2float(WT[j * rows + row]) * __half2float(x[j]);
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
        float total = 0;
        int nWarps = (blockDim.x + warpSize - 1) / warpSize;
        for (int i = 0; i < nWarps; i++)
            total += shared[i];
        out[row] = __float2half(total);
    }
}

void mongoose_sparse_matmul_fp16(
    void* out, const void* WT, const void* x,
    const int* activeIdx, int activeCount,
    int rows, int cols, cudaStream_t stream
) {
    int threads = 256;
    if (activeCount < 256) threads = ((activeCount + 31) / 32) * 32;
    if (threads < 32) threads = 32;
    sparse_matmul_fp16_kernel<<<rows, threads, 0, stream>>>(
        (__half*)out, (const __half*)WT, (const __half*)x,
        activeIdx, activeCount, rows, cols);
}

// === Causal Attention Backward (GQA-aware) ===
// One block per (position, head). Recomputes softmax scores from Q/K, then computes dQ/dK/dV.
// Supports GQA: kvDim may differ from dim. kvMul = numHeads / numKVHeads.
// Q[seqLen, dim], K[seqLen, kvDim], V[seqLen, kvDim], dOut[seqLen, dim]
// dQ[seqLen, dim], dK[seqLen, kvDim], dV[seqLen, kvDim]
// Uses atomicAdd for dK/dV since multiple positions write to the same KV position.
__global__ void causal_attention_backward_kernel(
    const float* Q, const float* K, const float* V, const float* dOut,
    float* dQ, float* dK, float* dV,
    int seqLen, int dim, int kvDim, int numHeads, int numKVHeads, int headDim
) {
    int pos = blockIdx.x;
    int head = blockIdx.y;
    int hOff = head * headDim;
    int kvHead = head / (numHeads / numKVHeads);
    int kvOff = kvHead * headDim;

    extern __shared__ float shared[];
    float* scores = shared;            // [pos+1]
    float* dW = scores + (pos + 1);    // [pos+1]

    float scale = rsqrtf((float)headDim);

    // Step 1: Recompute attention scores (same as forward)
    for (int j = threadIdx.x; j <= pos; j += blockDim.x) {
        float dot = 0.0f;
        for (int d = 0; d < headDim; d++) {
            dot += Q[pos * dim + hOff + d] * K[j * kvDim + kvOff + d];
        }
        scores[j] = dot * scale;
    }
    __syncthreads();

    // Softmax: find max
    float maxVal = -1e30f;
    for (int j = threadIdx.x; j <= pos; j += blockDim.x) {
        if (scores[j] > maxVal) maxVal = scores[j];
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        maxVal = fmaxf(maxVal, __shfl_down_sync(0xffffffff, maxVal, offset));
    __shared__ float blockMax[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    if (lane == 0) blockMax[wid] = maxVal;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = blockMax[0];
        for (int i = 1; i < (blockDim.x + warpSize - 1) / warpSize; i++)
            m = fmaxf(m, blockMax[i]);
        blockMax[0] = m;
    }
    __syncthreads();
    maxVal = blockMax[0];

    // Softmax: exp and sum
    float sumExp = 0.0f;
    for (int j = threadIdx.x; j <= pos; j += blockDim.x) {
        scores[j] = expf(scores[j] - maxVal);
        sumExp += scores[j];
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        sumExp += __shfl_down_sync(0xffffffff, sumExp, offset);
    __shared__ float blockSum[32];
    if (lane == 0) blockSum[wid] = sumExp;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0;
        for (int i = 0; i < (blockDim.x + warpSize - 1) / warpSize; i++)
            s += blockSum[i];
        blockSum[0] = s;
    }
    __syncthreads();
    sumExp = blockSum[0];

    for (int j = threadIdx.x; j <= pos; j += blockDim.x) {
        scores[j] /= sumExp;
    }
    __syncthreads();

    // Step 2: Compute dW[t] = dOut[pos,h] · V[t,kvH] and accumulate dV
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        float dw = 0.0f;
        for (int d = 0; d < headDim; d++) {
            float dO = dOut[pos * dim + hOff + d];
            dw += dO * V[t * kvDim + kvOff + d];
            atomicAdd(&dV[t * kvDim + kvOff + d], scores[t] * dO);
        }
        dW[t] = dw;
    }
    __syncthreads();

    // Step 3: wdw = sum(scores[t] * dW[t])
    float wdw = 0.0f;
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        wdw += scores[t] * dW[t];
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        wdw += __shfl_down_sync(0xffffffff, wdw, offset);
    // Use blockSum for this reduction too
    if (lane == 0) blockSum[wid] = wdw;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0;
        for (int i = 0; i < (blockDim.x + warpSize - 1) / warpSize; i++)
            s += blockSum[i];
        blockSum[0] = s;
    }
    __syncthreads();
    wdw = blockSum[0];

    // Step 4: ds[t] = scores[t] * (dW[t] - wdw) * scale, accumulate dQ and dK
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        float ds = scores[t] * (dW[t] - wdw) * scale;
        for (int d = 0; d < headDim; d++) {
            atomicAdd(&dQ[pos * dim + hOff + d], ds * K[t * kvDim + kvOff + d]);
            atomicAdd(&dK[t * kvDim + kvOff + d], ds * Q[pos * dim + hOff + d]);
        }
    }
}

void mongoose_causal_attention_backward(
    const float* Q, const float* K, const float* V, const float* dOut,
    float* dQ, float* dK, float* dV,
    int seqLen, int dim, int kvDim, int numHeads, int numKVHeads, cudaStream_t stream
) {
    int headDim = dim / numHeads;
    int threads = headDim < 256 ? headDim : 256;
    // Shared memory: scores[seqLen] + dW[seqLen]
    size_t sharedBytes = 2 * seqLen * sizeof(float);
    dim3 grid(seqLen, numHeads);

    // Zero dQ, dK, dV first (atomicAdd accumulates)
    cudaMemsetAsync(dQ, 0, seqLen * dim * sizeof(float), stream);
    cudaMemsetAsync(dK, 0, seqLen * kvDim * sizeof(float), stream);
    cudaMemsetAsync(dV, 0, seqLen * kvDim * sizeof(float), stream);

    causal_attention_backward_kernel<<<grid, threads, sharedBytes, stream>>>(
        Q, K, V, dOut, dQ, dK, dV,
        seqLen, dim, kvDim, numHeads, numKVHeads, headDim);
}

// === Decode Attention: single query position against full KV cache ===
// One block per head. Q is [1, dim], K_cache is [cacheLen, kvDim], V_cache is [cacheLen, kvDim].
// GQA-aware: kvHead = head / (numHeads/numKVHeads).
__global__ void decode_attention_kernel(
    const float* Q, const float* K_cache, const float* V_cache, float* out,
    int cacheLen, int dim, int kvDim, int numHeads, int numKVHeads, int headDim
) {
    int head = blockIdx.x;
    int hOff = head * headDim;
    int kvHead = head / (numHeads / numKVHeads);
    int kvOff = kvHead * headDim;

    extern __shared__ float shared[];
    float* scores = shared;

    float scale = rsqrtf((float)headDim);

    // Compute attention scores: Q[head] dot K_cache[t, kvHead] for all t
    for (int t = threadIdx.x; t < cacheLen; t += blockDim.x) {
        float dot = 0.0f;
        for (int d = 0; d < headDim; d++) {
            dot += Q[hOff + d] * K_cache[t * kvDim + kvOff + d];
        }
        scores[t] = dot * scale;
    }
    __syncthreads();

    // Softmax
    float maxVal = -1e30f;
    for (int t = threadIdx.x; t < cacheLen; t += blockDim.x) {
        if (scores[t] > maxVal) maxVal = scores[t];
    }
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        maxVal = fmaxf(maxVal, __shfl_down_sync(0xffffffff, maxVal, offset));
    __shared__ float blockMax[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    if (lane == 0) blockMax[wid] = maxVal;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = blockMax[0];
        for (int i = 1; i < (blockDim.x + warpSize - 1) / warpSize; i++)
            m = fmaxf(m, blockMax[i]);
        blockMax[0] = m;
    }
    __syncthreads();
    maxVal = blockMax[0];

    float sumExp = 0.0f;
    for (int t = threadIdx.x; t < cacheLen; t += blockDim.x) {
        scores[t] = expf(scores[t] - maxVal);
        sumExp += scores[t];
    }
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        sumExp += __shfl_down_sync(0xffffffff, sumExp, offset);
    __shared__ float blockSum[32];
    if (lane == 0) blockSum[wid] = sumExp;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0;
        for (int i = 0; i < (blockDim.x + warpSize - 1) / warpSize; i++)
            s += blockSum[i];
        blockSum[0] = s;
    }
    __syncthreads();
    sumExp = blockSum[0];

    for (int t = threadIdx.x; t < cacheLen; t += blockDim.x) {
        scores[t] /= sumExp;
    }
    __syncthreads();

    // Weighted sum of values
    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        float val = 0.0f;
        for (int t = 0; t < cacheLen; t++) {
            val += scores[t] * V_cache[t * kvDim + kvOff + d];
        }
        out[hOff + d] = val;
    }
}

void mongoose_decode_attention(
    const float* Q, const float* K_cache, const float* V_cache, float* out,
    int cacheLen, int dim, int kvDim, int numHeads, int numKVHeads, cudaStream_t stream
) {
    int headDim = dim / numHeads;
    int threads = cacheLen < 256 ? ((cacheLen + 31) / 32) * 32 : 256;
    if (threads < 32) threads = 32;
    size_t sharedBytes = cacheLen * sizeof(float);
    decode_attention_kernel<<<numHeads, threads, sharedBytes, stream>>>(
        Q, K_cache, V_cache, out, cacheLen, dim, kvDim, numHeads, numKVHeads, headDim);
}

// === Fused SiLU-Gate-Mul: out[i] = silu(gate[i]) * up[i] ===
// gate and up are [n], out is [n]. Eliminates 2 PCIe round-trips vs CPU.
__global__ void silu_gate_mul_kernel(const float* gate, const float* up, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        float sig = 1.0f / (1.0f + expf(-g));
        out[i] = g * sig * up[i];
    }
}

void mongoose_silu_gate_mul(const float* gate, const float* up, float* out, int n, cudaStream_t stream) {
    silu_gate_mul_kernel<<<(n+255)/256, 256, 0, stream>>>(gate, up, out, n);
}

// === SiLU-Gate-Mul backward: dGate and dUp from dOut ===
// dUp[i] = dOut[i] * silu(gate[i])
// dGate[i] = dOut[i] * up[i] * (sig + gate[i]*sig*(1-sig))
// where sig = sigmoid(gate[i])
__global__ void silu_gate_backward_kernel(
    const float* dOut, const float* gate, const float* up,
    float* dGate, float* dUp, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        float sig = 1.0f / (1.0f + expf(-g));
        float silu_g = g * sig;
        dUp[i] = dOut[i] * silu_g;
        dGate[i] = dOut[i] * up[i] * (sig + silu_g * (1.0f - sig));
    }
}

void mongoose_silu_gate_backward(
    const float* dOut, const float* gate, const float* up,
    float* dGate, float* dUp, int n, cudaStream_t stream
) {
    silu_gate_backward_kernel<<<(n+255)/256, 256, 0, stream>>>(dOut, gate, up, dGate, dUp, n);
}

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

// === Cross-entropy loss + gradient (fused) ===
// logits[vocabSize] per position, target token, outputs loss and dLogits.
// dLogits[v] = softmax(logits)[v] - (v == target ? 1 : 0), scaled by invN.
__global__ void cross_entropy_kernel(
    const float* hidden, const float* embedW, int D, int vocabSize,
    const int* targets, float* losses, float* dHidden, float invN, int nPos
) {
    int pos = blockIdx.x;
    if (pos >= nPos) return;

    // Compute logits = hidden[pos] @ embedW^T
    extern __shared__ float shared[];
    float* logits = shared; // [vocabSize]

    for (int v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        float dot = 0.0f;
        for (int j = 0; j < D; j++) {
            dot += hidden[pos*D + j] * embedW[v*D + j];
        }
        logits[v] = dot;
    }
    __syncthreads();

    // Find max for numerical stability
    float mx = -1e30f;
    for (int v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        if (logits[v] > mx) mx = logits[v];
    }
    // Reduce max across threads
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, offset));
    __shared__ float blockMax[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    if (lane == 0) blockMax[wid] = mx;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = blockMax[0];
        for (int i = 1; i < (blockDim.x + warpSize - 1) / warpSize; i++)
            m = fmaxf(m, blockMax[i]);
        blockMax[0] = m;
    }
    __syncthreads();
    mx = blockMax[0];

    // Exp and sum
    float sumExp = 0.0f;
    for (int v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        logits[v] = expf(logits[v] - mx);
        sumExp += logits[v];
    }
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        sumExp += __shfl_down_sync(0xffffffff, sumExp, offset);
    __shared__ float blockSum[32];
    if (lane == 0) blockSum[wid] = sumExp;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0;
        for (int i = 0; i < (blockDim.x + warpSize - 1) / warpSize; i++)
            s += blockSum[i];
        blockSum[0] = s;
    }
    __syncthreads();
    sumExp = blockSum[0];

    // Loss
    int target = targets[pos];
    if (threadIdx.x == 0) {
        float prob = logits[target] / sumExp;
        if (prob < 1e-10f) prob = 1e-10f;
        losses[pos] = -logf(prob);
    }

    // dHidden[pos] = sum_v (softmax[v] - target_v) * embedW[v]
    // Gradient flows through embed weights
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float dh = 0.0f;
        for (int v = 0; v < vocabSize; v++) {
            float sv = (logits[v] / sumExp) * invN;
            if (v == target) sv -= invN;
            dh += sv * embedW[v*D + j];
        }
        dHidden[pos*D + j] = dh;
    }
}

void mongoose_cross_entropy(
    const float* hidden, const float* embedW, int D, int vocabSize,
    const int* targets, float* losses, float* dHidden, float invN,
    int nPos, cudaStream_t stream
) {
    int threads = D < 256 ? D : 256;
    size_t sharedBytes = vocabSize * sizeof(float);
    cross_entropy_kernel<<<nPos, threads, sharedBytes, stream>>>(
        hidden, embedW, D, vocabSize, targets, losses, dHidden, invN, nPos);
}

// === Softmax + Cross-Entropy on pre-computed logits (large vocab) ===
// Operates on logits already in global memory [nPos, vocabSize].
// Two passes over the logits buffer (no recomputation):
//   Pass 1: max + sum_exp + loss (modifies logits in-place to exp values)
//   Pass 2: gradient = (softmax - one_hot) * invN, written to grad buffer
// One block per position. No vocab-sized shared memory.

__global__ void softmax_ce_kernel(
    float* __restrict__ logits,         // [nPos, vocabSize] — modified in-place
    const int* __restrict__ targets,    // [nPos]
    float* __restrict__ losses,         // [nPos]
    float* __restrict__ grad,           // [nPos, vocabSize]
    int vocabSize, float invN
) {
    int pos = blockIdx.x;
    float* row = logits + (long long)pos * vocabSize;
    float* grow = grad + (long long)pos * vocabSize;
    int target = targets[pos];
    int tid = threadIdx.x;
    int nThreads = blockDim.x;
    int lane = tid % warpSize;
    int wid = tid / warpSize;

    // Pass 1a: find max
    float localMax = -1e30f;
    for (int v = tid; v < vocabSize; v += nThreads) {
        float val = row[v];
        if (val > localMax) localMax = val;
    }
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        localMax = fmaxf(localMax, __shfl_down_sync(0xffffffff, localMax, offset));
    __shared__ float sdata[32];
    if (lane == 0) sdata[wid] = localMax;
    __syncthreads();
    if (tid == 0) {
        int nWarps = (nThreads + warpSize - 1) / warpSize;
        float m = sdata[0];
        for (int i = 1; i < nWarps; i++) m = fmaxf(m, sdata[i]);
        sdata[0] = m;
    }
    __syncthreads();
    float mx = sdata[0];

    // Pass 1b: exp + sum
    float localSum = 0.0f;
    for (int v = tid; v < vocabSize; v += nThreads) {
        float e = expf(row[v] - mx);
        row[v] = e;
        localSum += e;
    }
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        localSum += __shfl_down_sync(0xffffffff, localSum, offset);
    if (lane == 0) sdata[wid] = localSum;
    __syncthreads();
    if (tid == 0) {
        int nWarps = (nThreads + warpSize - 1) / warpSize;
        float s = 0;
        for (int i = 0; i < nWarps; i++) s += sdata[i];
        sdata[0] = s;
        float prob = row[target] / s;
        if (prob < 1e-10f) prob = 1e-10f;
        losses[pos] = -logf(prob);
    }
    __syncthreads();
    float sumExp = sdata[0];

    // Pass 2: gradient
    for (int v = tid; v < vocabSize; v += nThreads) {
        float sv = (row[v] / sumExp) * invN;
        if (v == target) sv -= invN;
        grow[v] = sv;
    }
}

void mongoose_softmax_ce(
    float* logits, const int* targets, float* losses, float* grad,
    int nPos, int vocabSize, float invN, cudaStream_t stream
) {
    int threads = vocabSize < 256 ? vocabSize : 256;
    softmax_ce_kernel<<<nPos, threads, 0, stream>>>(
        logits, targets, losses, grad, vocabSize, invN);
}

// === Helix DNA step — FP32 paired parameter update with rung coupling ===
__global__ void helix_dna_step_kernel(
    float* d1, float* d2,
    const float* g1, const float* g2,
    float* m1, float* m2,
    float* v1, float* v2,
    float lr, float beta1, float beta2, float bc1, float bc2,
    float eps, float wd,
    float backbone1, float glyco1, float hbond1,
    float hbond2, float glyco2, float backbone2,
    float bondStrength, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float ob1 = 1.0f - beta1, ob2 = 1.0f - beta2;

    float effGrad1 = g1[i] * glyco1 + g2[i] * hbond1 * bondStrength;
    float mi1 = beta1 * m1[i] + ob1 * effGrad1;
    float vi1 = beta2 * v1[i] + ob2 * effGrad1 * effGrad1;
    m1[i] = mi1; v1[i] = vi1;
    d1[i] -= lr * (mi1 / bc1 / (sqrtf(vi1 / bc2) + eps) + wd * backbone1 * d1[i]);

    float effGrad2 = g2[i] * glyco2 + g1[i] * hbond2 * bondStrength;
    float mi2 = beta1 * m2[i] + ob1 * effGrad2;
    float vi2 = beta2 * v2[i] + ob2 * effGrad2 * effGrad2;
    m2[i] = mi2; v2[i] = vi2;
    d2[i] -= lr * (mi2 / bc1 / (sqrtf(vi2 / bc2) + eps) + wd * backbone2 * d2[i]);
}

void mongoose_helix_dna_step(
    float* d1, float* d2, const float* g1, const float* g2,
    float* m1, float* m2, float* v1, float* v2,
    float lr, float beta1, float beta2, float bc1, float bc2,
    float eps, float wd,
    float backbone1, float glyco1, float hbond1,
    float hbond2, float glyco2, float backbone2,
    float bondStrength, int n, cudaStream_t stream
) {
    helix_dna_step_kernel<<<(n+255)/256, 256, 0, stream>>>(
        d1, d2, g1, g2, m1, m2, v1, v2,
        lr, beta1, beta2, bc1, bc2, eps, wd,
        backbone1, glyco1, hbond1, hbond2, glyco2, backbone2,
        bondStrength, n);
}

// === Needle: INT8 weight update with sub-quant delta accumulation ===
__global__ void helix_needle_kernel(
    int8_t* __restrict__ data_int8, float* __restrict__ scales,
    const float* __restrict__ grad,
    __half* __restrict__ mom, __half* __restrict__ vel,
    const float* __restrict__ rowMask,
    float lr, float beta1, float beta2, float bc1, float bc2,
    float eps, float wd, int n, int cols
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int row = i / cols;
    float maskVal = rowMask[row];
    if (maskVal == 0.0f) return;
    int ci = ((int)(maskVal - 1.0f)) * cols + (i % cols);

    float scale = scales[row] / 127.0f;
    float delta = __half2float(vel[ci]);
    float mi = __half2float(mom[ci]);
    float g = grad[i];
    float ob1 = 1.0f - beta1, ob2 = 1.0f - beta2;
    mi = beta1 * mi + ob1 * g;
    float mhat = mi / bc1;
    delta -= lr * (mhat / (sqrtf(ob2 * g * g / bc2) + eps) + wd * delta);

    float bucket = scale;
    if (delta > 0.5f * bucket || delta < -0.5f * bucket) {
        float w = (float)data_int8[i] * scale + delta;
        float qi = fminf(fmaxf(w / scale, -127.0f), 127.0f);
        float qr = rintf(qi);
        data_int8[i] = (int8_t)qr;
        delta = w - qr * scale;
    }
    mom[ci] = __float2half(mi);
    vel[ci] = __float2half(delta);
}

void mongoose_helix_needle(
    void* data_int8, float* scales, const float* grad,
    void* mom, void* vel, const void* mask,
    float lr, float beta1, float beta2, float bc1, float bc2,
    float eps, float wd, int n, int cols, cudaStream_t stream
) {
    helix_needle_kernel<<<(n+255)/256, 256, 0, stream>>>(
        (int8_t*)data_int8, scales, grad,
        (__half*)mom, (__half*)vel, (const float*)mask,
        lr, beta1, beta2, bc1, bc2, eps, wd, n, cols);
}

// === Needle paired: INT8 DNA-coupled update ===
__global__ void helix_needle_paired_kernel(
    int8_t* __restrict__ d1, int8_t* __restrict__ d2,
    float* __restrict__ s1, float* __restrict__ s2,
    const float* __restrict__ g1, const float* __restrict__ g2,
    __half* __restrict__ m1, __half* __restrict__ m2,
    __half* __restrict__ v1, __half* __restrict__ v2,
    const float* __restrict__ rowMask,
    float lr, float beta1, float beta2, float bc1, float bc2,
    float eps, float wd,
    float backbone1, float glyco1, float hbond1,
    float hbond2, float glyco2, float backbone2,
    float bondStrength, int n, int cols
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int row = i / cols;
    float maskVal = rowMask[row];
    if (maskVal == 0.0f) return;
    int ci = ((int)(maskVal - 1.0f)) * cols + (i % cols);

    float scale1 = s1[row] / 127.0f, scale2 = s2[row] / 127.0f;
    float delta1 = __half2float(v1[ci]), delta2 = __half2float(v2[ci]);
    float mi1 = __half2float(m1[ci]), mi2 = __half2float(m2[ci]);
    float ob1 = 1.0f - beta1, ob2 = 1.0f - beta2;

    float eg1 = g1[i] * glyco1 + g2[i] * hbond1 * bondStrength;
    mi1 = beta1 * mi1 + ob1 * eg1;
    delta1 -= lr * (mi1 / bc1 / (sqrtf(ob2 * eg1 * eg1 / bc2) + eps) + wd * backbone1 * delta1);

    float eg2 = g2[i] * glyco2 + g1[i] * hbond2 * bondStrength;
    mi2 = beta1 * mi2 + ob1 * eg2;
    delta2 -= lr * (mi2 / bc1 / (sqrtf(ob2 * eg2 * eg2 / bc2) + eps) + wd * backbone2 * delta2);

    float b1 = scale1;
    if (delta1 > 0.5f*b1 || delta1 < -0.5f*b1) {
        float w = (float)d1[i]*scale1 + delta1;
        float q = rintf(fminf(fmaxf(w/scale1,-127.f),127.f));
        d1[i] = (int8_t)q; delta1 = w - q*scale1;
    }
    float b2 = scale2;
    if (delta2 > 0.5f*b2 || delta2 < -0.5f*b2) {
        float w = (float)d2[i]*scale2 + delta2;
        float q = rintf(fminf(fmaxf(w/scale2,-127.f),127.f));
        d2[i] = (int8_t)q; delta2 = w - q*scale2;
    }
    m1[ci] = __float2half(mi1); v1[ci] = __float2half(delta1);
    m2[ci] = __float2half(mi2); v2[ci] = __float2half(delta2);
}

void mongoose_helix_needle_paired(
    void* d1, void* d2, float* s1, float* s2,
    const float* g1, const float* g2,
    void* m1, void* m2, void* v1, void* v2,
    const void* mask,
    float lr, float beta1, float beta2, float bc1, float bc2,
    float eps, float wd,
    float backbone1, float glyco1, float hbond1,
    float hbond2, float glyco2, float backbone2,
    float bondStrength, int n, int cols, cudaStream_t stream
) {
    helix_needle_paired_kernel<<<(n+255)/256, 256, 0, stream>>>(
        (int8_t*)d1, (int8_t*)d2, s1, s2, g1, g2,
        (__half*)m1, (__half*)m2, (__half*)v1, (__half*)v2,
        (const float*)mask,
        lr, beta1, beta2, bc1, bc2, eps, wd,
        backbone1, glyco1, hbond1, hbond2, glyco2, backbone2,
        bondStrength, n, cols);
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

// === INT8 Dequantization ===
// Dequantize INT8 weights to FP16 for cuBLAS mixed-precision matmul.
// Each row has a per-row absmax scale: fp16_out = int8_val * (scale / 127.0)
// This is the QLoRA/bitsandbytes pattern — store compressed, dequant on-the-fly.
// Memory-bandwidth limited, nearly free compared to the matmul itself.
//
// data_int8: [rows, cols] INT8 weights
// scales:    [rows] FP32 per-row absmax
// out_fp16:  [rows, cols] FP16 output
__global__ void dequant_int8_to_fp16_kernel(
    const int8_t* data_int8, const float* scales, __half* out_fp16,
    int rows, int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    float s = scales[row] / 127.0f;
    const int8_t* src = data_int8 + row * cols;
    __half* dst = out_fp16 + row * cols;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        dst[j] = __float2half((float)src[j] * s);
    }
}

void mongoose_dequant_int8_to_fp16(
    const void* data_int8, const float* scales, void* out_fp16,
    int rows, int cols, cudaStream_t stream
) {
    int threads = cols < 256 ? cols : 256;
    dequant_int8_to_fp16_kernel<<<rows, threads, 0, stream>>>(
        (const int8_t*)data_int8, scales, (__half*)out_fp16, rows, cols);
}

// === INT8 Dequantization to FP32 ===
// Same as above but outputs FP32. For cases where FP16 cuBLAS path isn't available
// or when we need FP32 precision (norms, biases).
__global__ void dequant_int8_to_fp32_kernel(
    const int8_t* data_int8, const float* scales, float* out_fp32,
    int rows, int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    float s = scales[row] / 127.0f;
    const int8_t* src = data_int8 + row * cols;
    float* dst = out_fp32 + row * cols;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        dst[j] = (float)src[j] * s;
    }
}

void mongoose_dequant_int8_to_fp32(
    const void* data_int8, const float* scales, float* out_fp32,
    int rows, int cols, cudaStream_t stream
) {
    int threads = cols < 256 ? cols : 256;
    dequant_int8_to_fp32_kernel<<<rows, threads, 0, stream>>>(
        (const int8_t*)data_int8, scales, out_fp32, rows, cols);
}

// === Fused Q8 matvec: out[row] = sum_k(act[k] * int8_weight[row,k] * scale[row]/127) ===
// One block per output row. Threads cooperatively reduce the dot product.
__global__ void q8_matvec_kernel(
    const float* act, const int8_t* weight, const float* scales,
    float* out, int K
) {
    int row = blockIdx.x;
    float scale = scales[row] / 127.0f;
    const int8_t* wRow = weight + row * K;

    float sum = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        sum += act[k] * float(wRow[k]) * scale;
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
        out[row] = total;
    }
}

void mongoose_q8_matvec(
    const float* act, const void* weight_int8, const float* scales,
    float* out, int N, int K, cudaStream_t stream
) {
    int threads = K < 1024 ? ((K + 31) / 32) * 32 : 1024;
    if (threads < 32) threads = 32;
    q8_matvec_kernel<<<N, threads, 0, stream>>>(
        act, (const int8_t*)weight_int8, scales, out, K);
}

// === Fused Q4 matvec: out[row] = sum_k(act[k] * dequant4(packed[row,k/2])) ===
__global__ void q4_matvec_kernel(
    const float* act, const uint8_t* weight, const float* scales,
    float* out, int K
) {
    int row = blockIdx.x;
    float scale = scales[row] / 7.0f;
    int halfK = K / 2;
    const uint8_t* wRow = weight + row * halfK;

    float sum = 0.0f;
    for (int k = threadIdx.x; k < halfK; k += blockDim.x) {
        uint8_t packed = wRow[k];
        float w0 = float(int(packed & 0xF) - 8) * scale;
        float w1 = float(int(packed >> 4) - 8) * scale;
        sum += act[k * 2] * w0 + act[k * 2 + 1] * w1;
    }

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
        out[row] = total;
    }
}

void mongoose_q4_matvec(
    const float* act, const void* weight_packed, const float* scales,
    float* out, int N, int K, cudaStream_t stream
) {
    int halfK = K / 2;
    int threads = halfK < 1024 ? ((halfK + 31) / 32) * 32 : 1024;
    if (threads < 32) threads = 32;
    q4_matvec_kernel<<<N, threads, 0, stream>>>(
        act, (const uint8_t*)weight_packed, scales, out, K);
}

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

// Dequant INT8 + FP32 delta → FP32 output.
__global__ void dequant_int8_delta_kernel(
    const int8_t* __restrict__ data, const float* __restrict__ scales,
    const float* __restrict__ delta, float* __restrict__ out, int n, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float scale = scales[i / cols] / 127.0f;
    out[i] = (float)data[i] * scale + delta[i];
}

void mongoose_dequant_int8_delta(
    const void* data, const float* scales, const float* delta,
    float* out, int n, int cols, cudaStream_t stream) {
    dequant_int8_delta_kernel<<<(n+255)/256, 256, 0, stream>>>(
        (const int8_t*)data, scales, delta, out, n, cols);
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

__global__ void causal_attention_fp16_kernel(
    const __half* Q, const __half* K, const __half* V, __half* out,
    int seqLen, int dim, int kvDim, int numHeads, int numKVHeads, int headDim
) {
    int pos = blockIdx.x;
    int head = blockIdx.y;
    int hOff = head * headDim;
    int kvHead = head / (numHeads / numKVHeads);
    int kvOff = kvHead * headDim;

    extern __shared__ float shared[];
    float* scores = shared;

    float scale = rsqrtf((float)headDim);

    for (int j = threadIdx.x; j <= pos; j += blockDim.x) {
        float dot = 0.0f;
        for (int d = 0; d < headDim; d++) {
            dot += __half2float(Q[pos * dim + hOff + d]) * __half2float(K[j * kvDim + kvOff + d]);
        }
        scores[j] = dot * scale;
    }
    __syncthreads();

    float maxVal = -1e30f;
    for (int j = threadIdx.x; j <= pos; j += blockDim.x) {
        if (scores[j] > maxVal) maxVal = scores[j];
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        maxVal = fmaxf(maxVal, __shfl_down_sync(0xffffffff, maxVal, offset));
    __shared__ float blockMax[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    if (lane == 0) blockMax[wid] = maxVal;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = blockMax[0];
        for (int i = 1; i < (blockDim.x + warpSize - 1) / warpSize; i++)
            m = fmaxf(m, blockMax[i]);
        blockMax[0] = m;
    }
    __syncthreads();
    maxVal = blockMax[0];

    float sumExp = 0.0f;
    for (int j = threadIdx.x; j <= pos; j += blockDim.x) {
        scores[j] = expf(scores[j] - maxVal);
        sumExp += scores[j];
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        sumExp += __shfl_down_sync(0xffffffff, sumExp, offset);
    __shared__ float blockSum[32];
    if (lane == 0) blockSum[wid] = sumExp;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0;
        for (int i = 0; i < (blockDim.x + warpSize - 1) / warpSize; i++)
            s += blockSum[i];
        blockSum[0] = s;
    }
    __syncthreads();
    sumExp = blockSum[0];

    for (int j = threadIdx.x; j <= pos; j += blockDim.x) {
        scores[j] /= sumExp;
    }
    __syncthreads();

    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        float val = 0.0f;
        for (int j = 0; j <= pos; j++) {
            val += scores[j] * __half2float(V[j * kvDim + kvOff + d]);
        }
        out[pos * dim + hOff + d] = __float2half(val);
    }
}

void mongoose_causal_attention_gqa_fp16(
    const void* Q, const void* K, const void* V, void* out,
    int seqLen, int dim, int kvDim, int numHeads, int numKVHeads, cudaStream_t stream
) {
    int headDim = dim / numHeads;
    int threads = headDim < 256 ? headDim : 256;
    size_t sharedBytes = seqLen * sizeof(float);
    dim3 grid(seqLen, numHeads);
    causal_attention_fp16_kernel<<<grid, threads, sharedBytes, stream>>>(
        (const __half*)Q, (const __half*)K, (const __half*)V, (__half*)out,
        seqLen, dim, kvDim, numHeads, numKVHeads, headDim);
}

__global__ void silu_gate_mul_fp16_kernel(const __half* gate, const __half* up, __half* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = __half2float(gate[i]);
        float sig = 1.0f / (1.0f + expf(-g));
        out[i] = __float2half(g * sig * __half2float(up[i]));
    }
}

void mongoose_silu_gate_mul_fp16(const void* gate, const void* up, void* out, int n, cudaStream_t stream) {
    silu_gate_mul_fp16_kernel<<<(n+255)/256, 256, 0, stream>>>(
        (const __half*)gate, (const __half*)up, (__half*)out, n);
}

__global__ void silu_gate_backward_fp16_kernel(
    const __half* dOut, const __half* gate, const __half* up,
    __half* dGate, __half* dUp, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = __half2float(gate[i]);
        float sig = 1.0f / (1.0f + expf(-g));
        float silu_g = g * sig;
        float dO = __half2float(dOut[i]);
        float u = __half2float(up[i]);
        dUp[i] = __float2half(dO * silu_g);
        dGate[i] = __float2half(dO * u * (sig + silu_g * (1.0f - sig)));
    }
}

void mongoose_silu_gate_backward_fp16(
    const void* dOut, const void* gate, const void* up,
    void* dGate, void* dUp, int n, cudaStream_t stream
) {
    silu_gate_backward_fp16_kernel<<<(n+255)/256, 256, 0, stream>>>(
        (const __half*)dOut, (const __half*)gate, (const __half*)up,
        (__half*)dGate, (__half*)dUp, n);
}

// === Sparse needle: per-weight update on conductor hot positions ===
// hotIdx[i] = flat position index into data_int8 / fp32_cache / grad.
// nHot threads — one per individual weight position.
// mom/delta compacted to [nHot], indexed by thread ID.
// grad is the full-size gradient buffer — read at hotIdx[tid] for real per-weight gradient.
// If grad is NULL, falls back to signalScale * momentum (forward-only mode).

__global__ void helix_needle_sparse_kernel(
    int8_t* __restrict__ data_int8,
    float* __restrict__ scales,
    float* __restrict__ fp32_cache,
    __half* __restrict__ mom,
    __half* __restrict__ delta_buf,
    const int* __restrict__ hotIdx,
    const float* __restrict__ grad,
    float signalScale,
    float lr, float beta1,
    float wd, int nHot, int cols
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nHot) return;

    int wi = hotIdx[tid];
    int row = wi / cols;

    float scale = scales[row] / 127.0f;
    float mi = __half2float(mom[tid]);
    float delta = __half2float(delta_buf[tid]);

    float g = (grad != NULL) ? grad[wi] : mi * signalScale;
    mi = beta1 * mi + (1.0f - beta1) * g;
    delta -= lr * (mi + wd * delta);

    if (fp32_cache != NULL) {
        float bucket = scale;
        if (delta > 0.5f * bucket || delta < -0.5f * bucket) {
            float w = (float)data_int8[wi] * scale + delta;
            float qi = w / scale;
            qi = fminf(fmaxf(qi, -127.0f), 127.0f);
            float qi_r = rintf(qi);
            data_int8[wi] = (int8_t)qi_r;
            delta = w - qi_r * scale;
            fp32_cache[wi] = qi_r * scale + delta;
        } else {
            fp32_cache[wi] = (float)data_int8[wi] * scale + delta;
        }
    }

    mom[tid] = __float2half(mi);
    delta_buf[tid] = __float2half(delta);
}

void mongoose_helix_needle_sparse(
    void* data_int8, float* scales, float* fp32_cache,
    void* mom, void* delta_buf, const int* hotIdx, const float* grad,
    float signalScale, float lr, float beta1,
    float wd, int nHot, int cols, cudaStream_t stream
) {
    if (nHot <= 0) return;
    int threads = 256;
    int blocks = (nHot + threads - 1) / threads;
    helix_needle_sparse_kernel<<<blocks, threads, 0, stream>>>(
        (int8_t*)data_int8, scales, fp32_cache,
        (__half*)mom, (__half*)delta_buf, hotIdx, grad,
        signalScale, lr, beta1, wd, nHot, cols);
}

// === Inline needle: forward-only, updates FP32 cache before matmul ===
// Dispatches over ALL elements (n = rows*cols), skips frozen rows via mask.
// rowMask[row] = 0: frozen (skip). rowMask[row] > 0: compact_row + 1.
// Momentum is compacted to [nHotRows * cols], indexed by compact row from mask.
// FP32 cache updated in-place — the following matmul sees corrected weights.

__global__ void helix_needle_inline_kernel(
    int8_t* __restrict__ data_int8,
    float* __restrict__ scales,
    float* __restrict__ fp32_cache,
    __half* __restrict__ mom,
    __half* __restrict__ delta_buf,
    const float* __restrict__ rowMask,
    float signalScale,
    float lr, float beta1,
    float wd, int n, int cols
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int row = i / cols;
    int col = i % cols;
    float maskVal = rowMask[row];
    if (maskVal == 0.0f) return;
    int compactRow = (int)(maskVal - 1.0f);
    int ci = compactRow * cols + col;

    float scale = scales[row] / 127.0f;
    float mi = __half2float(mom[ci]);
    float delta = __half2float(delta_buf[ci]);

    float sg = mi * signalScale;
    mi = beta1 * mi + (1.0f - beta1) * sg;
    delta -= lr * (mi + wd * delta);

    float bucket = scale;
    if (delta > 0.5f * bucket || delta < -0.5f * bucket) {
        float w = (float)data_int8[i] * scale + delta;
        float qi = w / scale;
        qi = fminf(fmaxf(qi, -127.0f), 127.0f);
        float qi_r = rintf(qi);
        data_int8[i] = (int8_t)qi_r;
        delta = w - qi_r * scale;
        fp32_cache[i] = qi_r * scale + delta;
    } else {
        fp32_cache[i] = (float)data_int8[i] * scale + delta;
    }

    mom[ci] = __float2half(mi);
    delta_buf[ci] = __float2half(delta);
}

void mongoose_helix_needle_inline(
    void* data_int8, float* scales, float* fp32_cache,
    void* mom, void* delta_buf, const void* mask,
    float signalScale, float lr, float beta1,
    float wd, int n, int cols, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    helix_needle_inline_kernel<<<blocks, threads, 0, stream>>>(
        (int8_t*)data_int8, scales, fp32_cache,
        (__half*)mom, (__half*)delta_buf, (const float*)mask,
        signalScale, lr, beta1, wd, n, cols);
}

} // extern "C"
