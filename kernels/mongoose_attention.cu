// mongoose_attention.cu — attention kernels
// Auto-extracted from mongoose.cu. Do not edit directly.

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

