// mongoose_loss.cu — loss kernels
// Auto-extracted from mongoose.cu. Do not edit directly.

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

