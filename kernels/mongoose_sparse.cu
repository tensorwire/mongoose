// mongoose_sparse.cu — sparse kernels
// Auto-extracted from mongoose.cu. Do not edit directly.

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

