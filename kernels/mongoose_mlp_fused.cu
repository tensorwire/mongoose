// mongoose_mlp_fused.cu — Single-launch fused MLP training kernel.
// Forward + BCE loss + backward + AdamW in ONE cooperative kernel.
// No cuBLAS, no multi-kernel orchestration, no buffer aliasing.
//
// Uses cudaLaunchCooperativeKernel for grid.sync() between layers.
// Compile with: -rdc=true (relocatable device code for cooperative groups)

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define MLP_TILE_K 32
#define MLP_MAX_LAYERS 8
#define MLP_MAX_DIM 1024

// --- Device helpers ---

__device__ void mlp_gemm_fwd(
    const float* A, const float* B_W, float* C,
    int M, int K, int N,
    int rowStart, int rowEnd
) {
    // A[M,K] @ B_W[N,K]^T -> C[M,N]  (only rows rowStart..rowEnd-1)
    // Each thread computes multiple output elements
    int nRows = rowEnd - rowStart;
    int nElems = nRows * N;
    for (int idx = threadIdx.x; idx < nElems; idx += blockDim.x) {
        int r = rowStart + idx / N;
        int c = idx % N;
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[r * K + k] * B_W[c * K + k];
        }
        C[r * N + c] = sum;
    }
}

__device__ void mlp_gemm_bwd_dw(
    const float* dOut, const float* input, float* dW,
    int B, int outDim, int inDim,
    int rowStart, int rowEnd
) {
    // dW[outDim, inDim] += (dOut[B, outDim]^T @ input[B, inDim]) / B
    // Only accumulate from rows rowStart..rowEnd-1
    // Use atomicAdd since multiple blocks contribute
    float invB = 1.0f / (float)B;
    int nElems = outDim * inDim;
    for (int idx = threadIdx.x; idx < nElems; idx += blockDim.x) {
        int o = idx / inDim;
        int k = idx % inDim;
        float sum = 0.0f;
        for (int b = rowStart; b < rowEnd; b++) {
            sum += dOut[b * outDim + o] * input[b * inDim + k];
        }
        atomicAdd(&dW[o * inDim + k], sum * invB);
    }
}

__device__ void mlp_gemm_bwd_dinput(
    const float* dOut, const float* W, float* dInput,
    int B, int outDim, int inDim,
    int rowStart, int rowEnd
) {
    // dInput[B, inDim] = dOut[B, outDim] @ W[outDim, inDim]
    // Only rows rowStart..rowEnd-1
    int nRows = rowEnd - rowStart;
    int nElems = nRows * inDim;
    for (int idx = threadIdx.x; idx < nElems; idx += blockDim.x) {
        int r = rowStart + idx / inDim;
        int c = idx % inDim;
        float sum = 0.0f;
        for (int o = 0; o < outDim; o++) {
            sum += dOut[r * outDim + o] * W[o * inDim + c];
        }
        dInput[r * inDim + c] = sum;
    }
}

__device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ float block_reduce_sum(float val) {
    __shared__ float warp_sums[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    val = warp_reduce_sum(val);
    if (lane == 0) warp_sums[wid] = val;
    __syncthreads();
    int nWarps = (blockDim.x + warpSize - 1) / warpSize;
    val = (threadIdx.x < nWarps) ? warp_sums[threadIdx.x] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val; // only thread 0 has the full sum
}

// Struct to pass all layer info via global memory (device-side array)
struct MLPLayerDesc {
    int inDim, outDim;
    int hasBN;  // 1 if this layer has BatchNorm
};

// --- The fused kernel ---
__global__ void mlp_fused_train_kernel(
    // Layer descriptors (device array)
    const MLPLayerDesc* layers, int nLayers,
    // Per-layer pointer arrays (device arrays of pointers)
    float** W, float** bias,
    float** gamma, float** beta,
    float** runMean, float** runVar,
    float** mW, float** vW,
    float** mB, float** vB,
    float** mG, float** vG,
    float** mBt, float** vBt,
    // Scratch (device arrays of pointers)
    float** act,     // [nLayers] activation buffers
    float** preBN,   // [nLayers] saved pre-BN for backward
    float** preReLU, // [nLayers] saved pre-ReLU for backward
    float** masks,   // [nLayers] dropout masks
    float** dW_buf,  // [nLayers] weight gradient accumulators
    float** dB_buf,  // [nLayers] bias gradient accumulators
    float** dGamma_buf, // [nLayers] BN gamma gradient
    float** dBeta_buf,  // [nLayers] BN beta gradient
    float* gradBuf,  // [B] BCE gradient
    // BN reduction workspace (device, zeroed before launch)
    float* bnReduceSum,   // [MAX_DIM] — global accumulator for BN mean
    float* bnReduceSumSq, // [MAX_DIM] — global accumulator for BN variance
    // Batch data
    float* input,    // [B, nFeatures]
    float* targets,  // [B]
    float* lossOut,  // [1]
    // Hyperparameters
    float lr, float wd, float b1, float b2, float bc1, float bc2,
    float bnMomentum, float eps, float dropoutP,
    unsigned long long dropSeed, unsigned long long dropCounter,
    int B
) {
    cg::grid_group grid = cg::this_grid();
    int nBlocks = gridDim.x;
    int bid = blockIdx.x;

    // Each block owns a contiguous slice of the batch
    int rowsPerBlock = (B + nBlocks - 1) / nBlocks;
    int rowStart = bid * rowsPerBlock;
    int rowEnd = rowStart + rowsPerBlock;
    if (rowEnd > B) rowEnd = B;
    if (rowStart >= B) rowStart = rowEnd = B; // empty block

    // ======================== FORWARD ========================
    for (int li = 0; li < nLayers; li++) {
        int inDim = layers[li].inDim;
        int outDim = layers[li].outDim;
        int isLast = (li == nLayers - 1);
        const float* layerInput = (li == 0) ? input : act[li - 1];

        // 1. Matmul: act[li] = layerInput @ W[li]^T
        mlp_gemm_fwd(layerInput, W[li], act[li], B, inDim, outDim, rowStart, rowEnd);
        __syncthreads();

        // 2. Bias add
        {
            int nElems = (rowEnd - rowStart) * outDim;
            for (int idx = threadIdx.x; idx < nElems; idx += blockDim.x) {
                int r = rowStart + idx / outDim;
                int c = idx % outDim;
                act[li][r * outDim + c] += bias[li][c];
            }
        }

        // 3. Save pre-BN
        {
            int nElems = (rowEnd - rowStart) * outDim;
            for (int idx = threadIdx.x; idx < nElems; idx += blockDim.x) {
                preBN[li][rowStart * outDim + idx] = act[li][rowStart * outDim + idx];
            }
        }

        // 4. BatchNorm (hidden layers only) — two-pass for numerical stability
        if (layers[li].hasBN && !isLast) {
            // Pass 1: accumulate sum for mean
            for (int d = threadIdx.x; d < outDim; d += blockDim.x) {
                float sum = 0.0f;
                for (int r = rowStart; r < rowEnd; r++) {
                    sum += act[li][r * outDim + d];
                }
                atomicAdd(&bnReduceSum[d], sum);
            }
            grid.sync();

            // Compute mean, clear sum buffer, then accumulate sum((x-mean)^2)
            for (int d = threadIdx.x; d < outDim; d += blockDim.x) {
                float rawSum = bnReduceSum[d];
                float mean = rawSum / (float)B;
                if (li == 0 && d == 0 && bid == 0) {
                    printf("[BN-DIAG] L0 feat[0]: rawSum=%.6f B=%d mean=%.6f nBlocks=%d rowStart=%d rowEnd=%d\n",
                        rawSum, B, mean, gridDim.x, rowStart, rowEnd);
                }
                bnReduceSum[d] = mean;
            }
            grid.sync();

            // Pass 2: accumulate sum of squared deviations from mean
            for (int d = threadIdx.x; d < outDim; d += blockDim.x) {
                float mean = bnReduceSum[d];
                float sumDiffSq = 0.0f;
                for (int r = rowStart; r < rowEnd; r++) {
                    float diff = act[li][r * outDim + d] - mean;
                    sumDiffSq += diff * diff;
                }
                atomicAdd(&bnReduceSumSq[d], sumDiffSq);
            }
            grid.sync();

            // Normalize + scale + shift + update running stats
            for (int d = threadIdx.x; d < outDim; d += blockDim.x) {
                float mean = bnReduceSum[d];
                float variance = bnReduceSumSq[d] / (float)B;
                float invstd = rsqrtf(variance + 1e-5f);

                if (bid == 0) {
                    runMean[li][d] = (1.0f - bnMomentum) * runMean[li][d] + bnMomentum * mean;
                    runVar[li][d] = (1.0f - bnMomentum) * runVar[li][d] + bnMomentum * variance;
                }

                float g = gamma[li][d];
                float b_val = beta[li][d];
                for (int r = rowStart; r < rowEnd; r++) {
                    float xval = act[li][r * outDim + d];
                    float xhat = (xval - mean) * invstd;
                    float result = g * xhat + b_val;
                    if (li == 0 && d == 0 && r == 0 && bid == 0) {
                        printf("[BN-EL] L0 [0,0]: xval=%.6f mean=%.6f var=%.6f invstd=%.6f xhat=%.6f gamma=%.6f beta=%.6f result=%.6f\n",
                            xval, mean, variance, invstd, xhat, g, b_val, result);
                    }
                    act[li][r * outDim + d] = result;
                }
            }

            // Clear reduction buffers for next layer
            grid.sync();
            for (int d = threadIdx.x; d < outDim; d += blockDim.x) {
                if (bid == 0) {
                    bnReduceSum[d] = 0.0f;
                    bnReduceSumSq[d] = 0.0f;
                }
            }
        }

        // 5. Save pre-ReLU (for ReLU backward)
        if (!isLast) {
            int nElems = (rowEnd - rowStart) * outDim;
            for (int idx = threadIdx.x; idx < nElems; idx += blockDim.x) {
                preReLU[li][rowStart * outDim + idx] = act[li][rowStart * outDim + idx];
            }
        }

        // 6. ReLU (hidden layers only)
        if (!isLast) {
            int nElems = (rowEnd - rowStart) * outDim;
            for (int idx = threadIdx.x; idx < nElems; idx += blockDim.x) {
                int pos = rowStart * outDim + idx;
                act[li][pos] = fmaxf(act[li][pos], 0.0f);
            }
        }

        // 7. Dropout (hidden layers only, training)
        if (!isLast && dropoutP > 0.0f) {
            int nElems = (rowEnd - rowStart) * outDim;
            float scale = 1.0f / (1.0f - dropoutP);
            for (int idx = threadIdx.x; idx < nElems; idx += blockDim.x) {
                int pos = rowStart * outDim + idx;
                // Unique sequence per (element, layer), step as offset
                // Ensures fully independent masks across layers and steps
                int seqId = pos + li * B * outDim;
                curandStatePhilox4_32_10_t state;
                curand_init(dropSeed, seqId, dropCounter, &state);
                float r = curand_uniform(&state);
                if (r < dropoutP) {
                    act[li][pos] = 0.0f;
                    masks[li][pos] = 0.0f;
                } else {
                    act[li][pos] *= scale;
                    masks[li][pos] = scale;
                }
            }
        }

        grid.sync();
    }

    // ======================== LOSS ========================
    // BCEWithLogitsLoss: loss = max(x,0) - x*t + log(1+exp(-|x|))
    // grad = (sigmoid(x) - t) / B
    {
        float* logits = act[nLayers - 1]; // [B, 1]
        float localLoss = 0.0f;
        for (int i = rowStart + threadIdx.x; i < rowEnd; i += blockDim.x) {
            float x = logits[i];
            float t = targets[i];
            float absX = fabsf(x);
            float loss = fmaxf(x, 0.0f) - x * t + logf(1.0f + expf(-absX));
            localLoss += loss;
            float sig = 1.0f / (1.0f + expf(-x));
            gradBuf[i] = (sig - t) / (float)B;
        }
        // Reduce loss across block
        localLoss = block_reduce_sum(localLoss);
        if (threadIdx.x == 0) {
            atomicAdd(lossOut, localLoss / (float)B);
        }
    }

    grid.sync();

    // ======================== BACKWARD ========================
    // dOut starts as gradBuf (BCE gradient)
    for (int li = nLayers - 1; li >= 0; li--) {
        int inDim = layers[li].inDim;
        int outDim = layers[li].outDim;
        int isLast = (li == nLayers - 1);
        float* dOut = (li == nLayers - 1) ? gradBuf : act[li]; // reuse act as dOut for hidden

        // 1. Dropout backward (hidden only)
        if (!isLast && dropoutP > 0.0f) {
            int nElems = (rowEnd - rowStart) * outDim;
            for (int idx = threadIdx.x; idx < nElems; idx += blockDim.x) {
                int pos = rowStart * outDim + idx;
                dOut[pos] *= masks[li][pos];
            }
        }

        // 2. ReLU backward (hidden only)
        if (!isLast) {
            int nElems = (rowEnd - rowStart) * outDim;
            for (int idx = threadIdx.x; idx < nElems; idx += blockDim.x) {
                int pos = rowStart * outDim + idx;
                dOut[pos] *= (preReLU[li][pos] > 0.0f) ? 1.0f : 0.0f;
            }
        }

        // 3. BatchNorm backward — two-pass variance for numerical stability
        if (layers[li].hasBN && !isLast) {
            // Pass 1: accumulate sum for mean
            for (int d = threadIdx.x; d < outDim; d += blockDim.x) {
                float sum = 0.0f;
                for (int r = rowStart; r < rowEnd; r++) {
                    sum += preBN[li][r * outDim + d];
                }
                atomicAdd(&bnReduceSum[d], sum);
            }
            grid.sync();

            // Compute mean, store in bnReduceSum
            for (int d = threadIdx.x; d < outDim; d += blockDim.x) {
                bnReduceSum[d] = bnReduceSum[d] / (float)B;
            }
            grid.sync();

            // Pass 2: accumulate sum((x - mean)^2)
            for (int d = threadIdx.x; d < outDim; d += blockDim.x) {
                float mean = bnReduceSum[d];
                float sumDiffSq = 0.0f;
                for (int r = rowStart; r < rowEnd; r++) {
                    float diff = preBN[li][r * outDim + d] - mean;
                    sumDiffSq += diff * diff;
                }
                atomicAdd(&bnReduceSumSq[d], sumDiffSq);
            }
            grid.sync();

            // Compute dGamma, dBeta
            for (int d = threadIdx.x; d < outDim; d += blockDim.x) {
                float mean = bnReduceSum[d];
                float variance = bnReduceSumSq[d] / (float)B;
                float invstd = rsqrtf(variance + 1e-5f);

                float dgamma_local = 0.0f, dbeta_local = 0.0f;
                for (int r = rowStart; r < rowEnd; r++) {
                    float xhat = (preBN[li][r * outDim + d] - mean) * invstd;
                    dgamma_local += dOut[r * outDim + d] * xhat;
                    dbeta_local += dOut[r * outDim + d];
                }
                atomicAdd(&dGamma_buf[li][d], dgamma_local / (float)B);
                atomicAdd(&dBeta_buf[li][d], dbeta_local / (float)B);
            }
            grid.sync();

            // Compute dx
            for (int d = threadIdx.x; d < outDim; d += blockDim.x) {
                float mean = bnReduceSum[d];
                float variance = bnReduceSumSq[d] / (float)B;
                float invstd = rsqrtf(variance + 1e-5f);
                float g = gamma[li][d];
                float dg = dGamma_buf[li][d] * (float)B;
                float db = dBeta_buf[li][d] * (float)B;
                float invB = 1.0f / (float)B;

                for (int r = rowStart; r < rowEnd; r++) {
                    float xhat = (preBN[li][r * outDim + d] - mean) * invstd;
                    float dxhat = g * dOut[r * outDim + d];
                    dOut[r * outDim + d] = invstd * (dxhat - invB * (db * g + dg * g * xhat));
                }
            }

            // Clear reduction buffers
            grid.sync();
            for (int d = threadIdx.x; d < outDim; d += blockDim.x) {
                if (bid == 0) {
                    bnReduceSum[d] = 0.0f;
                    bnReduceSumSq[d] = 0.0f;
                }
            }
            grid.sync();
        }

        // 4. Weight gradient: dW += dOut^T @ layerInput (atomicAdd across blocks)
        const float* layerInput = (li == 0) ? input : act[li - 1];
        mlp_gemm_bwd_dw(dOut, layerInput, dW_buf[li], B, outDim, inDim, rowStart, rowEnd);

        // 5. Bias gradient: dB += mean(dOut, axis=0)
        for (int d = threadIdx.x; d < outDim; d += blockDim.x) {
            float sum = 0.0f;
            for (int r = rowStart; r < rowEnd; r++) {
                sum += dOut[r * outDim + d];
            }
            atomicAdd(&dB_buf[li][d], sum / (float)B);
        }

        // 6. Input gradient: dInput = dOut @ W (write into act[li-1] for next layer backward)
        if (li > 0) {
            // Write into act[li-1] which becomes dOut for layer li-1
            mlp_gemm_bwd_dinput(dOut, W[li], act[li - 1], B, outDim, inDim, rowStart, rowEnd);
        }

        grid.sync();
    }

    // ======================== OPTIMIZER (AdamW) ========================
    // Each thread handles a partition of ALL parameters
    int totalThreads = nBlocks * blockDim.x;
    int globalTid = bid * blockDim.x + threadIdx.x;

    for (int li = 0; li < nLayers; li++) {
        int inDim = layers[li].inDim;
        int outDim = layers[li].outDim;
        int isLast = (li == nLayers - 1);

        // Weight update
        int nW = outDim * inDim;
        for (int j = globalTid; j < nW; j += totalThreads) {
            float g_val = dW_buf[li][j];
            float mi = b1 * mW[li][j] + (1.0f - b1) * g_val;
            float vi = b2 * vW[li][j] + (1.0f - b2) * g_val * g_val;
            mW[li][j] = mi;
            vW[li][j] = vi;
            float mhat = mi / bc1;
            float vhat = vi / bc2;
            W[li][j] -= lr * (mhat / (sqrtf(vhat) + eps) + wd * W[li][j]);
        }

        // Bias update
        for (int j = globalTid; j < outDim; j += totalThreads) {
            float g_val = dB_buf[li][j];
            float mi = b1 * mB[li][j] + (1.0f - b1) * g_val;
            float vi = b2 * vB[li][j] + (1.0f - b2) * g_val * g_val;
            mB[li][j] = mi;
            vB[li][j] = vi;
            float mhat = mi / bc1;
            float vhat = vi / bc2;
            bias[li][j] -= lr * (mhat / (sqrtf(vhat) + eps) + wd * bias[li][j]);
        }

        // BN gamma/beta update (no weight decay for BN params)
        if (layers[li].hasBN && !isLast) {
            for (int j = globalTid; j < outDim; j += totalThreads) {
                float g_val = dGamma_buf[li][j];
                float mi = b1 * mG[li][j] + (1.0f - b1) * g_val;
                float vi = b2 * vG[li][j] + (1.0f - b2) * g_val * g_val;
                mG[li][j] = mi;
                vG[li][j] = vi;
                float mhat = mi / bc1;
                float vhat = vi / bc2;
                gamma[li][j] -= lr * (mhat / (sqrtf(vhat) + eps));
            }
            for (int j = globalTid; j < outDim; j += totalThreads) {
                float g_val = dBeta_buf[li][j];
                float mi = b1 * mBt[li][j] + (1.0f - b1) * g_val;
                float vi = b2 * vBt[li][j] + (1.0f - b2) * g_val * g_val;
                mBt[li][j] = mi;
                vBt[li][j] = vi;
                float mhat = mi / bc1;
                float vhat = vi / bc2;
                beta[li][j] -= lr * (mhat / (sqrtf(vhat) + eps));
            }
        }
    }
}

// --- Host wrapper (extern "C" for dlsym) ---

extern "C" void mongoose_mlp_fused_train(
    MLPLayerDesc* d_layers, int nLayers,
    float** d_W, float** d_bias,
    float** d_gamma, float** d_beta,
    float** d_runMean, float** d_runVar,
    float** d_mW, float** d_vW,
    float** d_mB, float** d_vB,
    float** d_mG, float** d_vG,
    float** d_mBt, float** d_vBt,
    float** d_act, float** d_preBN, float** d_preReLU,
    float** d_masks, float** d_dW, float** d_dB,
    float** d_dGamma, float** d_dBeta,
    float* d_gradBuf,
    float* d_bnReduceSum, float* d_bnReduceSumSq,
    float* d_input, float* d_targets, float* d_lossOut,
    float lr, float wd, float b1, float b2, float bc1, float bc2,
    float bnMomentum, float eps_val, float dropoutP,
    unsigned long long dropSeed, unsigned long long dropCounter,
    int B, int maxDim,
    cudaStream_t stream
) {
    // Zero gradient accumulators and loss
    for (int i = 0; i < nLayers; i++) {
        // We need host-side pointers to zero device memory — caller handles this
    }
    cudaMemsetAsync(d_lossOut, 0, sizeof(float), stream);
    cudaMemsetAsync(d_bnReduceSum, 0, maxDim * sizeof(float), stream);
    cudaMemsetAsync(d_bnReduceSumSq, 0, maxDim * sizeof(float), stream);

    int blockSize = 256;
    int nBlocks = (B + 15) / 16; // TILE_B = 16

    // Query max blocks for cooperative launch
    int maxBlocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocks, mlp_fused_train_kernel, blockSize, 0);
    int numSMs = 0;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    int maxCoopBlocks = maxBlocks * numSMs;
    if (nBlocks > maxCoopBlocks) nBlocks = maxCoopBlocks;
    if (nBlocks < 1) nBlocks = 1;

    static int logged = 0;
    if (!logged) {
        fprintf(stderr, "[FUSED] launch: %d blocks x %d threads (maxCoop=%d, SMs=%d, maxPerSM=%d, requested=%d)\n",
            nBlocks, blockSize, maxCoopBlocks, numSMs, maxBlocks, (B + 15) / 16);
        logged = 1;
    }

    void* args[] = {
        &d_layers, &nLayers,
        &d_W, &d_bias,
        &d_gamma, &d_beta,
        &d_runMean, &d_runVar,
        &d_mW, &d_vW,
        &d_mB, &d_vB,
        &d_mG, &d_vG,
        &d_mBt, &d_vBt,
        &d_act, &d_preBN, &d_preReLU,
        &d_masks, &d_dW, &d_dB,
        &d_dGamma, &d_dBeta,
        &d_gradBuf,
        &d_bnReduceSum, &d_bnReduceSumSq,
        &d_input, &d_targets, &d_lossOut,
        &lr, &wd, &b1, &b2, &bc1, &bc2,
        &bnMomentum, &eps_val, &dropoutP,
        &dropSeed, &dropCounter,
        &B
    };

    cudaLaunchCooperativeKernel(
        (void*)mlp_fused_train_kernel,
        dim3(nBlocks), dim3(blockSize),
        args, 0, stream
    );
}
