//go:build linux && cgo

package mongoose

/*
#cgo LDFLAGS: -lcublas -lcublasLt -lcudart
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// tw_cublas_handle is defined in cuda.go's C block
extern cublasHandle_t tw_cublas_handle;

// GPU memory management
void* tw_gpu_alloc(size_t bytes) {
    void* ptr;
    cudaMalloc(&ptr, bytes);
    return ptr;
}

void tw_gpu_free(void* ptr) {
    cudaFree(ptr);
}

void tw_gpu_upload(void* dst, const void* src, size_t bytes) {
    cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
}

void tw_gpu_download(void* dst, const void* src, size_t bytes) {
    cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
}

void tw_gpu_zero(void* ptr, size_t bytes) {
    cudaMemset(ptr, 0, bytes);
}

// Pinned host memory — page-locked, L3-resident, zero-copy GPU access
void* tw_gpu_alloc_pinned(size_t bytes) {
    void* ptr = NULL;
    cudaError_t err = cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault);
    if (err != cudaSuccess) return NULL;
    return ptr;
}

int tw_register_host_memory(void* ptr, size_t bytes) {
    cudaError_t err = cudaHostRegister(ptr, bytes, cudaHostRegisterDefault);
    return (int)err;
}

// FP32 matmul where A is on device, B and C are pinned host memory.
// cuBLAS reads B from L3 cache, writes C to L3 cache. Zero upload for B.
int tw_gpu_sgemm_l3(const float* dA, const float* pinnedB, float* pinnedC, int m, int k, int n) {
    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t s = cublasGemmEx(tw_cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        pinnedB, CUDA_R_32F, n,
        dA, CUDA_R_32F, k,
        &beta,
        pinnedC, CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return (int)s;
}

// C = A^T @ B where A is on device, B and C are pinned.
int tw_gpu_sgemm_transA_l3(const float* dA, const float* pinnedB, float* pinnedC, int m, int k, int n) {
    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t s = cublasGemmEx(tw_cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        n, k, m,
        &alpha,
        pinnedB, CUDA_R_32F, n,
        dA, CUDA_R_32F, k,
        &beta,
        pinnedC, CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return (int)s;
}

// C = A @ B^T where A is pinned, B is on device, C is pinned.
// This is the main inference/training matmul: hidden (pinned) @ weight^T (device) → output (pinned)
int tw_gpu_sgemm_transB_l3(const float* pinnedA, const float* dB, float* pinnedC, int m, int k, int n) {
    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t s = cublasGemmEx(tw_cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        dB, CUDA_R_32F, k,
        pinnedA, CUDA_R_32F, k,
        &beta,
        pinnedC, CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return (int)s;
}

// FP32 matmul with TF32 tensor cores
int tw_gpu_sgemm(const float* dA, const float* dB, float* dC, int m, int k, int n) {
    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t s = cublasGemmEx(tw_cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        dB, CUDA_R_32F, n,
        dA, CUDA_R_32F, k,
        &beta,
        dC, CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return (int)s;
}

// FP16 matmul with FP32 accumulation — maximum tensor core throughput
// Inputs: FP16 on GPU, output: FP32 on GPU
// This is what PyTorch mixed-precision training uses
int tw_gpu_hgemm(const void* dA, const void* dB, float* dC, int m, int k, int n) {
    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t s = cublasGemmEx(tw_cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        dB, CUDA_R_16F, n,
        dA, CUDA_R_16F, k,
        &beta,
        dC, CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return (int)s;
}

// --- cublasLt for maximum FP16 throughput ---
// PyTorch uses cublasLtMatmul (not cublasGemmEx) with:
//   - 32 MiB workspace for split-K algorithms
//   - Heuristic algo selection
//   - FP16 output (not FP32)
//   - Alignment hints

static cublasLtHandle_t tw_lt_handle = NULL;
static void* tw_lt_workspace = NULL;
static size_t tw_lt_workspace_size = 32 * 1024 * 1024; // 32 MiB, matches PyTorch on Blackwell

int tw_cublaslt_init() {
    if (tw_lt_handle) return 0;
    cublasLtCreate(&tw_lt_handle);
    cudaMalloc(&tw_lt_workspace, tw_lt_workspace_size);
    return tw_lt_handle && tw_lt_workspace ? 0 : -1;
}

// Compute alignment of a pointer (used for cuBLASLt alignment hints)
static uint32_t tw_ptr_alignment(uintptr_t ptr) {
    // Find largest power-of-2 alignment, capped at 256
    for (uint32_t a = 256; a > 1; a >>= 1) {
        if ((ptr & (a - 1)) == 0) return a;
    }
    return 1;
}

// --- Auto-tuning algo cache ---
// On first call for a given (m,k,n) shape, benchmark all candidate algos
// and cache the fastest. Subsequent calls use the cached winner.
// This is what PyTorch does — we were leaving 5% on the table by taking
// the first heuristic suggestion without benchmarking alternatives.

#define ALGO_CACHE_SIZE 64

typedef struct {
    int m, k, n;
    cublasLtMatmulAlgo_t algo;
    int valid;
} algo_cache_entry_t;

static algo_cache_entry_t tw_algo_cache[ALGO_CACHE_SIZE];
static int tw_algo_cache_count = 0;

static cublasLtMatmulAlgo_t* tw_find_cached_algo(int m, int k, int n) {
    for (int i = 0; i < tw_algo_cache_count; i++) {
        if (tw_algo_cache[i].m == m && tw_algo_cache[i].k == k && tw_algo_cache[i].n == n) {
            return &tw_algo_cache[i].algo;
        }
    }
    return NULL;
}

static void tw_cache_algo(int m, int k, int n, cublasLtMatmulAlgo_t algo) {
    if (tw_algo_cache_count < ALGO_CACHE_SIZE) {
        tw_algo_cache[tw_algo_cache_count].m = m;
        tw_algo_cache[tw_algo_cache_count].k = k;
        tw_algo_cache[tw_algo_cache_count].n = n;
        tw_algo_cache[tw_algo_cache_count].algo = algo;
        tw_algo_cache[tw_algo_cache_count].valid = 1;
        tw_algo_cache_count++;
    }
}

// FP16 matmul via cublasLtMatmul with auto-tuning algo selection.
// First call for each (m,k,n) shape: benchmarks up to 8 candidate algos,
// picks the fastest, caches it. Subsequent calls use the cached winner.
int tw_gpu_hgemm_lt(const void* dA, const void* dB, void* dC, int m, int k, int n) {
    if (!tw_lt_handle) tw_cublaslt_init();

    float alpha = 1.0f, beta = 0.0f;

    // Check algo cache first
    cublasLtMatmulAlgo_t* cached = tw_find_cached_algo(m, k, n);
    if (cached) {
        // Fast path: use cached best algo, minimal setup
        cublasLtMatmulDesc_t opDesc;
        cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        cublasOperation_t opN = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
        cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

        cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
        cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, n, k, n);
        cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, k, m, k);
        cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, n, m, n);

        cublasStatus_t status = cublasLtMatmul(tw_lt_handle, opDesc,
            &alpha, dB, Adesc, dA, Bdesc,
            &beta, dC, Cdesc, dC, Cdesc,
            cached, tw_lt_workspace, tw_lt_workspace_size, 0);

        cublasLtMatrixLayoutDestroy(Adesc);
        cublasLtMatrixLayoutDestroy(Bdesc);
        cublasLtMatrixLayoutDestroy(Cdesc);
        cublasLtMatmulDescDestroy(opDesc);
        return (int)status;
    }

    // Slow path: auto-tune — benchmark multiple algos, cache the winner

    cublasLtMatmulDesc_t opDesc;
    cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t opN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, n, k, n);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, k, m, k);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, n, m, n);

    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &tw_lt_workspace_size, sizeof(tw_lt_workspace_size));

    uint32_t aAlign = tw_ptr_alignment((uintptr_t)dB);
    uint32_t bAlign = tw_ptr_alignment((uintptr_t)dA);
    uint32_t cAlign = tw_ptr_alignment((uintptr_t)dC);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &aAlign, sizeof(aAlign));
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &bAlign, sizeof(bAlign));
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &cAlign, sizeof(cAlign));

    // Request up to 8 candidate algos
    #define MAX_ALGOS 8
    cublasLtMatmulHeuristicResult_t heurs[MAX_ALGOS];
    int nResults = 0;
    cublasLtMatmulAlgoGetHeuristic(tw_lt_handle, opDesc, Adesc, Bdesc, Cdesc, Cdesc,
        pref, MAX_ALGOS, heurs, &nResults);

    cublasStatus_t status = CUBLAS_STATUS_NOT_SUPPORTED;

    if (nResults == 1) {
        // Only one algo — use it directly
        status = cublasLtMatmul(tw_lt_handle, opDesc,
            &alpha, dB, Adesc, dA, Bdesc,
            &beta, dC, Cdesc, dC, Cdesc,
            &heurs[0].algo, tw_lt_workspace, tw_lt_workspace_size, 0);
        tw_cache_algo(m, k, n, heurs[0].algo);
    } else if (nResults > 1) {
        // Benchmark each algo — 3 runs each, pick fastest
        int bestIdx = 0;
        float bestTime = 1e30f;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        for (int a = 0; a < nResults; a++) {
            // Warmup
            cublasLtMatmul(tw_lt_handle, opDesc,
                &alpha, dB, Adesc, dA, Bdesc,
                &beta, dC, Cdesc, dC, Cdesc,
                &heurs[a].algo, tw_lt_workspace, tw_lt_workspace_size, 0);

            // Timed run
            cudaEventRecord(start, 0);
            for (int r = 0; r < 3; r++) {
                cublasLtMatmul(tw_lt_handle, opDesc,
                    &alpha, dB, Adesc, dA, Bdesc,
                    &beta, dC, Cdesc, dC, Cdesc,
                    &heurs[a].algo, tw_lt_workspace, tw_lt_workspace_size, 0);
            }
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);

            float ms = 0;
            cudaEventElapsedTime(&ms, start, stop);
            if (ms < bestTime) {
                bestTime = ms;
                bestIdx = a;
            }
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // Run the winner for the actual result
        status = cublasLtMatmul(tw_lt_handle, opDesc,
            &alpha, dB, Adesc, dA, Bdesc,
            &beta, dC, Cdesc, dC, Cdesc,
            &heurs[bestIdx].algo, tw_lt_workspace, tw_lt_workspace_size, 0);

        tw_cache_algo(m, k, n, heurs[bestIdx].algo);
    }

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatmulDescDestroy(opDesc);

    return (int)status;
}

// Upload FP32 host data as FP16 on GPU (conversion on CPU, upload FP16)
void* tw_gpu_upload_fp16(const float* hostData, int n) {
    // Convert on CPU
    unsigned short* hostFP16 = (unsigned short*)malloc(n * 2);
    for (int i = 0; i < n; i++) {
        // Simple FP32→FP16 conversion
        unsigned int bits = *(unsigned int*)&hostData[i];
        unsigned int sign = (bits >> 16) & 0x8000;
        int exp = ((bits >> 23) & 0xFF) - 127 + 15;
        unsigned int frac = bits & 0x007FFFFF;
        if (exp <= 0) hostFP16[i] = sign;
        else if (exp >= 31) hostFP16[i] = sign | 0x7C00;
        else hostFP16[i] = sign | (exp << 10) | (frac >> 13);
    }
    void* dOut;
    cudaMalloc(&dOut, n * 2);
    cudaMemcpy(dOut, hostFP16, n * 2, cudaMemcpyHostToDevice);
    free(hostFP16);
    return dOut;
}

// C = A^T @ B: A[m,k]->A^T[k,m], B[m,n], C[k,n] row-major
int tw_gpu_sgemm_transA(const float* dA, const float* dB, float* dC, int m, int k, int n) {
    float alpha = 1.0f, beta = 0.0f;
    return (int)cublasGemmEx(tw_cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_T, n, k, m,
        &alpha, dB, CUDA_R_32F, n, dA, CUDA_R_32F, k,
        &beta, dC, CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// C = A @ B^T: A[m,k], B[n,k]->B^T[k,n], C[m,n] row-major
int tw_gpu_sgemm_transB(const float* dA, const float* dB, float* dC, int m, int k, int n) {
    float alpha = 1.0f, beta = 0.0f;
    return (int)cublasGemmEx(tw_cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
        &alpha, dB, CUDA_R_32F, k, dA, CUDA_R_32F, k,
        &beta, dC, CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// C = A @ B^T: FP16 inputs, FP32 output. A[m,k] FP16, B[n,k] FP16, C[m,n] FP32.
// Uses cublasLtMatmul with explicit 32MB workspace cap to prevent cuBLAS from
// allocating multi-GB internal buffers (seen on Blackwell with cublasGemmEx).
int tw_gpu_hgemm_transB(const void* dA, const void* dB, float* dC, int m, int k, int n) {
    if (!tw_lt_handle) tw_cublaslt_init();

    float alpha = 1.0f, beta = 0.0f;

    cublasLtMatmulDesc_t opDesc;
    cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

    // cuBLAS col-major: C[n,m] = op(A)*op(B) where A=dB (transposed), B=dA (not transposed)
    // A (after transpose): [n,k], stored as [k,n] col-major = dB[n,k] row-major, lda=k
    // B (no transpose): [k,m], stored as [k,m] col-major = dA[m,k] row-major, ldb=k
    // C: [n,m] col-major = dC[m,n] row-major, ldc=n
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, k, n, k);  // pre-transpose: [k,n]
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, k, m, k);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, n, m, n);

    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &tw_lt_workspace_size, sizeof(tw_lt_workspace_size));

    cublasLtMatmulHeuristicResult_t heur;
    int nResults = 0;
    cublasLtMatmulAlgoGetHeuristic(tw_lt_handle, opDesc, Adesc, Bdesc, Cdesc, Cdesc,
        pref, 1, &heur, &nResults);

    cublasStatus_t status = CUBLAS_STATUS_NOT_SUPPORTED;
    if (nResults > 0) {
        status = cublasLtMatmul(tw_lt_handle, opDesc,
            &alpha, dB, Adesc, dA, Bdesc,
            &beta, dC, Cdesc, dC, Cdesc,
            &heur.algo, tw_lt_workspace, tw_lt_workspace_size, 0);
    }

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatmulDescDestroy(opDesc);
    return (int)status;
}

// GPU transpose via cuBLAS geam: dst = src^T
// src is [rows, cols] row-major, dst is [cols, rows] row-major
int tw_gpu_transpose(const float* src, float* dst, int rows, int cols) {
    float one = 1.0f, zero = 0.0f;
    // cuBLAS column-major: src[rows,cols] row-major = src[cols,rows] col-major
    // We want dst[cols,rows] row-major = dst[rows,cols] col-major = src^T col-major
    cublasStatus_t s = cublasSgeam(tw_cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        rows, cols,
        &one, src, cols,
        &zero, src, rows,
        dst, rows);
    return (int)s;
}

int tw_gpu_add(const float* dA, const float* dB, float* dC, int n) {
    cudaMemcpy(dC, dA, n * sizeof(float), cudaMemcpyDeviceToDevice);
    float alpha = 1.0f;
    cublasSaxpy(tw_cublas_handle, n, &alpha, dB, 1, dC, 1);
    return 0;
}

int tw_gpu_add_inplace(float* dA, const float* dB, int n) {
    float alpha = 1.0f;
    cublasSaxpy(tw_cublas_handle, n, &alpha, dB, 1, dA, 1);
    return 0;
}

int tw_gpu_scale(const float* dA, float* dB, float alpha, int n) {
    cudaMemcpy(dB, dA, n * sizeof(float), cudaMemcpyDeviceToDevice);
    cublasSscal(tw_cublas_handle, n, &alpha, dB, 1);
    return 0;
}

void tw_gpu_copy_d2d(void* dst, const void* src, size_t bytes) {
    cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
}
*/
import "C"

import (
	"fmt"
	"log"
	"math"
	"sync"
	"unsafe"
)

// cuPtr wraps a CUDA device pointer.
type cuPtr struct {
	ptr unsafe.Pointer
}

func (p *cuPtr) RawPtr() unsafe.Pointer { return p.ptr }

// Implement TensorEngine for CUDA

func (c *CUDA) FromHost(data []float32, shape []int) *Tensor {
	size := 1
	for _, s := range shape {
		size *= s
	}
	ptr := c.poolGet(size) // uses arena if active
	C.tw_gpu_upload(ptr, unsafe.Pointer(&data[0]), C.size_t(size*4))

	return &Tensor{
		Shape:  shape,
		Size:   size,
		device: &cuPtr{ptr: ptr},
		eng:    c,
	}
}

func (c *CUDA) Zeros(shape []int) *Tensor {
	size := 1
	for _, s := range shape {
		size *= s
	}
	ptr := c.poolGet(size)
	C.tw_gpu_zero(ptr, C.size_t(size*4))

	return &Tensor{
		Shape:  shape,
		Size:   size,
		device: &cuPtr{ptr: ptr},
		eng:    c,
	}
}

func (c *CUDA) ToHost(t *Tensor) []float32 {
	cu := t.device.(*cuPtr)
	data := make([]float32, t.Size)
	C.tw_gpu_download(unsafe.Pointer(&data[0]), cu.ptr, C.size_t(t.Size*4))
	return data
}

func (c *CUDA) Release(t *Tensor) {
	if t.device != nil {
		cu := t.device.(*cuPtr)
		c.poolPut(t.Size, cu.ptr)
		t.device = nil
	}
}

func (c *CUDA) MatMulT(a, b *Tensor, m, k, n int) *Tensor {
	size := m * n
	ptr := c.poolGet(size)

	C.tw_gpu_sgemm(
		(*C.float)(a.device.(*cuPtr).ptr),
		(*C.float)(b.device.(*cuPtr).ptr),
		(*C.float)(ptr),
		C.int(m), C.int(k), C.int(n),
	)

	return &Tensor{
		Shape:  []int{m, n},
		Size:   size,
		device: &cuPtr{ptr: ptr},
		eng:    c,
	}
}

// FromHostFP16 uploads FP32 data as FP16 on GPU (for mixed-precision training).
func (c *CUDA) FromHostFP16(data []float32, shape []int) *Tensor {
	size := 1
	for _, s := range shape {
		size *= s
	}
	ptr := C.tw_gpu_upload_fp16((*C.float)(unsafe.Pointer(&data[0])), C.int(size))

	return &Tensor{
		Shape:  shape,
		Size:   size,
		device: &cuPtr{ptr: ptr},
		eng:    c,
	}
}

// MatMulFP16T computes C = A @ B where A,B are FP16, output C is FP32.
// Uses cublasGemmEx with tensor cores. For maximum throughput, use MatMulFP16 (FP16 output).
func (c *CUDA) MatMulFP16T(a, b *Tensor, m, k, n int) *Tensor {
	size := m * n
	ptr := c.poolGet(size)

	C.tw_gpu_hgemm(
		a.device.(*cuPtr).ptr,
		b.device.(*cuPtr).ptr,
		(*C.float)(ptr),
		C.int(m), C.int(k), C.int(n),
	)

	return &Tensor{
		Shape:  []int{m, n},
		Size:   size,
		device: &cuPtr{ptr: ptr},
		eng:    c,
	}
}

// MatMulFP16TransposeBT computes C[m,n] = A[m,k] @ B[n,k]^T.
// A,B are FP16, C is FP32. Uses tensor cores for 2x throughput vs FP32.
func (c *CUDA) MatMulFP16TransposeBT(a, b *Tensor, m, k, n int) *Tensor {
	size := m * n
	ptr := c.poolGet(size)
	ret := C.tw_gpu_hgemm_transB(
		a.DevicePtr(),
		b.DevicePtr(),
		(*C.float)(ptr),
		C.int(m), C.int(k), C.int(n),
	)
	if ret != 0 {
		log.Printf("[CUDA] hgemm_transB FAILED: status=%d m=%d k=%d n=%d", ret, m, k, n)
	}
	return &Tensor{Shape: []int{m, n}, Size: size, device: &cuPtr{ptr: ptr}, eng: c}
}

// MatMulFP16TInto computes C = A @ B into an existing FP32 output tensor (no allocation).
func (c *CUDA) MatMulFP16TInto(a, b, out *Tensor, m, k, n int) {
	C.tw_gpu_hgemm(
		a.device.(*cuPtr).ptr,
		b.device.(*cuPtr).ptr,
		(*C.float)(out.device.(*cuPtr).ptr),
		C.int(m), C.int(k), C.int(n),
	)
}

// MatMulFP16 computes C = A @ B where A,B,C are all FP16.
// Uses cublasLtMatmul with 32 MiB workspace and heuristic algo selection.
// Same API path PyTorch uses — maximum tensor core throughput on Blackwell.
// Output is FP16 (half the bandwidth of FP32 output).
func (c *CUDA) MatMulFP16(a, b *Tensor, m, k, n int) *Tensor {
	// FP16 output: 2 bytes per element
	// Pool is keyed by float-count, so use ceil(fp16_bytes / 4)
	fp16Bytes := m * n * 2
	poolKey := (fp16Bytes + 3) / 4
	ptr := c.poolGet(poolKey)

	ret := C.tw_gpu_hgemm_lt(
		a.device.(*cuPtr).ptr,
		b.device.(*cuPtr).ptr,
		ptr,
		C.int(m), C.int(k), C.int(n),
	)
	if ret != 0 {
		// Fallback to cublasGemmEx path (FP32 output, slightly slower)
		c.poolPut(poolKey, ptr)
		return c.MatMulFP16T(a, b, m, k, n)
	}

	return &Tensor{
		Shape:  []int{m, n},
		Size:   poolKey,
		device: &cuPtr{ptr: ptr},
		eng:    c,
	}
}

var poolAllocBytes int64
var poolAllocCount int64
var poolReuseCount int64
var poolFreeCount int64

// PoolStats returns (total bytes allocated, alloc count, reuse count, free count).
func PoolStats() (int64, int64, int64, int64) {
	return poolAllocBytes, poolAllocCount, poolReuseCount, poolFreeCount
}

// poolGet returns a GPU buffer — from arena if active, else legacy pool.
func (c *CUDA) poolGet(sizeFloats int) unsafe.Pointer {
	if c.Arena != nil {
		ptr := c.Arena.AllocFloats(sizeFloats)
		if ptr != nil {
			poolReuseCount++
			return ptr
		}
		// Arena exhausted — fall through to cudaMalloc
		log.Printf("[arena] fallthrough to cudaMalloc for %d floats", sizeFloats)
	}
	c.poolMu.Lock()
	if free := c.pool[sizeFloats]; len(free) > 0 {
		ptr := free[len(free)-1]
		c.pool[sizeFloats] = free[:len(free)-1]
		c.poolMu.Unlock()
		poolReuseCount++
		return ptr
	}
	c.poolMu.Unlock()
	poolAllocBytes += int64(sizeFloats * 4)
	poolAllocCount++
	return C.tw_gpu_alloc(C.size_t(sizeFloats * 4))
}

// poolPut returns a GPU buffer to the pool for reuse.
// Pool cap per size. Training with N layers needs many same-sized tensors alive
// simultaneously (layer caches). This is not a leak — just high water mark.
// The pool stabilizes after step 1.
func (c *CUDA) poolPut(sizeFloats int, ptr unsafe.Pointer) {
	if c.Arena != nil {
		c.Arena.FreeFloats(ptr, sizeFloats)
		return
	}
	// Legacy pool fallback with cap 4
	c.poolMu.Lock()
	if len(c.pool[sizeFloats]) >= 4 {
		c.poolMu.Unlock()
		C.tw_gpu_free(ptr)
		poolFreeCount++
		return
	}
	c.pool[sizeFloats] = append(c.pool[sizeFloats], ptr)
	c.poolMu.Unlock()
}

func (c *CUDA) MatMulTransposeAT(a, b *Tensor, m, k, n int) *Tensor {
	size := k * n
	ptr := c.poolGet(size)
	C.tw_gpu_sgemm_transA(
		(*C.float)(a.device.(*cuPtr).ptr),
		(*C.float)(b.device.(*cuPtr).ptr),
		(*C.float)(ptr),
		C.int(m), C.int(k), C.int(n),
	)
	return &Tensor{Shape: []int{k, n}, Size: size, device: &cuPtr{ptr: ptr}, eng: c}
}

func (c *CUDA) MatMulTransposeBT(a, b *Tensor, m, k, n int) *Tensor {
	size := m * n
	ptr := c.poolGet(size)
	C.tw_gpu_sgemm_transB(
		(*C.float)(a.device.(*cuPtr).ptr),
		(*C.float)(b.device.(*cuPtr).ptr),
		(*C.float)(ptr),
		C.int(m), C.int(k), C.int(n),
	)
	return &Tensor{Shape: []int{m, n}, Size: size, device: &cuPtr{ptr: ptr}, eng: c}
}

func (c *CUDA) AddInPlace(a, b *Tensor) {
	C.tw_gpu_add_inplace((*C.float)(a.device.(*cuPtr).ptr), (*C.float)(b.device.(*cuPtr).ptr), C.int(a.Size))
}

func (c *CUDA) ReLUBackwardT(dOut, fwdInput *Tensor) *Tensor {
	if KernelsLoaded() {
		out := c.Zeros(dOut.Shape)
		KReLUBackward(out.device.(*cuPtr).ptr, dOut.device.(*cuPtr).ptr, fwdInput.device.(*cuPtr).ptr, dOut.Size)
		return out
	}
	// Fallback: download, mask on CPU, upload
	dData := c.ToHost(dOut)
	fData := c.ToHost(fwdInput)
	for i := range dData {
		if fData[i] <= 0 {
			dData[i] = 0
		}
	}
	return c.FromHost(dData, dOut.Shape)
}

func (c *CUDA) CopyT(src *Tensor) *Tensor {
	size := src.Size
	ptr := c.poolGet(size)
	C.tw_gpu_copy_d2d(ptr, src.device.(*cuPtr).ptr, C.size_t(size*4))
	return &Tensor{Shape: src.Shape, Size: size, device: &cuPtr{ptr: ptr}, eng: c}
}

// CopyInto copies src GPU data into dst. No allocation. dst must be same size or larger.
func (c *CUDA) CopyInto(dst, src *Tensor) {
	C.tw_gpu_copy_d2d(dst.device.(*cuPtr).ptr, src.device.(*cuPtr).ptr, C.size_t(src.Size*4))
}

func (c *CUDA) AddT(a, b *Tensor) *Tensor {
	result := c.Zeros([]int{a.Size})
	cuA := a.device.(*cuPtr)
	cuB := b.device.(*cuPtr)
	cuC := result.device.(*cuPtr)
	C.tw_gpu_add((*C.float)(cuA.ptr), (*C.float)(cuB.ptr), (*C.float)(cuC.ptr), C.int(a.Size))
	return result
}

func (c *CUDA) ScaleT(a *Tensor, s float32) *Tensor {
	result := c.Zeros([]int{a.Size})
	cuA := a.device.(*cuPtr)
	cuC := result.device.(*cuPtr)
	C.tw_gpu_scale((*C.float)(cuA.ptr), (*C.float)(cuC.ptr), C.float(s), C.int(a.Size))
	return result
}

func (c *CUDA) ReLUT(a *Tensor) *Tensor {
	if KernelsLoaded() {
		out := c.Zeros(a.Shape)
		KReLUOut(a.device.(*cuPtr).ptr, out.device.(*cuPtr).ptr, a.Size)
		return out
	}
	// Fallback: download, apply on CPU, upload
	data := c.ToHost(a)
	for i := range data {
		if data[i] < 0 {
			data[i] = 0
		}
	}
	return c.FromHost(data, a.Shape)
}

func (c *CUDA) TransposeT(a *Tensor, rows, cols int) *Tensor {
	size := rows * cols
	ptr := c.poolGet(size)
	C.tw_gpu_transpose(
		(*C.float)(a.device.(*cuPtr).ptr),
		(*C.float)(ptr),
		C.int(rows), C.int(cols),
	)
	return &Tensor{Shape: []int{cols, rows}, Size: size, device: &cuPtr{ptr: ptr}, eng: c}
}

// === INT8 Quantized Weight Support ===
// Store weights as INT8 in VRAM (1 byte/element) with per-row FP32 scales.
// Dequantize to FP16 or FP32 on-the-fly for cuBLAS matmuls.
// Memory: 14B params → ~14GB INT8 vs 56GB FP32.

// QuantizedTensor holds INT8-quantized weight data with per-row absmax scales.
type QuantizedTensor struct {
	DataInt8 []int8
	Scales   []float32
	Shape    []int
	Rows     int
	Cols     int
}

// Int8Tensor holds quantized weights on GPU with their scale factors.
type Int8Tensor struct {
	DataPtr  unsafe.Pointer
	ScalePtr unsafe.Pointer
	Rows     int
	Cols     int
	eng      *CUDA
}

func (q *Int8Tensor) VRAMBytes() int {
	return q.Rows*q.Cols + q.Rows*4
}

func (c *CUDA) FromHostInt8(qt *QuantizedTensor) *Int8Tensor {
	nBytes := C.size_t(qt.Rows * qt.Cols)
	dataPtr := C.tw_gpu_alloc(nBytes)
	C.tw_gpu_upload(dataPtr, unsafe.Pointer(&qt.DataInt8[0]), nBytes)

	scaleBytes := C.size_t(qt.Rows * 4)
	scalePtr := C.tw_gpu_alloc(scaleBytes)
	C.tw_gpu_upload(scalePtr, unsafe.Pointer(&qt.Scales[0]), scaleBytes)

	return &Int8Tensor{
		DataPtr:  dataPtr,
		ScalePtr: scalePtr,
		Rows:     qt.Rows,
		Cols:     qt.Cols,
		eng:      c,
	}
}

func (c *CUDA) ReleaseInt8(q *Int8Tensor) {
	if q.DataPtr != nil { C.tw_gpu_free(q.DataPtr); q.DataPtr = nil }
	if q.ScalePtr != nil { C.tw_gpu_free(q.ScalePtr); q.ScalePtr = nil }
}

func (c *CUDA) DequantToFP16(q *Int8Tensor, fp16Buf unsafe.Pointer) {
	KDequantInt8ToFP16(q.DataPtr, q.ScalePtr, fp16Buf, q.Rows, q.Cols)
}

func (c *CUDA) DequantToFP32(q *Int8Tensor, fp32Buf unsafe.Pointer) {
	KDequantInt8ToFP32(q.DataPtr, q.ScalePtr, fp32Buf, q.Rows, q.Cols)
}

func (c *CUDA) AllocFP16Buffer(nElements int) unsafe.Pointer {
	return C.tw_gpu_alloc(C.size_t(nElements * 2))
}

func (c *CUDA) AllocFP16Tensor(nElements int, shape []int) *Tensor {
	ptr := C.tw_gpu_alloc(C.size_t(nElements * 2))
	return &Tensor{Shape: shape, Size: nElements, device: &cuPtr{ptr: ptr}}
}

func (c *CUDA) FreeFP16Tensor(t *Tensor) {
	if t != nil && t.device != nil {
		C.tw_gpu_free(t.device.(*cuPtr).ptr)
		t.device = nil
	}
}


// === TrainEngine implementation ===
// Host-slice operations backed by GPU compute.
// Tensor cache: maps host slice data pointer → GPU tensor.
// Weight matrices are uploaded once per step and reused across matmul calls.
// Call ClearTrainCache() between steps.

// trainTensorCache maps host data pointer to cached GPU tensor.
// This eliminates re-uploading weight matrices that don't change within a step.
var trainCache map[uintptr]*Tensor
var trainCacheMu sync.Mutex

func (c *CUDA) cachedUpload(data []float32, shape []int) *Tensor {
	ptr := uintptr(unsafe.Pointer(&data[0]))
	trainCacheMu.Lock()
	if trainCache == nil { trainCache = make(map[uintptr]*Tensor) }
	if t, ok := trainCache[ptr]; ok {
		trainCacheMu.Unlock()
		return t
	}
	trainCacheMu.Unlock()
	t := c.FromHost(data, shape)
	trainCacheMu.Lock()
	trainCache[ptr] = t
	trainCacheMu.Unlock()
	return t
}

// ClearTrainCache releases all cached GPU tensors. Call between training steps.
func (c *CUDA) ClearTrainCache() {
	trainCacheMu.Lock()
	for _, t := range trainCache {
		c.Release(t)
	}
	trainCache = make(map[uintptr]*Tensor)
	trainCacheMu.Unlock()
}

func (c *CUDA) MatMulTransBInto(out, A, B []float32, m, k, n int) {
	tA := c.FromHost(A, []int{m, k})
	tB := c.FromHost(B, []int{n, k})
	tC := c.MatMulTransposeBT(tA, tB, m, k, n)
	copy(out, c.ToHost(tC))
	c.Release(tA); c.Release(tB); c.Release(tC)
}

func (c *CUDA) MatMulInto(out, A, B []float32, m, k, n int) {
	tA := c.FromHost(A, []int{m, k})
	tB := c.FromHost(B, []int{k, n})
	tC := c.MatMulT(tA, tB, m, k, n)
	copy(out, c.ToHost(tC))
	c.Release(tA); c.Release(tB); c.Release(tC)
}

func (c *CUDA) MatMulAddInto(G, A, B []float32, m, k, n int) {
	tA := c.FromHost(A, []int{m, k})
	tB := c.FromHost(B, []int{m, n})
	tC := c.MatMulTransposeAT(tA, tB, m, k, n)
	result := c.ToHost(tC)
	for i := range result { G[i] += result[i] }
	c.Release(tA); c.Release(tB); c.Release(tC)
}

func (c *CUDA) MatMulTransA(A, B []float32, m, k, n int) []float32 {
	tA := c.FromHost(A, []int{m, k})
	tB := c.FromHost(B, []int{m, n})
	tC := c.MatMulTransposeAT(tA, tB, m, k, n)
	result := c.ToHost(tC)
	c.Release(tA); c.Release(tB); c.Release(tC)
	return result
}

func (c *CUDA) GER(G, x, y []float32, m, n int, alpha float32) {
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ { G[i*n+j] += alpha * x[i] * y[j] }
	}
}

func (c *CUDA) Nrm2(x []float32) float32 {
	var ss float32
	for _, v := range x { ss += v * v }
	return float32(math.Sqrt(float64(ss)))
}

func (c *CUDA) Scal(x []float32, alpha float32) {
	for i := range x { x[i] *= alpha }
}

func (c *CUDA) AdamWStep(D, G, M, V []float32, n int,
	lr, beta1, beta2, bc1, bc2, eps, wd float32) {
	// Use GPU AdamW kernel via TensorEngine
	if KernelsLoaded() {
		tD := c.FromHost(D, []int{n})
		tG := c.FromHost(G, []int{n})
		tM := c.FromHost(M, []int{n})
		tV := c.FromHost(V, []int{n})
		KAdamW(tD.DevicePtr(), tG.DevicePtr(), tM.DevicePtr(), tV.DevicePtr(),
			lr, wd, int(bc1*1000), n) // step approximation
		copy(D, c.ToHost(tD))
		copy(M, c.ToHost(tM))
		copy(V, c.ToHost(tV))
		c.Release(tD); c.Release(tG); c.Release(tM); c.Release(tV)
		return
	}
	// CPU fallback
	ob1 := 1 - beta1; ob2 := 1 - beta2
	for i := 0; i < n; i++ {
		g := G[i]
		M[i] = beta1*M[i] + ob1*g
		V[i] = beta2*V[i] + ob2*g*g
		mh := M[i] / bc1; vh := V[i] / bc2
		D[i] -= lr * (mh/(float32(math.Sqrt(float64(vh)))+eps) + wd*D[i])
	}
}

// === L3 Bridge — pinned host memory for zero-copy GPU access ===

// AllocL3Bridge allocates pinned host memory via cudaHostAlloc.
func (c *CUDA) AllocL3Bridge(bytes int) *L3Bridge {
	ptr := C.tw_gpu_alloc_pinned(C.size_t(bytes))
	if ptr == nil {
		return nil
	}
	C.memset(ptr, 0, C.size_t(bytes))
	log.Printf("[cuda] L3 bridge: %d MB pinned host memory", bytes/(1024*1024))
	return &L3Bridge{Ptr: unsafe.Pointer(ptr), Size: bytes}
}

// RegisterL3Bridge registers external host memory with CUDA for DMA.
func (c *CUDA) RegisterL3Bridge(bridge *L3Bridge) error {
	if bridge == nil {
		return nil
	}
	ret := C.tw_register_host_memory(bridge.Ptr, C.size_t(bridge.Size))
	if ret != 0 {
		return fmt.Errorf("cudaHostRegister failed: %d", ret)
	}
	log.Printf("[cuda] L3 bridge registered — %d MB", bridge.Size/(1024*1024))
	return nil
}

// MatMulL3 computes C = W @ X^T where W is on GPU, X and C are pinned L3 memory.
// W[rows,cols] on device, X[n,cols] pinned, C[n,rows] pinned.
// cuBLAS reads X from L3 cache, writes C to L3 cache. Zero upload for activations.
func (c *CUDA) MatMulL3(wT *Tensor, pinnedX unsafe.Pointer, pinnedOut unsafe.Pointer, m, k, n int) {
	wPtr := wT.DevicePtr()
	C.tw_gpu_sgemm_transB_l3((*C.float)(pinnedX), (*C.float)(wPtr), (*C.float)(pinnedOut), C.int(m), C.int(k), C.int(n))
}
