//go:build linux && cgo

package mongoose

/*
// CUDA library paths — override with CGO_LDFLAGS if cuBLAS is not in standard paths.
// Common locations: /usr/lib, /usr/local/cuda/lib64, pip nvidia-cublas-cu12
#cgo LDFLAGS: -lcublas -lcudart
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

cublasHandle_t tw_cublas_handle;
int tw_cuda_initialized = 0;

static void* tw_cublas_workspace = NULL;
static size_t tw_cublas_workspace_size = 32 * 1024 * 1024; // 32 MiB

static int tw_cuda_device_idx = 0;

int tw_cuda_init() { return tw_cuda_init_device(0); }

int tw_cuda_device_count() {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

int tw_cuda_init_device(int deviceIdx) {
    if (tw_cuda_initialized) return 0;
    tw_cuda_device_idx = deviceIdx;
    cudaError_t cerr = cudaSetDevice(deviceIdx);
    if (cerr != cudaSuccess) return -1;
    cublasStatus_t serr = cublasCreate(&tw_cublas_handle);
    if (serr != CUBLAS_STATUS_SUCCESS) return -2;
    cublasSetMathMode(tw_cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);
    // Cap cuBLAS workspace to 32MB — Blackwell cublasGemmEx allocates multi-GB otherwise
    cudaMalloc(&tw_cublas_workspace, tw_cublas_workspace_size);
    cublasSetWorkspace(tw_cublas_handle, tw_cublas_workspace, tw_cublas_workspace_size);
    tw_cuda_initialized = 1;

    // Enable peer access for NVLink multi-GPU
    int nDevices = 0;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        if (i == deviceIdx) continue;
        int canAccess = 0;
        cudaDeviceCanAccessPeer(&canAccess, deviceIdx, i);
        if (canAccess) {
            cudaDeviceEnablePeerAccess(i, 0);
        }
    }

    return 0;
}

void tw_cuda_cleanup() {
    if (tw_cuda_initialized) {
        cublasDestroy(tw_cublas_handle);
        tw_cuda_initialized = 0;
    }
}

int tw_sgemm(const float* A, const float* B, float* C, int m, int k, int n) {
    void *dA, *dB, *dC;
    cudaMalloc(&dA, m * k * sizeof(float));
    cudaMalloc(&dB, k * n * sizeof(float));
    cudaMalloc(&dC, m * n * sizeof(float));

    cudaMemcpy(dA, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, k * n * sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t status = cublasGemmEx(tw_cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        (const float*)dB, CUDA_R_32F, n,
        (const float*)dA, CUDA_R_32F, k,
        &beta,
        (float*)dC, CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    cudaMemcpy(C, dC, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return (int)status;
}

// tw_sgemm_gpu: same but with persistent GPU buffers (for training hot path)
// Caller manages GPU memory via tw_alloc/tw_free/tw_upload/tw_download
void* tw_alloc(size_t bytes) {
    void* ptr;
    cudaMalloc(&ptr, bytes);
    return ptr;
}

void tw_free(void* ptr) {
    cudaFree(ptr);
}

void tw_upload(void* dst, const void* src, size_t bytes) {
    cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
}

void tw_download(void* dst, const void* src, size_t bytes) {
    cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
}

int tw_sgemm_dev(const float* dA, const float* dB, float* dC, int m, int k, int n) {
    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t status = cublasGemmEx(tw_cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        dB, CUDA_R_32F, n,
        dA, CUDA_R_32F, k,
        &beta,
        dC, CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return (int)status;
}

void tw_cuda_synchronize() {
    cudaDeviceSynchronize();
}

size_t tw_cuda_vram_total() {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return total;
}

size_t tw_cuda_vram_free() {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return free;
}

// CUDA event timing — GPU-side measurement, same as torch.cuda.Event
typedef struct {
    cudaEvent_t start;
    cudaEvent_t stop;
} tw_timer_t;

tw_timer_t tw_timer_create() {
    tw_timer_t t;
    cudaEventCreate(&t.start);
    cudaEventCreate(&t.stop);
    return t;
}

void tw_timer_start(tw_timer_t* t) {
    cudaEventRecord(t->start, 0);
}

float tw_timer_stop(tw_timer_t* t) {
    cudaEventRecord(t->stop, 0);
    cudaEventSynchronize(t->stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, t->start, t->stop);
    return ms;
}

void tw_timer_destroy(tw_timer_t* t) {
    cudaEventDestroy(t->start);
    cudaEventDestroy(t->stop);
}

const char* tw_cuda_device_name() {
    static char name[256];
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    snprintf(name, sizeof(name), "%s", prop.name);
    return name;
}
*/
import "C"

import (
	"fmt"
	"log"
	"math"
	"runtime"
	"sync"
	"time"
	"unsafe"
)

// CUDA implements Engine using NVIDIA cuBLAS.
// Uses tensor cores when available. Fastest path for NVIDIA GPUs.
type CUDA struct {
	deviceName string
	pool       map[int][]unsafe.Pointer // legacy buffer pool (used when arena is nil)
	poolMu     sync.Mutex
	Arena      *GPUArena                // arena allocator — replaces pool when active
}

// NewCUDA initializes the CUDA backend on device 0.
func NewCUDA() *CUDA { return NewCUDADevice(0) }

// NewCUDADevice initializes CUDA on a specific GPU device index.
func NewCUDADevice(deviceIdx int) *CUDA {
	ret := C.tw_cuda_init_device(C.int(deviceIdx))
	if ret != 0 {
		log.Printf("WARN compute => CUDA init failed (device %d, code %d)", deviceIdx, ret)
		return nil
	}

	name := C.GoString(C.tw_cuda_device_name())
	log.Printf("[compute] CUDA initialized: %s (device %d)", name, deviceIdx)

	return &CUDA{deviceName: name, pool: make(map[int][]unsafe.Pointer)}
}

func (c *CUDA) Name() string {
	return fmt.Sprintf("cuda/%s", c.deviceName)
}

// MatMul computes C = A @ B via cuBLAS sgemm.
func (c *CUDA) MatMul(a, b []float32, m, k, n int) []float32 {
	out := make([]float32, m*n)

	ret := C.tw_sgemm(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&b[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(m), C.int(k), C.int(n),
	)
	if ret != 0 {
		log.Printf("WARN compute => cuBLAS sgemm failed: %d", ret)
	}

	return out
}

// RMSNorm on CPU (small vector).
func (c *CUDA) RMSNorm(x, weight []float32, eps float32) {
	n := len(x)
	var ss float32
	for i := 0; i < n; i++ {
		ss += x[i] * x[i]
	}
	ss = ss/float32(n) + eps
	ss = float32(1.0 / math.Sqrt(float64(ss)))
	for i := 0; i < n; i++ {
		x[i] = x[i] * ss * weight[i]
	}
}

func (c *CUDA) SoftMax(x []float32, n int) {
	maxVal := x[0]
	for i := 1; i < n; i++ {
		if x[i] > maxVal {
			maxVal = x[i]
		}
	}
	var sum float32
	for i := 0; i < n; i++ {
		x[i] = float32(math.Exp(float64(x[i] - maxVal)))
		sum += x[i]
	}
	inv := 1.0 / sum
	for i := 0; i < n; i++ {
		x[i] *= inv
	}
}

func (c *CUDA) ReLU(x []float32) {
	for i := range x {
		if x[i] < 0 {
			x[i] = 0
		}
	}
}

func (c *CUDA) VRAM() uint64 { return uint64(C.tw_cuda_vram_total()) }

// Sync forces all pending CUDA work on all streams to complete.
func (c *CUDA) Sync() { C.tw_cuda_synchronize() }

// CUDATimer provides GPU-side timing via CUDA events.
// Same mechanism as torch.cuda.Event — measures kernel execution, not CPU overhead.
type CUDATimer struct {
	t C.tw_timer_t
}

// NewCUDATimer creates a GPU-side timer.
func NewCUDATimer() *CUDATimer { t := C.tw_timer_create(); return &CUDATimer{t: t} }

// Start records the start event on the default stream.
func (t *CUDATimer) Start() { C.tw_timer_start(&t.t) }

// StopMs records the stop event, synchronizes, and returns elapsed milliseconds.
func (t *CUDATimer) StopMs() float32 { return float32(C.tw_timer_stop(&t.t)) }

// Destroy releases the CUDA events.
func (t *CUDATimer) Destroy() { C.tw_timer_destroy(&t.t) }

func (c *CUDA) Benchmark() float64 {
	const dim = 512
	a := make([]float32, dim*dim)
	b := make([]float32, dim*dim)
	for i := range a {
		a[i] = 0.001 * float32(i%1000)
		b[i] = 0.001 * float32(i%997)
	}

	runtime.GC()
	// Warmup
	c.MatMul(a, b, dim, dim, dim)

	start := time.Now()
	iterations := 100
	for i := 0; i < iterations; i++ {
		c.MatMul(a, b, dim, dim, dim)
	}
	elapsed := time.Since(start)

	flops := float64(2*dim*dim*dim*iterations) / elapsed.Seconds()
	return flops / 1e9
}
