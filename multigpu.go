//go:build linux && cgo

package mongoose

/*
#cgo LDFLAGS: -lcublas -lcublasLt -lcudart
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Per-device state
typedef struct {
    int device_id;
    cublasHandle_t handle;
    cublasLtHandle_t lt_handle;
    void* lt_workspace;
    size_t lt_workspace_size;
    cudaStream_t stream;  // dedicated stream per device for parallel dispatch
    char name[256];
    size_t total_mem;
    size_t free_mem;
    int sm_count;
    int compute_major;
    int compute_minor;
    int can_peer[16]; // can_peer[j] = 1 if this device can P2P with device j
} tw_device_t;

static tw_device_t tw_devices[16];
static int tw_device_count = 0;

int tw_multi_init() {
    int count;
    cudaGetDeviceCount(&count);
    if (count <= 0) return -1;
    if (count > 16) count = 16;
    tw_device_count = count;

    for (int i = 0; i < count; i++) {
        tw_devices[i].device_id = i;
        cudaSetDevice(i);

        struct cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        snprintf(tw_devices[i].name, sizeof(tw_devices[i].name), "%s", prop.name);
        tw_devices[i].total_mem = prop.totalGlobalMem;
        tw_devices[i].sm_count = prop.multiProcessorCount;
        tw_devices[i].compute_major = prop.major;
        tw_devices[i].compute_minor = prop.minor;

        size_t free, total;
        cudaMemGetInfo(&free, &total);
        tw_devices[i].free_mem = free;

        cudaStreamCreate(&tw_devices[i].stream);
        cublasCreate(&tw_devices[i].handle);
        cublasSetMathMode(tw_devices[i].handle, CUBLAS_TF32_TENSOR_OP_MATH);
        cublasSetStream(tw_devices[i].handle, tw_devices[i].stream);

        cublasLtCreate(&tw_devices[i].lt_handle);
        tw_devices[i].lt_workspace_size = 32 * 1024 * 1024; // 32 MiB
        cudaMalloc(&tw_devices[i].lt_workspace, tw_devices[i].lt_workspace_size);

        // Check P2P access
        for (int j = 0; j < count; j++) {
            if (i == j) {
                tw_devices[i].can_peer[j] = 1;
            } else {
                int access = 0;
                cudaDeviceCanAccessPeer(&access, i, j);
                tw_devices[i].can_peer[j] = access;
                if (access) {
                    cudaDeviceEnablePeerAccess(j, 0);
                }
            }
        }
    }

    // Reset to device 0
    cudaSetDevice(0);
    return count;
}

int tw_multi_device_count() { return tw_device_count; }

const char* tw_multi_device_name(int dev) {
    if (dev < 0 || dev >= tw_device_count) return "invalid";
    return tw_devices[dev].name;
}

size_t tw_multi_device_mem(int dev) {
    if (dev < 0 || dev >= tw_device_count) return 0;
    return tw_devices[dev].total_mem;
}

size_t tw_multi_device_free(int dev) {
    if (dev < 0 || dev >= tw_device_count) return 0;
    // Refresh
    cudaSetDevice(dev);
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    tw_devices[dev].free_mem = free;
    return free;
}

int tw_multi_can_peer(int src, int dst) {
    if (src < 0 || src >= tw_device_count || dst < 0 || dst >= tw_device_count) return 0;
    return tw_devices[src].can_peer[dst];
}

// Allocate memory on a specific device
void* tw_multi_alloc(int dev, size_t bytes) {
    cudaSetDevice(dev);
    void* ptr;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) return NULL;
    return ptr;
}

void tw_multi_free(int dev, void* ptr) {
    cudaSetDevice(dev);
    cudaFree(ptr);
}

void tw_multi_upload(int dev, void* dst, const void* src, size_t bytes) {
    cudaSetDevice(dev);
    cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
}

void tw_multi_download(int dev, void* dst, const void* src, size_t bytes) {
    cudaSetDevice(dev);
    cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
}

// P2P copy between devices — uses NVLink if available, PCIe otherwise
void tw_multi_peer_copy(int src_dev, const void* src, int dst_dev, void* dst, size_t bytes) {
    cudaMemcpyPeer(dst, dst_dev, src, src_dev, bytes);
}

// MatMul on a specific device
int tw_multi_sgemm(int dev, const float* dA, const float* dB, float* dC, int m, int k, int n) {
    cudaSetDevice(dev);
    float alpha = 1.0f, beta = 0.0f;
    return (int)cublasGemmEx(tw_devices[dev].handle,
        CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
        &alpha, dB, CUDA_R_32F, n, dA, CUDA_R_32F, k,
        &beta, dC, CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// FP16 MatMul on a specific device via cublasLt
int tw_multi_hgemm_lt(int dev, const void* dA, const void* dB, void* dC, int m, int k, int n) {
    cudaSetDevice(dev);

    float alpha = 1.0f, beta = 0.0f;

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
        &tw_devices[dev].lt_workspace_size, sizeof(size_t));

    cublasLtMatmulHeuristicResult_t heur;
    int nResults = 0;
    cublasLtMatmulAlgoGetHeuristic(tw_devices[dev].lt_handle, opDesc, Adesc, Bdesc, Cdesc, Cdesc,
        pref, 1, &heur, &nResults);

    cublasStatus_t status = CUBLAS_STATUS_NOT_SUPPORTED;
    if (nResults > 0) {
        status = cublasLtMatmul(tw_devices[dev].lt_handle, opDesc,
            &alpha, dB, Adesc, dA, Bdesc,
            &beta, dC, Cdesc, dC, Cdesc,
            &heur.algo,
            tw_devices[dev].lt_workspace, tw_devices[dev].lt_workspace_size, 0);
    }

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatmulDescDestroy(opDesc);

    return (int)status;
}

// FP16 MatMul with B transposed on a specific device via cublasGemmEx.
// Single call, zero descriptor overhead. C[m,n] = A[m,k] @ B[n,k]^T, all FP16.
int tw_multi_hgemm_transB(int dev, const void* dA, const void* dB, void* dC, int m, int k, int n) {
    float alpha = 1.0f, beta = 0.0f;
    return (int)cublasGemmEx(tw_devices[dev].handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        dB, CUDA_R_16F, k,
        dA, CUDA_R_16F, k,
        &beta,
        dC, CUDA_R_16F, n,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// FP16 MatMul on device: C = A @ B, FP16 in, FP32 out. For dW gradient accumulation.
int tw_multi_hgemm_fp32out(int dev, const void* dA, const void* dB, float* dC, int m, int k, int n) {
    float alpha = 1.0f, beta = 0.0f;
    return (int)cublasGemmEx(tw_devices[dev].handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        dB, CUDA_R_16F, n,
        dA, CUDA_R_16F, k,
        &beta,
        dC, CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// FP16 TransA on device: C[k,n] = A[m,k]^T @ B[m,n], FP16 in, FP32 out. For dW grads.
int tw_multi_hgemm_transA_fp32out(int dev, const void* dA, const void* dB, float* dC, int m, int k, int n) {
    float alpha = 1.0f, beta = 0.0f;
    return (int)cublasGemmEx(tw_devices[dev].handle,
        CUBLAS_OP_N, CUBLAS_OP_T, n, k, m,
        &alpha,
        dB, CUDA_R_16F, n,
        dA, CUDA_R_16F, k,
        &beta,
        dC, CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void tw_multi_sync(int dev) {
    cudaSetDevice(dev);
    cudaDeviceSynchronize();
}

void tw_multi_sync_all() {
    for (int i = 0; i < tw_device_count; i++) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }
}

// Fire matmuls on multiple devices from a single thread — zero goroutine overhead.
// Each device has its own cuBLAS handle on its own CUDA stream.
// cudaSetDevice + cublasGemmEx is non-blocking — the kernel is queued and returns
// immediately. We fire all devices, then sync all at the end.
typedef struct {
    int dev;
    const float* dA;
    const float* dB;
    float* dC;
    int m, k, n;
} multi_matmul_op_t;

// Batch N iterations of parallel matmul across devices.
// NO cudaSetDevice in the hot loop — each cuBLAS handle is bound to its
// own device + stream at init time. Launches go directly to the right GPU.
// Sync via per-device streams at the end — not cudaDeviceSynchronize.
void tw_multi_matmul_batch(multi_matmul_op_t* ops, int num_ops, int iters) {
    float alpha = 1.0f, beta = 0.0f;

    for (int it = 0; it < iters; it++) {
        for (int i = 0; i < num_ops; i++) {
            // No cudaSetDevice — handle is already on the right device+stream
            cublasGemmEx(tw_devices[ops[i].dev].handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                ops[i].n, ops[i].m, ops[i].k,
                &alpha,
                ops[i].dB, CUDA_R_32F, ops[i].n,
                ops[i].dA, CUDA_R_32F, ops[i].k,
                &beta,
                ops[i].dC, CUDA_R_32F, ops[i].n,
                CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
    }
    // Sync per-device streams (lighter than cudaDeviceSynchronize)
    for (int i = 0; i < num_ops; i++) {
        cudaStreamSynchronize(tw_devices[ops[i].dev].stream);
    }
}

// Single-iteration version for API compatibility
void tw_multi_matmul_parallel(multi_matmul_op_t* ops, int count) {
    tw_multi_matmul_batch(ops, count, 1);
}

// FP16 parallel op: all-FP16 via per-device cublasLt handles.
typedef struct {
    int dev;
    const void* dA;
    const void* dB;
    void* dC;
    int m, k, n;
    int transB; // 1 = B^T, 0 = no transpose
} multi_hgemm_op_t;

// FP16 batch dispatch across devices — same zero-overhead pattern as FP32.
// Each op uses the device's cublasLt handle + stream. No cudaSetDevice in hot loop.
void tw_multi_hgemm_batch(multi_hgemm_op_t* ops, int num_ops) {
    float alpha = 1.0f, beta = 0.0f;

    for (int i = 0; i < num_ops; i++) {
        int d = ops[i].dev;
        int m = ops[i].m, k = ops[i].k, n = ops[i].n;

        cublasLtMatmulDesc_t opDesc;
        cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

        cublasOperation_t opN = CUBLAS_OP_N;
        cublasOperation_t opT = CUBLAS_OP_T;
        if (ops[i].transB) {
            cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
            cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        } else {
            cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
            cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        }

        cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
        if (ops[i].transB) {
            cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, k, n, k);
            cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, k, m, k);
        } else {
            cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, n, k, n);
            cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, k, m, k);
        }
        cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, n, m, n);

        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &tw_devices[d].lt_workspace_size, sizeof(size_t));

        cublasLtMatmulHeuristicResult_t heur;
        int nResults = 0;
        cublasLtMatmulAlgoGetHeuristic(tw_devices[d].lt_handle, opDesc,
            Adesc, Bdesc, Cdesc, Cdesc, pref, 1, &heur, &nResults);

        if (nResults > 0) {
            cublasLtMatmul(tw_devices[d].lt_handle, opDesc,
                &alpha, ops[i].dB, Adesc, ops[i].dA, Bdesc,
                &beta, ops[i].dC, Cdesc, ops[i].dC, Cdesc,
                &heur.algo,
                tw_devices[d].lt_workspace, tw_devices[d].lt_workspace_size,
                tw_devices[d].stream);
        }

        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(Adesc);
        cublasLtMatrixLayoutDestroy(Bdesc);
        cublasLtMatrixLayoutDestroy(Cdesc);
        cublasLtMatmulDescDestroy(opDesc);
    }
    for (int i = 0; i < num_ops; i++) {
        cudaStreamSynchronize(tw_devices[ops[i].dev].stream);
    }
}

// Getters for device properties (can't access static arrays from Go)
int tw_multi_sm_count(int dev) { return (dev >= 0 && dev < tw_device_count) ? tw_devices[dev].sm_count : 0; }
int tw_multi_compute_major(int dev) { return (dev >= 0 && dev < tw_device_count) ? tw_devices[dev].compute_major : 0; }
int tw_multi_compute_minor(int dev) { return (dev >= 0 && dev < tw_device_count) ? tw_devices[dev].compute_minor : 0; }

void tw_set_device(int dev) { cudaSetDevice(dev); }

// Allocate FP16 buffer on a specific device. Returns device pointer.
void* tw_multi_alloc_fp16(int dev, int nElements) {
    cudaSetDevice(dev);
    void* ptr;
    cudaError_t err = cudaMalloc(&ptr, (size_t)nElements * 2);
    if (err != cudaSuccess) return NULL;
    return ptr;
}

// Allocate raw bytes on a specific device.
void* tw_multi_alloc_bytes(int dev, size_t bytes) {
    cudaSetDevice(dev);
    void* ptr;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) return NULL;
    return ptr;
}

// Allocate + zero FP32 buffer on a specific device. Returns nFloats * 4 bytes.
void* tw_multi_zeros_fp32(int dev, int nFloats) {
    cudaSetDevice(dev);
    void* ptr;
    size_t bytes = (size_t)nFloats * 4;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) return NULL;
    cudaMemset(ptr, 0, bytes);
    return ptr;
}

// Zero a buffer on a specific device.
void tw_multi_zero(int dev, void* ptr, size_t bytes) {
    cudaSetDevice(dev);
    cudaMemset(ptr, 0, bytes);
}

// Upload host data to a specific device — for initializing FP32 weights on remote GPUs.
void tw_multi_upload_fp32(int dev, void* dst, const float* src, int nFloats) {
    cudaSetDevice(dev);
    cudaMemcpy(dst, src, (size_t)nFloats * 4, cudaMemcpyHostToDevice);
}

// Download device data to host.
void tw_multi_download_fp32(int dev, float* dst, const void* src, int nFloats) {
    cudaSetDevice(dev);
    cudaMemcpy(dst, src, (size_t)nFloats * 4, cudaMemcpyDeviceToHost);
}

// P2P copy using async on source device's stream for overlap
void tw_multi_peer_copy_async(int srcDev, const void* src, int dstDev, void* dst, size_t bytes) {
    cudaMemcpyPeerAsync(dst, dstDev, src, srcDev, bytes, tw_devices[srcDev].stream);
}

// Interleave scatter: extract every Nth row starting at offset into a packed buffer.
// src[seqLen, cols] → dst[seqLen/stride, cols], taking rows where (row % stride == offset).
// Both src and dst on the CURRENT device. elemSize = bytes per element (2 for FP16, 4 for FP32).
void tw_interleave_extract(const void* src, void* dst, int seqLen, int cols, int stride, int offset, int elemSize) {
    int outRow = 0;
    size_t rowBytes = (size_t)cols * elemSize;
    for (int i = offset; i < seqLen; i += stride) {
        cudaMemcpy((char*)dst + (size_t)outRow * rowBytes, (const char*)src + (size_t)i * rowBytes, rowBytes, cudaMemcpyDeviceToDevice);
        outRow++;
    }
}

// Interleave gather: scatter packed rows back into strided positions.
// src[seqLen/stride, cols] → dst[seqLen, cols], writing to rows where (row % stride == offset).
void tw_interleave_insert(const void* src, void* dst, int seqLen, int cols, int stride, int offset, int elemSize) {
    int inRow = 0;
    size_t rowBytes = (size_t)cols * elemSize;
    for (int i = offset; i < seqLen; i += stride) {
        cudaMemcpy((char*)dst + (size_t)i * rowBytes, (const char*)src + (size_t)inRow * rowBytes, rowBytes, cudaMemcpyDeviceToDevice);
        inRow++;
    }
}

// Sparse P2P: copy only specified rows from src to dst.
// rows[] = row indices, nRows = count, cols = elements per row.
// Copies nRows * cols * elemSize bytes total, scattered by row index.
void tw_multi_sparse_p2p(int srcDev, const void* src, int dstDev, void* dst,
                          const int* rows, int nRows, int cols, int elemSize) {
    for (int i = 0; i < nRows; i++) {
        int r = rows[i];
        if (r < 0) continue;
        size_t offset = (size_t)r * cols * elemSize;
        cudaMemcpyPeer((char*)dst + offset, dstDev, (const char*)src + offset, srcDev, (size_t)cols * elemSize);
    }
}
*/
import "C"

import (
	"fmt"
	"log"
	"sync"
	"unsafe"
)

// MultiCUDA manages multiple NVIDIA GPUs for parallel computation.
// Each device has its own cuBLAS handle, cublasLt handle, workspace, and buffer pool.
type MultiCUDA struct {
	DeviceCount int
	devices     []DeviceInfo
	pools       []map[int][]unsafe.Pointer
	poolMus     []sync.Mutex
}

// DeviceInfo describes a single GPU in a multi-GPU system.
type DeviceInfo struct {
	ID           int
	Name         string
	TotalMem     uint64
	FreeMem      uint64
	SMCount      int
	ComputeMajor int
	ComputeMinor int
	CanPeer      []bool // CanPeer[j] = can P2P with device j
}

// NewMultiCUDA initializes all available CUDA devices.
// Returns nil if no GPUs are available.
func NewMultiCUDA() *MultiCUDA {
	count := int(C.tw_multi_init())
	if count <= 0 {
		return nil
	}

	mc := &MultiCUDA{
		DeviceCount: count,
		devices:     make([]DeviceInfo, count),
		pools:       make([]map[int][]unsafe.Pointer, count),
		poolMus:     make([]sync.Mutex, count),
	}

	for i := 0; i < count; i++ {
		mc.pools[i] = make(map[int][]unsafe.Pointer)
		mc.devices[i] = DeviceInfo{
			ID:           i,
			Name:         C.GoString(C.tw_multi_device_name(C.int(i))),
			TotalMem:     uint64(C.tw_multi_device_mem(C.int(i))),
			SMCount:      int(C.tw_multi_sm_count(C.int(i))),
			ComputeMajor: int(C.tw_multi_compute_major(C.int(i))),
			ComputeMinor: int(C.tw_multi_compute_minor(C.int(i))),
			CanPeer:      make([]bool, count),
		}
		for j := 0; j < count; j++ {
			mc.devices[i].CanPeer[j] = C.tw_multi_can_peer(C.int(i), C.int(j)) == 1
		}
	}

	// Log topology
	for i := 0; i < count; i++ {
		d := mc.devices[i]
		log.Printf("[multi-gpu] Device %d: %s (%d MB, %d SMs, compute %d.%d)",
			i, d.Name, d.TotalMem/1024/1024, d.SMCount, d.ComputeMajor, d.ComputeMinor)
		peers := []int{}
		for j := 0; j < count; j++ {
			if i != j && d.CanPeer[j] {
				peers = append(peers, j)
			}
		}
		if len(peers) > 0 {
			log.Printf("[multi-gpu]   P2P peers: %v (NVLink/PCIe direct)", peers)
		}
	}

	return mc
}

// Device returns info for a specific device.
func (mc *MultiCUDA) Device(id int) DeviceInfo {
	mc.devices[id].FreeMem = uint64(C.tw_multi_device_free(C.int(id)))
	return mc.devices[id]
}

// DeviceTensor represents a tensor on a specific GPU device.
type DeviceTensor struct {
	DeviceID int
	Ptr      unsafe.Pointer
	Shape    []int
	Size     int // total elements
}

// Alloc allocates GPU memory on a specific device.
func (mc *MultiCUDA) Alloc(dev, sizeFloats int) unsafe.Pointer {
	mc.poolMus[dev].Lock()
	if free := mc.pools[dev][sizeFloats]; len(free) > 0 {
		ptr := free[len(free)-1]
		mc.pools[dev][sizeFloats] = free[:len(free)-1]
		mc.poolMus[dev].Unlock()
		return ptr
	}
	mc.poolMus[dev].Unlock()
	return C.tw_multi_alloc(C.int(dev), C.size_t(sizeFloats*4))
}

// Free returns GPU memory to the pool (or actually frees if pool full).
func (mc *MultiCUDA) Free(dev, sizeFloats int, ptr unsafe.Pointer) {
	mc.poolMus[dev].Lock()
	if len(mc.pools[dev][sizeFloats]) >= 4 {
		mc.poolMus[dev].Unlock()
		C.tw_multi_free(C.int(dev), ptr)
		return
	}
	mc.pools[dev][sizeFloats] = append(mc.pools[dev][sizeFloats], ptr)
	mc.poolMus[dev].Unlock()
}

// Upload copies host data to a specific device.
func (mc *MultiCUDA) Upload(dev int, data []float32) *DeviceTensor {
	size := len(data)
	ptr := mc.Alloc(dev, size)
	C.tw_multi_upload(C.int(dev), ptr, unsafe.Pointer(&data[0]), C.size_t(size*4))
	return &DeviceTensor{DeviceID: dev, Ptr: ptr, Size: size}
}

// Download copies device data to host.
func (mc *MultiCUDA) Download(dt *DeviceTensor) []float32 {
	data := make([]float32, dt.Size)
	C.tw_multi_download(C.int(dt.DeviceID), unsafe.Pointer(&data[0]), dt.Ptr, C.size_t(dt.Size*4))
	return data
}

// Release returns a tensor's memory to the pool.
func (mc *MultiCUDA) Release(dt *DeviceTensor) {
	if dt != nil && dt.Ptr != nil {
		mc.Free(dt.DeviceID, dt.Size, dt.Ptr)
		dt.Ptr = nil
	}
}

// PeerCopyInto copies data from src device into an existing dst pointer on dstDev.
func (mc *MultiCUDA) PeerCopyInto(srcDev int, src unsafe.Pointer, dstDev int, dst unsafe.Pointer, bytes int) {
	C.tw_multi_peer_copy(C.int(srcDev), src, C.int(dstDev), dst, C.size_t(bytes))
}

// PeerCopy copies data between two devices. Uses NVLink if available.
func (mc *MultiCUDA) PeerCopy(src *DeviceTensor, dstDev int) *DeviceTensor {
	dst := &DeviceTensor{
		DeviceID: dstDev,
		Ptr:      mc.Alloc(dstDev, src.Size),
		Shape:    src.Shape,
		Size:     src.Size,
	}
	C.tw_multi_peer_copy(C.int(src.DeviceID), src.Ptr, C.int(dstDev), dst.Ptr, C.size_t(src.Size*4))
	return dst
}

// MatMulOnDevice runs TF32 matmul on a specific device.
func (mc *MultiCUDA) MatMulOnDevice(dev int, a, b unsafe.Pointer, m, k, n int) unsafe.Pointer {
	size := m * n
	out := mc.Alloc(dev, size)
	C.tw_multi_sgemm(C.int(dev), (*C.float)(a), (*C.float)(b), (*C.float)(out), C.int(m), C.int(k), C.int(n))
	return out
}

// MatMulFP16TransBOnDevice runs FP16 C = A @ B^T on a specific device. All FP16.
func (mc *MultiCUDA) MatMulFP16TransBOnDevice(dev int, a, b, out unsafe.Pointer, m, k, n int) {
	C.tw_multi_hgemm_transB(C.int(dev), a, b, out, C.int(m), C.int(k), C.int(n))
}

// MatMulFP16FP32OutOnDevice runs FP16 C = A @ B on device. FP16 in, FP32 out.
func (mc *MultiCUDA) MatMulFP16FP32OutOnDevice(dev int, a, b unsafe.Pointer, out unsafe.Pointer, m, k, n int) {
	C.tw_multi_hgemm_fp32out(C.int(dev), a, b, (*C.float)(out), C.int(m), C.int(k), C.int(n))
}

// MatMulFP16TransAFP32OutOnDevice: C[k,n] = A[m,k]^T @ B[m,n]. FP16 in, FP32 out.
func (mc *MultiCUDA) MatMulFP16TransAFP32OutOnDevice(dev int, a, b unsafe.Pointer, out unsafe.Pointer, m, k, n int) {
	C.tw_multi_hgemm_transA_fp32out(C.int(dev), a, b, (*C.float)(out), C.int(m), C.int(k), C.int(n))
}

// MatMulFP16OnDevice runs FP16 cublasLt matmul on a specific device.
func (mc *MultiCUDA) MatMulFP16OnDevice(dev int, a, b unsafe.Pointer, m, k, n int) unsafe.Pointer {
	fp16Bytes := m * n * 2
	poolKey := (fp16Bytes + 3) / 4
	out := mc.Alloc(dev, poolKey)
	C.tw_multi_hgemm_lt(C.int(dev), a, b, out, C.int(m), C.int(k), C.int(n))
	return out
}

// SyncDevice synchronizes a specific device.
func (mc *MultiCUDA) SyncDevice(dev int) {
	C.tw_multi_sync(C.int(dev))
}

// SyncAll synchronizes all devices.
func (mc *MultiCUDA) SyncAll() {
	C.tw_multi_sync_all()
}

// ParallelMatMul fires matmuls on multiple devices from a single thread.
// Zero goroutine overhead — all launches are queued async, then synced at the end.
func (mc *MultiCUDA) ParallelMatMul(ops []ParallelOp) {
	mc.ParallelMatMulBatch(ops, 1)
}

// ParallelMatMulBatch fires N iterations of parallel matmul across devices.
// ALL iterations on ALL devices are queued before any sync — maximum overlap.
// This matches PyTorch's approach: queue everything, sync once at the end.
func (mc *MultiCUDA) ParallelMatMulBatch(ops []ParallelOp, iters int) {
	if len(ops) == 0 {
		return
	}
	cOps := make([]C.multi_matmul_op_t, len(ops))
	for i, op := range ops {
		cOps[i].dev = C.int(op.DeviceID)
		cOps[i].dA = (*C.float)(op.A)
		cOps[i].dB = (*C.float)(op.B)
		cOps[i].dC = (*C.float)(op.C)
		cOps[i].m = C.int(op.M)
		cOps[i].k = C.int(op.K)
		cOps[i].n = C.int(op.N)
	}
	C.tw_multi_matmul_batch(&cOps[0], C.int(len(ops)), C.int(iters))
}

// ParallelOp describes a matmul to fire on a specific device.
type ParallelOp struct {
	DeviceID   int
	A, B, C    unsafe.Pointer
	M, K, N    int
}

// ParallelFP16Op describes an FP16 matmul to fire on a specific device.
type ParallelFP16Op struct {
	DeviceID   int
	A, B, C    unsafe.Pointer
	M, K, N    int
	TransB     bool // true = C = A @ B^T, false = C = A @ B
}

// ParallelMatMulFP16 fires all-FP16 cublasLt matmuls across devices.
// Zero goroutine overhead — all launches queued async via per-device streams.
func (mc *MultiCUDA) ParallelMatMulFP16(ops []ParallelFP16Op) {
	if len(ops) == 0 {
		return
	}
	cOps := make([]C.multi_hgemm_op_t, len(ops))
	for i, op := range ops {
		cOps[i].dev = C.int(op.DeviceID)
		cOps[i].dA = op.A
		cOps[i].dB = op.B
		cOps[i].dC = op.C
		cOps[i].m = C.int(op.M)
		cOps[i].k = C.int(op.K)
		cOps[i].n = C.int(op.N)
		if op.TransB {
			cOps[i].transB = C.int(1)
		}
	}
	C.tw_multi_hgemm_batch(&cOps[0], C.int(len(ops)))
}

// SetDevice switches the CUDA context to a specific GPU.
// All subsequent kernel launches (including dlopen'd kernels on the default stream)
// will fire on this device until the next SetDevice call.
func SetDevice(dev int) {
	C.tw_set_device(C.int(dev))
}

// AllocFP16OnDevice allocates an FP16 buffer on a specific GPU.
func (mc *MultiCUDA) AllocFP16OnDevice(dev, nElements int) unsafe.Pointer {
	return C.tw_multi_alloc_fp16(C.int(dev), C.int(nElements))
}

// AllocBytesOnDevice allocates raw bytes on a specific GPU.
func (mc *MultiCUDA) AllocBytesOnDevice(dev, bytes int) unsafe.Pointer {
	return C.tw_multi_alloc_bytes(C.int(dev), C.size_t(bytes))
}

// ZerosFP32OnDevice allocates and zeros an FP32 buffer on a specific GPU, wrapped as a Tensor.
func (mc *MultiCUDA) ZerosFP32OnDevice(dev, nFloats int) *Tensor {
	ptr := C.tw_multi_zeros_fp32(C.int(dev), C.int(nFloats))
	return TensorFromDevicePtr(unsafe.Pointer(ptr), nFloats)
}

// ZeroOnDevice zeros a buffer on a specific device.
func (mc *MultiCUDA) ZeroOnDevice(dev int, ptr unsafe.Pointer, bytes int) {
	C.tw_multi_zero(C.int(dev), ptr, C.size_t(bytes))
}

// UploadFP32OnDevice copies host FP32 data to a buffer on a specific device.
func (mc *MultiCUDA) UploadFP32OnDevice(dev int, dst unsafe.Pointer, src []float32) {
	C.tw_multi_upload_fp32(C.int(dev), dst, (*C.float)(unsafe.Pointer(&src[0])), C.int(len(src)))
}

// DownloadFP32FromDevice copies device FP32 data to host.
func (mc *MultiCUDA) DownloadFP32FromDevice(dev int, src unsafe.Pointer, nFloats int) []float32 {
	data := make([]float32, nFloats)
	C.tw_multi_download_fp32(C.int(dev), (*C.float)(unsafe.Pointer(&data[0])), src, C.int(nFloats))
	return data
}

// FromHostFP32OnDevice allocates + uploads FP32 data to a specific device as a Tensor.
func (mc *MultiCUDA) FromHostFP32OnDevice(dev int, data []float32, shape []int) *Tensor {
	size := 1
	for _, s := range shape { size *= s }
	ptr := C.tw_multi_zeros_fp32(C.int(dev), C.int(size))
	C.tw_multi_upload_fp32(C.int(dev), unsafe.Pointer(ptr), (*C.float)(unsafe.Pointer(&data[0])), C.int(size))
	return &Tensor{Shape: shape, Size: size, device: &rawPtr{unsafe.Pointer(ptr)}}
}

// FromHostFP16OnDevice uploads FP32 host data as FP16 to a specific device.
// Converts on GPU 0 then P2P copies if dev != 0.
func (mc *MultiCUDA) FromHostFP16OnDevice(dev int, data []float32, shape []int, cuda0FP16Convert func([]float32, []int) *Tensor) *Tensor {
	size := 1
	for _, s := range shape { size *= s }
	if dev == 0 {
		return cuda0FP16Convert(data, shape)
	}
	tmp := cuda0FP16Convert(data, shape)
	dstPtr := C.tw_multi_alloc_fp16(C.int(dev), C.int(size))
	C.tw_multi_peer_copy(C.int(0), tmp.DevicePtr(), C.int(dev), unsafe.Pointer(dstPtr), C.size_t(size*2))
	return TensorFromDevicePtr(unsafe.Pointer(dstPtr), size)
}

// SparsePeerCopy copies only the specified rows between devices.
// Each row is cols elements wide. elemSize is bytes per element (4 for FP32, 2 for FP16).
func (mc *MultiCUDA) SparsePeerCopy(srcDev int, src unsafe.Pointer, dstDev int, dst unsafe.Pointer,
	rows []int32, cols, elemSize int) {
	if len(rows) == 0 { return }
	C.tw_multi_sparse_p2p(C.int(srcDev), src, C.int(dstDev), dst,
		(*C.int)(unsafe.Pointer(&rows[0])), C.int(len(rows)), C.int(cols), C.int(elemSize))
}

// PeerCopyAsync copies between devices without blocking the caller.
func (mc *MultiCUDA) PeerCopyAsync(srcDev int, src unsafe.Pointer, dstDev int, dst unsafe.Pointer, bytes int) {
	C.tw_multi_peer_copy_async(C.int(srcDev), src, C.int(dstDev), dst, C.size_t(bytes))
}

// InterleaveExtract extracts every Nth row (stride) starting at offset into a packed buffer.
// src[seqLen, cols] → dst[seqLen/stride, cols]. Both on current device. elemSize: 2=FP16, 4=FP32.
func InterleaveExtract(src, dst unsafe.Pointer, seqLen, cols, stride, offset, elemSize int) {
	C.tw_interleave_extract(src, dst, C.int(seqLen), C.int(cols), C.int(stride), C.int(offset), C.int(elemSize))
}

// InterleaveInsert scatters packed rows back into strided positions.
// src[seqLen/stride, cols] → dst[seqLen, cols] at rows where row%stride==offset.
func InterleaveInsert(src, dst unsafe.Pointer, seqLen, cols, stride, offset, elemSize int) {
	C.tw_interleave_insert(src, dst, C.int(seqLen), C.int(cols), C.int(stride), C.int(offset), C.int(elemSize))
}

// String returns a summary of the multi-GPU topology.
func (mc *MultiCUDA) String() string {
	s := fmt.Sprintf("MultiCUDA: %d devices\n", mc.DeviceCount)
	for i := 0; i < mc.DeviceCount; i++ {
		d := mc.Device(i)
		s += fmt.Sprintf("  [%d] %s — %d MB total, %d MB free, %d SMs\n",
			i, d.Name, d.TotalMem/1024/1024, d.FreeMem/1024/1024, d.SMCount)
	}
	return s
}
