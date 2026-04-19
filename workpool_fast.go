//go:build linux && cgo

package mongoose

/*
#cgo LDFLAGS: -lcublas -lcublasLt -lcudart -lpthread
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <stdio.h>

#define MAX_QUEUE_SIZE 4096
#define MAX_WORKERS 16

typedef struct {
    float* A;
    float* B;
    float* C;
    int m, k, n;
    int id;
} fast_work_item_t;

typedef void (*fast_matmul_fn)(const float* A, const float* B, float* C, int m, int k, int n, void* ctx);

typedef struct {
    int id;
    fast_matmul_fn fn;
    void* ctx;
    fast_work_item_t* items; // pointer into shared items array
    int start_idx;
    int end_idx;
    pthread_t thread;
    long items_done;  // plain long, written only by owning thread, read after join
} fast_worker_t;

// Worker thread — processes its slice, counts items, exits.
void* fast_worker_loop(void* arg) {
    fast_worker_t* w = (fast_worker_t*)arg;
    w->items_done = 0;
    for (int i = w->start_idx; i < w->end_idx; i++) {
        fast_work_item_t* item = &w->items[i];
        w->fn(item->A, item->B, item->C, item->m, item->k, item->n, w->ctx);
        w->items_done++;
    }
    return NULL;
}

// CPU matmul for baseline comparison
void fast_cpu_matmul(const float* A, const float* B, float* C, int m, int k, int n, void* ctx) {
    (void)ctx;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int p = 0; p < k; p++) {
                sum += A[i*k + p] * B[p*n + j];
            }
            C[i*n + j] = sum;
        }
    }
}

// CUDA matmul — ctx is a cublasHandle_t* with device already set
extern cublasHandle_t tw_cublas_handle;
extern int tw_cuda_initialized;

void fast_cuda_matmul(const float* A, const float* B, float* C, int m, int k, int n, void* ctx) {
    int dev = *(int*)ctx;
    cudaSetDevice(dev);

    void *dA, *dB, *dC;
    size_t sA = m * k * sizeof(float);
    size_t sB = k * n * sizeof(float);
    size_t sC = m * n * sizeof(float);
    cudaMalloc(&dA, sA);
    cudaMalloc(&dB, sB);
    cudaMalloc(&dC, sC);
    cudaMemcpy(dA, A, sA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sB, cudaMemcpyHostToDevice);

    float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(tw_cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
        &alpha, dB, CUDA_R_32F, n, dA, CUDA_R_32F, k,
        &beta, dC, CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaDeviceSynchronize();

    cudaMemcpy(C, dC, sC, cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

typedef struct {
    fast_work_item_t* items;
    int count;
    int num_workers;
    fast_worker_t workers[MAX_WORKERS];
    long wall_ns;
} fast_batch_result_t;

// Run a batch: partition, launch pthreads, join, return per-worker counts.
void fast_run_batch(fast_batch_result_t* result, fast_matmul_fn* fns, void** ctxs) {
    int count = result->count;
    int nw = result->num_workers;
    if (nw > MAX_WORKERS) nw = MAX_WORKERS;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    // Partition
    int assigned = 0;
    for (int i = 0; i < nw; i++) {
        int share = count / nw;
        if (i < count % nw) share++;
        result->workers[i].id = i;
        result->workers[i].fn = fns[i];
        result->workers[i].ctx = ctxs[i];
        result->workers[i].items = result->items;
        result->workers[i].start_idx = assigned;
        result->workers[i].end_idx = assigned + share;
        result->workers[i].items_done = 0;
        assigned += share;
    }

    // Launch
    for (int i = 0; i < nw; i++) {
        pthread_create(&result->workers[i].thread, NULL, fast_worker_loop, &result->workers[i]);
    }

    // Join
    for (int i = 0; i < nw; i++) {
        pthread_join(result->workers[i].thread, NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    result->wall_ns = (t1.tv_sec - t0.tv_sec) * 1000000000L + (t1.tv_nsec - t0.tv_nsec);
}

long fast_worker_items(fast_batch_result_t* r, int worker_id) {
    return r->workers[worker_id].items_done;
}
*/
import "C"

import (
	"fmt"
	"time"
	"unsafe"
)

// FastWorkPool uses pthreads — zero Go runtime overhead per work item.
type FastWorkPool struct{}

// FastWorkResult holds results from a fast pool run.
type FastWorkResult struct {
	TotalItems int
	WallTime   time.Duration
	PerWorker  []int
}

// RunFastCPUPool runs a batch of CPU matmuls across N pthreads.
func RunFastCPUPool(numWorkers int, dim int, batchSize int) FastWorkResult {
	return runFastPool(numWorkers, dim, batchSize, false)
}

// RunFastCUDAPool runs a batch of CUDA matmuls across N pthreads.
// Each thread calls cuBLAS directly — no Go/CGo per item.
func RunFastCUDAPool(numWorkers int, dim int, batchSize int) FastWorkResult {
	return runFastPool(numWorkers, dim, batchSize, true)
}

func runFastPool(numWorkers int, dim int, batchSize int, useCUDA bool) FastWorkResult {
	if batchSize > 4096 {
		batchSize = 4096
	}
	if numWorkers > 16 {
		numWorkers = 16
	}

	matSize := dim * dim

	// Allocate items + matrices in C memory
	items := (*C.fast_work_item_t)(C.calloc(C.size_t(batchSize), C.size_t(unsafe.Sizeof(C.fast_work_item_t{}))))
	defer C.free(unsafe.Pointer(items))

	itemSlice := unsafe.Slice((*C.fast_work_item_t)(items), batchSize)
	for i := 0; i < batchSize; i++ {
		itemSlice[i].A = (*C.float)(C.malloc(C.size_t(matSize * 4)))
		itemSlice[i].B = (*C.float)(C.malloc(C.size_t(matSize * 4)))
		itemSlice[i].C = (*C.float)(C.malloc(C.size_t(matSize * 4)))
		itemSlice[i].m = C.int(dim)
		itemSlice[i].k = C.int(dim)
		itemSlice[i].n = C.int(dim)
		itemSlice[i].id = C.int(i)

		aSlice := unsafe.Slice((*C.float)(unsafe.Pointer(itemSlice[i].A)), matSize)
		bSlice := unsafe.Slice((*C.float)(unsafe.Pointer(itemSlice[i].B)), matSize)
		for j := 0; j < matSize; j++ {
			aSlice[j] = C.float(0.001 * float32(j%1000))
			bSlice[j] = C.float(0.001 * float32(j%997))
		}
	}

	// Set up function pointers and contexts
	fns := make([]C.fast_matmul_fn, numWorkers)
	ctxs := make([]unsafe.Pointer, numWorkers)

	// Device IDs must be in C memory (CGo forbids Go pointers in C contexts)
	var devIDs *C.int
	if useCUDA {
		devIDs = (*C.int)(C.calloc(C.size_t(numWorkers), C.size_t(unsafe.Sizeof(C.int(0)))))
		defer C.free(unsafe.Pointer(devIDs))
	}

	for i := 0; i < numWorkers; i++ {
		if useCUDA {
			fns[i] = C.fast_matmul_fn(C.fast_cuda_matmul)
			devSlice := unsafe.Slice(devIDs, numWorkers)
			devSlice[i] = C.int(0) // device 0 for all workers (single GPU)
			ctxs[i] = unsafe.Pointer(&devSlice[i])
		} else {
			fns[i] = C.fast_matmul_fn(C.fast_cpu_matmul)
			ctxs[i] = nil
		}
	}

	// Build result struct
	var result C.fast_batch_result_t
	result.items = items
	result.count = C.int(batchSize)
	result.num_workers = C.int(numWorkers)

	C.fast_run_batch(&result, &fns[0], &ctxs[0])

	wallTime := time.Duration(int64(result.wall_ns))

	// Read per-worker counts
	perWorker := make([]int, numWorkers)
	for i := 0; i < numWorkers; i++ {
		perWorker[i] = int(C.fast_worker_items(&result, C.int(i)))
	}

	// Free matrices
	for i := 0; i < batchSize; i++ {
		C.free(unsafe.Pointer(itemSlice[i].A))
		C.free(unsafe.Pointer(itemSlice[i].B))
		C.free(unsafe.Pointer(itemSlice[i].C))
	}

	return FastWorkResult{
		TotalItems: batchSize,
		WallTime:   wallTime,
		PerWorker:  perWorker,
	}
}

// BenchmarkFastVsGo compares pthread pool vs Go WorkPool overhead.
func BenchmarkFastVsGo(dim, batchSize, numWorkers int) {
	fmt.Printf("\n=== Fast (pthread) vs Go WorkPool: %dx%d, batch=%d, workers=%d ===\n",
		dim, dim, batchSize, numWorkers)

	// Fast (pthread) pool
	fast := RunFastCPUPool(numWorkers, dim, batchSize)
	fastGFLOPS := float64(2*dim*dim*dim*batchSize) / fast.WallTime.Seconds() / 1e9

	// Go WorkPool with CPU workers
	goPool := NewWorkPool()
	for i := 0; i < numWorkers; i++ {
		goPool.AddEngine(fmt.Sprintf("cpu-%d", i), &CPU{}, 1.0)
	}
	a := make([]float32, dim*dim)
	b := make([]float32, dim*dim)
	for j := range a {
		a[j] = 0.001 * float32(j%1000)
		b[j] = 0.001 * float32(j%997)
	}
	items := make([]WorkItem, batchSize)
	for i := range items {
		items[i] = WorkItem{ID: i, A: a, B: b, M: dim, K: dim, N: dim}
	}
	_, goTime, goCounts := goPool.RunTimed(items)
	goGFLOPS := float64(2*dim*dim*dim*batchSize) / goTime.Seconds() / 1e9

	overhead := goTime.Seconds()/fast.WallTime.Seconds() - 1

	fmt.Printf("  pthread:  %v (%.1f GFLOPS)\n", fast.WallTime, fastGFLOPS)
	fmt.Printf("  Go pool:  %v (%.1f GFLOPS)\n", goTime, goGFLOPS)
	fmt.Printf("  Go overhead: %.1f%%\n", overhead*100)
	fmt.Printf("  Fast workers: %v\n", fast.PerWorker)
	fmt.Printf("  Go workers: %v\n", goCounts)
}

// BenchmarkFastCUDA benchmarks CUDA matmul throughput via pthreads.
func BenchmarkFastCUDA(dim, batchSize int) {
	fmt.Printf("\n=== Fast CUDA (pthread, 1 worker): %dx%d, batch=%d ===\n", dim, dim, batchSize)

	result := RunFastCUDAPool(1, dim, batchSize)
	gflops := float64(2*dim*dim*dim*batchSize) / result.WallTime.Seconds() / 1e9

	fmt.Printf("  Time:    %v\n", result.WallTime)
	fmt.Printf("  GFLOPS:  %.1f\n", gflops)
	fmt.Printf("  Per-op:  %.1f µs\n", float64(result.WallTime.Microseconds())/float64(batchSize))
	fmt.Printf("  Workers: %v\n", result.PerWorker)
}
