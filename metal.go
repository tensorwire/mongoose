//go:build darwin && cgo

package mongoose

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework Foundation
#include <stdlib.h>
#include <stdint.h>

typedef void* MTLDeviceRef;
typedef void* MTLCommandQueueRef;
typedef void* MTLBufferRef;

int mtl_init(void);
const char* mtl_device_name(void);
uint64_t mtl_recommended_max_working_set_size(void);

MTLBufferRef mtl_alloc(size_t bytes);
void mtl_free(MTLBufferRef buf);
void mtl_upload(MTLBufferRef buf, const void* src, size_t bytes);
void mtl_download(void* dst, MTLBufferRef buf, size_t bytes);
void mtl_zero(MTLBufferRef buf, size_t bytes);

int mtl_sgemm(MTLBufferRef A, MTLBufferRef B, MTLBufferRef C, int m, int k, int n);
int mtl_sgemm_transA(MTLBufferRef A, MTLBufferRef B, MTLBufferRef C, int m, int k, int n);
int mtl_sgemm_transB(MTLBufferRef A, MTLBufferRef B, MTLBufferRef C, int m, int k, int n);

void mtl_begin_batch(void);
void mtl_end_batch(void);

int mtl_graph_sgemm(void* aRef, void* bRef, void* cRef, int m, int k, int n, int transA, int transB);
void mtl_graph_sync(void);

int mtl_graph_build_full(int dim, int kvDim, int headDim,
                         int nHeads, int nKVHeads, int ffnDim,
                         int vocabSize, int nLayers, int seqLen,
                         float ropeTheta, int mode);
float mtl_graph_train_step(int* tokens, int* targets, int n,
                           void** weightBufs, void** gradBufs, int nWeights,
                           float learningRate, int mode);
int mtl_graph_full_built(void);
int mtl_graph_num_weights(void);
int mtl_graph_set_variable(int varIdx, const float* data, int nFloats);
int mtl_graph_read_variable(int varIdx, float* dst, int nFloats);
int mtl_graph_apply_weights(int varIdx, const float* data, int nFloats);
int mtl_graph_num_diffable(void);
int mtl_graph_accum_adam_step(float learningRate, float accumScale);

int mtl_init_compute(void);
int mtl_compute_ready(void);
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

type Metal struct {
	deviceName string
	pool       map[int][]C.MTLBufferRef
	poolMu     sync.Mutex
}

func NewMetal() *Metal {
	ret := C.mtl_init()
	if ret != 0 {
		log.Printf("WARN mongoose => Metal init failed (code %d)", ret)
		return nil
	}

	name := C.GoString(C.mtl_device_name())
	m := &Metal{deviceName: name, pool: make(map[int][]C.MTLBufferRef)}

	if C.mtl_init_compute() == 0 {
		log.Printf("[mongoose] Metal initialized: %s (compute kernels ready)", name)
	} else {
		log.Printf("[mongoose] Metal initialized: %s (compute kernels failed, CPU fallback)", name)
	}

	return m
}

func (m *Metal) Name() string     { return fmt.Sprintf("metal/%s", m.deviceName) }
func (m *Metal) Close()           {}
func (m *Metal) BeginBatch()      { C.mtl_begin_batch() }
func (m *Metal) EndBatch()        { C.mtl_end_batch() }
func (m *Metal) Sync()            { C.mtl_graph_sync() }
func MtlComputeReady() bool       { return C.mtl_compute_ready() == 1 }

func (m *Metal) MatMul(a, b []float32, rows, k, n int) []float32 {
	bufA := m.poolGet(len(a))
	bufB := m.poolGet(len(b))
	bufC := m.poolGet(rows * n)

	C.mtl_upload(bufA, unsafe.Pointer(&a[0]), C.size_t(len(a)*4))
	C.mtl_upload(bufB, unsafe.Pointer(&b[0]), C.size_t(len(b)*4))
	C.mtl_sgemm(bufA, bufB, bufC, C.int(rows), C.int(k), C.int(n))

	out := make([]float32, rows*n)
	C.mtl_download(unsafe.Pointer(&out[0]), bufC, C.size_t(rows*n*4))

	m.poolPut(len(a), bufA)
	m.poolPut(len(b), bufB)
	m.poolPut(rows*n, bufC)
	return out
}

func (m *Metal) RMSNorm(x, weight []float32, eps float32) {
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

func (m *Metal) SoftMax(x []float32, n int) {
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

func (m *Metal) ReLU(x []float32) {
	for i := range x {
		if x[i] < 0 {
			x[i] = 0
		}
	}
}

func (m *Metal) VRAM() uint64 {
	return uint64(C.mtl_recommended_max_working_set_size())
}

func (m *Metal) Benchmark() float64 {
	const dim = 512
	a := make([]float32, dim*dim)
	b := make([]float32, dim*dim)
	for i := range a {
		a[i] = 0.001 * float32(i%1000)
		b[i] = 0.001 * float32(i%997)
	}

	runtime.GC()
	m.MatMul(a, b, dim, dim, dim)

	start := time.Now()
	iterations := 50
	for range iterations {
		m.MatMul(a, b, dim, dim, dim)
	}
	elapsed := time.Since(start)

	flops := float64(2*dim*dim*dim*iterations) / elapsed.Seconds()
	return flops / 1e9
}

// TrainEngine — BLAS on []float32 via MPS
func (m *Metal) MatMulTransBInto(out, A, B []float32, rows, k, n int) {
	bufA := m.poolGet(rows * k)
	bufB := m.poolGet(n * k)
	bufC := m.poolGet(rows * n)
	C.mtl_upload(bufA, unsafe.Pointer(&A[0]), C.size_t(rows*k*4))
	C.mtl_upload(bufB, unsafe.Pointer(&B[0]), C.size_t(n*k*4))
	C.mtl_sgemm_transB(bufA, bufB, bufC, C.int(rows), C.int(k), C.int(n))
	C.mtl_download(unsafe.Pointer(&out[0]), bufC, C.size_t(rows*n*4))
	m.poolPut(rows*k, bufA)
	m.poolPut(n*k, bufB)
	m.poolPut(rows*n, bufC)
}

func (m *Metal) MatMulInto(out, A, B []float32, rows, k, n int) {
	bufA := m.poolGet(rows * k)
	bufB := m.poolGet(k * n)
	bufC := m.poolGet(rows * n)
	C.mtl_upload(bufA, unsafe.Pointer(&A[0]), C.size_t(rows*k*4))
	C.mtl_upload(bufB, unsafe.Pointer(&B[0]), C.size_t(k*n*4))
	C.mtl_sgemm(bufA, bufB, bufC, C.int(rows), C.int(k), C.int(n))
	C.mtl_download(unsafe.Pointer(&out[0]), bufC, C.size_t(rows*n*4))
	m.poolPut(rows*k, bufA)
	m.poolPut(k*n, bufB)
	m.poolPut(rows*n, bufC)
}

func (m *Metal) MatMulAddInto(G, A, B []float32, rows, k, n int) {
	bufA := m.poolGet(rows * k)
	bufB := m.poolGet(k * n)
	bufC := m.poolGet(rows * n)
	C.mtl_upload(bufA, unsafe.Pointer(&A[0]), C.size_t(rows*k*4))
	C.mtl_upload(bufB, unsafe.Pointer(&B[0]), C.size_t(k*n*4))
	C.mtl_sgemm_transA(bufA, bufB, bufC, C.int(rows), C.int(k), C.int(n))
	tmp := make([]float32, rows*n)
	C.mtl_download(unsafe.Pointer(&tmp[0]), bufC, C.size_t(rows*n*4))
	for i := range G {
		G[i] += tmp[i]
	}
	m.poolPut(rows*k, bufA)
	m.poolPut(k*n, bufB)
	m.poolPut(rows*n, bufC)
}

func (m *Metal) MatMulTransA(A, B []float32, rows, k, n int) []float32 {
	out := make([]float32, rows*n)
	bufA := m.poolGet(rows * k)
	bufB := m.poolGet(k * n)
	bufC := m.poolGet(rows * n)
	C.mtl_upload(bufA, unsafe.Pointer(&A[0]), C.size_t(rows*k*4))
	C.mtl_upload(bufB, unsafe.Pointer(&B[0]), C.size_t(k*n*4))
	C.mtl_sgemm_transA(bufA, bufB, bufC, C.int(rows), C.int(k), C.int(n))
	C.mtl_download(unsafe.Pointer(&out[0]), bufC, C.size_t(rows*n*4))
	m.poolPut(rows*k, bufA)
	m.poolPut(k*n, bufB)
	m.poolPut(rows*n, bufC)
	return out
}

func (m *Metal) Nrm2(x []float32) float32 {
	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	return float32(math.Sqrt(ss))
}

func (m *Metal) Scal(x []float32, alpha float32) {
	for i := range x {
		x[i] *= alpha
	}
}

func (m *Metal) GER(G, x, y []float32, rows, n int, alpha float32) {
	for i := 0; i < rows; i++ {
		for j := 0; j < n; j++ {
			G[i*n+j] += alpha * x[i] * y[j]
		}
	}
}

func (m *Metal) AdamWStep(D, G, M, V []float32, n int, lr, beta1, beta2, bc1, bc2, eps, wd float32) {
	for i := 0; i < n; i++ {
		M[i] = beta1*M[i] + (1-beta1)*G[i]
		V[i] = beta2*V[i] + (1-beta2)*G[i]*G[i]
		mHat := M[i] / bc1
		vHat := V[i] / bc2
		D[i] -= lr * (mHat/(float32(math.Sqrt(float64(vHat)))+eps) + wd*D[i])
	}
}

// GraphTrainEngine — fused dispatch
func (m *Metal) BuildFullGraph(dim, kvDim, headDim, nHeads, nKVHeads, ffnDim,
	vocabSize, nLayers, seqLen int, ropeTheta float64, mode int) int {
	return int(C.mtl_graph_build_full(
		C.int(dim), C.int(kvDim), C.int(headDim),
		C.int(nHeads), C.int(nKVHeads), C.int(ffnDim),
		C.int(vocabSize), C.int(nLayers), C.int(seqLen),
		C.float(ropeTheta), C.int(mode)))
}

func (m *Metal) GraphTrainStepAdam(tokens, targets []int32, lr float32) float32 {
	return float32(C.mtl_graph_train_step(
		(*C.int)(unsafe.Pointer(&tokens[0])),
		(*C.int)(unsafe.Pointer(&targets[0])),
		C.int(len(tokens)),
		nil, nil, 0, C.float(lr), 1))
}

func (m *Metal) GraphFullBuilt() bool { return C.mtl_graph_full_built() == 1 }
func (m *Metal) GraphNumWeights() int { return int(C.mtl_graph_num_weights()) }
func (m *Metal) GraphNumDiffable() int { return int(C.mtl_graph_num_diffable()) }

func (m *Metal) GraphSetVariable(varIdx int, data []float32) int {
	return int(C.mtl_graph_set_variable(C.int(varIdx), (*C.float)(unsafe.Pointer(&data[0])), C.int(len(data))))
}

func (m *Metal) GraphReadVariable(varIdx int, dst []float32) int {
	return int(C.mtl_graph_read_variable(C.int(varIdx), (*C.float)(unsafe.Pointer(&dst[0])), C.int(len(dst))))
}

func (m *Metal) GraphApplyWeights(varIdx int, data []float32) int {
	return int(C.mtl_graph_apply_weights(C.int(varIdx), (*C.float)(unsafe.Pointer(&data[0])), C.int(len(data))))
}

func (m *Metal) GraphAccumAdamStep(lr float32, accumScale float32) int {
	return int(C.mtl_graph_accum_adam_step(C.float(lr), C.float(accumScale)))
}

func (m *Metal) GraphTrainStepAccum(tokens, targets []int32) float32 {
	return float32(C.mtl_graph_train_step(
		(*C.int)(unsafe.Pointer(&tokens[0])),
		(*C.int)(unsafe.Pointer(&targets[0])),
		C.int(len(tokens)),
		nil, nil, 0, 0, 2))
}

func (m *Metal) poolGet(sizeFloats int) C.MTLBufferRef {
	m.poolMu.Lock()
	if free := m.pool[sizeFloats]; len(free) > 0 {
		buf := free[len(free)-1]
		m.pool[sizeFloats] = free[:len(free)-1]
		m.poolMu.Unlock()
		return buf
	}
	m.poolMu.Unlock()
	return C.mtl_alloc(C.size_t(sizeFloats * 4))
}

func (m *Metal) poolPut(sizeFloats int, buf C.MTLBufferRef) {
	m.poolMu.Lock()
	if len(m.pool[sizeFloats]) >= 8 {
		m.poolMu.Unlock()
		C.mtl_free(buf)
		return
	}
	m.pool[sizeFloats] = append(m.pool[sizeFloats], buf)
	m.poolMu.Unlock()
}
