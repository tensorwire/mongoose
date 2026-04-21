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

// Inference graph
int mtl_infer_build(int dim, int kvDim, int headDim,
                    int nHeads, int nKVHeads, int ffnDim,
                    int vocabSize, int nLayers, float ropeTheta);
int mtl_infer_num_weights(void);
int mtl_infer_set_weight(int idx, const float* data, int nFloats);
int mtl_infer_forward(float* hiddenIO, float* cosData, float* sinData,
                      float* qOut, float* kOut, float* vOut,
                      float* attnIn, float* logitsOut, int layer);
int mtl_infer_forward_b(float* hiddenIO, float* attnOut, int layer);

// Fused compute-shader inference (one command buffer per token)
int mtl_fused_build(int dim, int kvDim, int headDim,
                    int nHeads, int nKVHeads, int ffnDim,
                    int vocabSize, int nLayers, int maxSeq);
int mtl_fused_num_weights(void);
int mtl_fused_set_weight(int idx, const float* data, int nFloats);
int mtl_fused_step(const float* hiddenIn, const float* cosData, const float* sinData,
                   int pos, float* logitsOut);

// Fused single-dispatch inference (MPSGraph — deprecated)
int mtl_fused_infer_build(int dim, int kvDim, int headDim,
                          int nHeads, int nKVHeads, int ffnDim,
                          int vocabSize, int nLayers, int maxSeq,
                          float ropeTheta);
int mtl_fused_infer_num_weights(void);
int mtl_fused_infer_set_weight(int idx, const float* data, int nFloats);
int mtl_fused_infer_step(float* hiddenIn, float* cosData, float* sinData,
                         int pos, float* logitsOut);
int mtl_fused_infer_reset(void);

// Fused training compute kernels
void mtl_fused_begin(void);
void mtl_fused_end(void);
void mtl_fused_begin_slot(int slot);
void mtl_fused_end_slot(int slot);
void mtl_fused_set_slot(int slot);
void mtl_fused_sync_all(void);
void mtl_fused_gemm_bt(void* a, void* b, void* c, int M, int K, int N);
void mtl_fused_gemm_nn(void* a, void* b, void* c, int M, int K, int N);
void mtl_fused_gemm_tn(void* a, void* b, void* c, int M, int K, int N);
void mtl_fused_gemm_f32_bt(void* a, void* b, void* c, int M, int K, int N);
void mtl_fused_gemm_f32_nn(void* a, void* b, void* c, int M, int K, int N);
void mtl_fused_gemm_f32_tn(void* a, void* b, void* c, int M, int K, int N);
void mtl_fused_rmsnorm(void* x, void* w, void* scale, int seqLen, int dim);
void mtl_fused_rmsnorm_bwd(void* dOut, void* xIn, void* w, void* scale, void* dx, int seqLen, int dim);
void mtl_fused_rope(void* x, int headDim, int nHeads, float theta, int stride, int seqLen);
void mtl_fused_attn(void* q, void* k, void* v, void* out, void* scores, int dim, int kvDim, int headDim, int nHeads, int nKVHeads, int seqLen);
void mtl_fused_attention_bwd_q(void* dOut, void* q, void* k, void* v, void* scores, void* dQ, void* dK, void* dV, int dim, int kvDim, int headDim, int nHeads, int nKVHeads, int seqLen, int qLen);
void mtl_fused_silu_gate_mul(void* gate, void* up, void* out, int n);
void mtl_silu_gate_backward_gpu(void* dOut, void* gatePre, void* upOut, void* gateAct, void* dGatePre, void* dUp, int n);
void mtl_fused_add_inplace(void* a, void* b, int n);
void mtl_fused_copy(void* dst, void* src, int n);
void mtl_ce_loss(void* logits, void* targets, void* losses, int seqLen, int vocabSize);
void mtl_adamw_gpu(void* param, void* grad, void* m, void* v, float lr, float beta1, float beta2, float bc1, float bc2, float eps, float wd, int n);
void mtl_dna_rung_gpu(void* d1, void* g1, void* m1, void* v1, void* d2, void* g2, void* m2, void* v2, float bb1, float gly1, float hb1, float hb2, float gly2, float bb2, float bondStr, float lr, float beta1, float beta2, float bc1, float bc2, float eps, float wd, int n);
void mtl_dna_rung_warm(void* d1, void* g1, void* d2, void* g2, void* cache, int m1Off, int v1Off, int m2Off, int v2Off, float bb1, float gly1, float hb1, float hb2, float gly2, float bb2, float bondStr, float lr, float beta1, float beta2, float bc1, float bc2, float eps, float wd, int n);
void mtl_adamw_warm(void* param, void* grad, void* cache, int mOff, int vOff, float lr, float beta1, float beta2, float bc1, float bc2, float eps, float wd, int n);
void* mtl_shared_ptr(MTLBufferRef buf);
void mtl_grad_norm_sq(void* grad, void* out, int n);

// Fused training: grad clipping + needle optimizer (encode into active fused encoder)
void mtl_fused_zero_scalar(void* buf);
void mtl_fused_barrier_buffers(void);
void mtl_fused_grad_norm_sq(void* grad, void* sumSq, int n);
void mtl_fused_compute_clip_scale(void* sumSq, void* clipScale, float maxNorm);

// ICB training
typedef struct {
    int dim, kvDim, headDim, nHeads, nKVHeads, ffnDim, vocabSize, seqLen, nLayers;
    void* hidden; void* normedFinal; void* finalNorm; void* finalScales;
    void* lmMaxLogit; void* lmSumExp; void* lmLoss; void* targetsGPU;
    void* dHidden; void* dScratch; void* dEmbed;
    void* gradSumSq; void* clipScaleBuf; void* scores;
    void* embed; void* embedData; void* embedScales; void* embedDelta;
    void* embedMom; void* embedVel; void* embedMask; void* embedLive;
    void** norm1; void** norm2;
    void** a_xIn; void** a_normed; void** a_Q; void** a_K; void** a_V; void** a_attnOut;
    void** a_xMid; void** a_normed2; void** a_gatePre; void** a_upOut; void** a_ffnMid;
    void** a_rmsScale1; void** a_rmsScale2; void** a_gateAct;
    void** wq_data; void** wq_scales; void** wq_delta; void** wq_mom; void** wq_vel; void** wq_live; void** wq_mask;
    void** wk_data; void** wk_scales; void** wk_delta; void** wk_mom; void** wk_vel; void** wk_live; void** wk_mask;
    void** wv_data; void** wv_scales; void** wv_delta; void** wv_mom; void** wv_vel; void** wv_live; void** wv_mask;
    void** wo_data; void** wo_scales; void** wo_delta; void** wo_mom; void** wo_vel; void** wo_live; void** wo_mask;
    void** gate_data; void** gate_scales; void** gate_delta; void** gate_mom; void** gate_vel; void** gate_live; void** gate_mask;
    void** up_data; void** up_scales; void** up_delta; void** up_mom; void** up_vel; void** up_live; void** up_mask;
    void** down_data; void** down_scales; void** down_delta; void** down_mom; void** down_vel; void** down_live; void** down_mask;
    void** b_dFfnMid; void** b_dGate; void** b_dUp; void** b_dN2; void** b_dx;
    void** b_dAttnOut; void** b_dQ; void** b_dK; void** b_dV; void** b_dN1;
    void** b_dWDown; void** b_dWGate; void** b_dWUp; void** b_dWO; void** b_dWQ; void** b_dWK; void** b_dWV;
    void* lrBuf; void* bc1Buf; void* bc2Buf; void* maxNormBuf;
    void* bb1Buf; void* gly1Buf; void* hb1Buf; void* hb2Buf; void* gly2Buf; void* bb2Buf; void* bondStrBuf;
} ICBBuildParams;
int mtl_icb_build_training(ICBBuildParams* p);
void mtl_icb_execute_fwd(void);
void mtl_icb_execute_full(void);
void mtl_fused_grad_clip_scale(void* grad, void* sumSq, float maxNorm, int n);
void mtl_fused_commit_slot(int slot);
void mtl_fused_wait_slot(int slot);
void mtl_fused_dequant_delta(void* src, void* scales, void* delta, void* dst, int n, int cols);
void mtl_fused_dequant_delta_sparse(void* src, void* scales, void* delta, void* dst, void* mask, int n, int cols);
void mtl_fused_lm_head_pass1(void* hidden, void* embed, void* maxBuf, void* sumExp, int dim, int vocabSize, int n);
void mtl_fused_lm_head_pass2(void* hidden, void* embed, void* maxBuf, void* sumExp, void* targets, void* dHidden, void* loss, int dim, int vocabSize, int n);
void mtl_fused_gemm_tn_sparse(void* a, void* b, void* c, void* mask, int M, int K, int N);
void mtl_fused_end_async(void);
void mtl_fused_wait(void);
void mtl_fused_needle(void* data, void* scales, void* grad, void* mom, void* vel, void* mask, void* delta, float lr, float beta1, float beta2, float bc1, float bc2, float eps, float wd, int n, int cols, void* live, void* clipBuf);
void mtl_fused_needle_paired(void* d1, void* d2, void* s1, void* s2, void* g1, void* g2, void* m1, void* m2, void* v1, void* v2, void* mask, void* delta1, void* delta2, float lr, float beta1, float beta2, float bc1, float bc2, float eps, float wd, float backbone1, float glyco1, float hbond1, float hbond2, float glyco2, float backbone2, float bondStrength, int n, int cols, void* live1, void* live2, void* clipBuf);

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

// --- Inference Graph ---

func (m *Metal) BuildInferGraph(dim, kvDim, headDim, nHeads, nKVHeads, ffnDim, vocabSize, nLayers int, ropeTheta float64) int {
	return int(C.mtl_infer_build(C.int(dim), C.int(kvDim), C.int(headDim),
		C.int(nHeads), C.int(nKVHeads), C.int(ffnDim),
		C.int(vocabSize), C.int(nLayers), C.float(ropeTheta)))
}

func (m *Metal) InferNumWeights() int {
	return int(C.mtl_infer_num_weights())
}

func (m *Metal) InferSetWeight(idx int, data []float32) int {
	return int(C.mtl_infer_set_weight(C.int(idx), (*C.float)(unsafe.Pointer(&data[0])), C.int(len(data))))
}

func (m *Metal) InferForwardA(hidden []float32, cosSlice, sinSlice []float32, qOut, kOut, vOut []float32, layer int) int {
	return int(C.mtl_infer_forward(
		(*C.float)(unsafe.Pointer(&hidden[0])),
		(*C.float)(unsafe.Pointer(&cosSlice[0])),
		(*C.float)(unsafe.Pointer(&sinSlice[0])),
		(*C.float)(unsafe.Pointer(&qOut[0])),
		(*C.float)(unsafe.Pointer(&kOut[0])),
		(*C.float)(unsafe.Pointer(&vOut[0])),
		nil, nil, C.int(layer)))
}

func (m *Metal) InferForwardB(hidden []float32, attnOut []float32, layer int) int {
	return int(C.mtl_infer_forward_b(
		(*C.float)(unsafe.Pointer(&hidden[0])),
		(*C.float)(unsafe.Pointer(&attnOut[0])),
		C.int(layer)))
}

func (m *Metal) InferLogits(hidden []float32, logitsOut []float32) int {
	return int(C.mtl_infer_forward(
		(*C.float)(unsafe.Pointer(&hidden[0])),
		nil, nil, nil, nil, nil, nil,
		(*C.float)(unsafe.Pointer(&logitsOut[0])),
		C.int(10000))) // layer >= nLayers triggers final path
}

func (m *Metal) BuildFused(dim, kvDim, headDim, nHeads, nKVHeads, ffnDim, vocabSize, nLayers, maxSeq int) int {
	return int(C.mtl_fused_build(C.int(dim), C.int(kvDim), C.int(headDim),
		C.int(nHeads), C.int(nKVHeads), C.int(ffnDim),
		C.int(vocabSize), C.int(nLayers), C.int(maxSeq)))
}

func (m *Metal) FusedNumWeights() int {
	return int(C.mtl_fused_num_weights())
}

func (m *Metal) FusedSetWeight(idx int, data []float32) int {
	return int(C.mtl_fused_set_weight(C.int(idx), (*C.float)(unsafe.Pointer(&data[0])), C.int(len(data))))
}

func (m *Metal) FusedStep(hidden []float32, cosSlice, sinSlice []float32, pos int, logitsOut []float32) int {
	return int(C.mtl_fused_step(
		(*C.float)(unsafe.Pointer(&hidden[0])),
		(*C.float)(unsafe.Pointer(&cosSlice[0])),
		(*C.float)(unsafe.Pointer(&sinSlice[0])),
		C.int(pos),
		(*C.float)(unsafe.Pointer(&logitsOut[0]))))
}

func (m *Metal) BuildFusedInfer(dim, kvDim, headDim, nHeads, nKVHeads, ffnDim, vocabSize, nLayers, maxSeq int, ropeTheta float64) int {
	return int(C.mtl_fused_infer_build(C.int(dim), C.int(kvDim), C.int(headDim),
		C.int(nHeads), C.int(nKVHeads), C.int(ffnDim),
		C.int(vocabSize), C.int(nLayers), C.int(maxSeq), C.float(ropeTheta)))
}

func (m *Metal) FusedInferNumWeights() int {
	return int(C.mtl_fused_infer_num_weights())
}

func (m *Metal) FusedInferSetWeight(idx int, data []float32) int {
	return int(C.mtl_fused_infer_set_weight(C.int(idx), (*C.float)(unsafe.Pointer(&data[0])), C.int(len(data))))
}

func (m *Metal) FusedInferStep(hidden []float32, cosSlice, sinSlice []float32, pos int, logitsOut []float32) int {
	return int(C.mtl_fused_infer_step(
		(*C.float)(unsafe.Pointer(&hidden[0])),
		(*C.float)(unsafe.Pointer(&cosSlice[0])),
		(*C.float)(unsafe.Pointer(&sinSlice[0])),
		C.int(pos),
		(*C.float)(unsafe.Pointer(&logitsOut[0]))))
}

func (m *Metal) FusedInferReset() int {
	return int(C.mtl_fused_infer_reset())
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

// === Fused Training Compute Kernels ===

func (m *Metal) FusedBegin()             { C.mtl_fused_begin() }
func (m *Metal) FusedEnd()               { C.mtl_fused_end() }
func (m *Metal) FusedBeginSlot(slot int)  { C.mtl_fused_begin_slot(C.int(slot)) }
func (m *Metal) FusedEndSlot(slot int)    { C.mtl_fused_end_slot(C.int(slot)) }
func (m *Metal) FusedSetSlot(slot int)    { C.mtl_fused_set_slot(C.int(slot)) }
func (m *Metal) FusedCommitSlot(slot int) { C.mtl_fused_commit_slot(C.int(slot)) }
func (m *Metal) FusedWaitSlot(slot int)   { C.mtl_fused_wait_slot(C.int(slot)) }
func (m *Metal) FusedSyncAll()           { C.mtl_fused_sync_all() }

func (m *Metal) FusedGemmBT(a, b, c *Tensor, M, K, N int) {
	C.mtl_fused_gemm_bt(MtlBufPtr(a), MtlBufPtr(b), MtlBufPtr(c), C.int(M), C.int(K), C.int(N))
}
func (m *Metal) FusedGemmNN(a, b, c *Tensor, M, K, N int) {
	C.mtl_fused_gemm_nn(MtlBufPtr(a), MtlBufPtr(b), MtlBufPtr(c), C.int(M), C.int(K), C.int(N))
}
func (m *Metal) FusedGemmTN(a, b, c *Tensor, M, K, N int) {
	C.mtl_fused_gemm_tn(MtlBufPtr(a), MtlBufPtr(b), MtlBufPtr(c), C.int(M), C.int(K), C.int(N))
}
func (m *Metal) FusedGemmF32BT(a, b, c *Tensor, M, K, N int) {
	C.mtl_fused_gemm_f32_bt(MtlBufPtr(a), MtlBufPtr(b), MtlBufPtr(c), C.int(M), C.int(K), C.int(N))
}
func (m *Metal) FusedGemmF32NN(a, b, c *Tensor, M, K, N int) {
	C.mtl_fused_gemm_f32_nn(MtlBufPtr(a), MtlBufPtr(b), MtlBufPtr(c), C.int(M), C.int(K), C.int(N))
}
func (m *Metal) FusedGemmF32TN(a, b, c *Tensor, M, K, N int) {
	C.mtl_fused_gemm_f32_tn(MtlBufPtr(a), MtlBufPtr(b), MtlBufPtr(c), C.int(M), C.int(K), C.int(N))
}
func (m *Metal) FusedRMSNorm(x, w, scale *Tensor, seqLen, dim int) {
	C.mtl_fused_rmsnorm(MtlBufPtr(x), MtlBufPtr(w), MtlBufPtr(scale), C.int(seqLen), C.int(dim))
}
func (m *Metal) FusedRMSNormBwd(dOut, xIn, w, scale, dx *Tensor, seqLen, dim int) {
	C.mtl_fused_rmsnorm_bwd(MtlBufPtr(dOut), MtlBufPtr(xIn), MtlBufPtr(w), MtlBufPtr(scale), MtlBufPtr(dx), C.int(seqLen), C.int(dim))
}
func (m *Metal) FusedRoPE(x *Tensor, headDim, nHeads int, theta float32, stride, seqLen int) {
	C.mtl_fused_rope(MtlBufPtr(x), C.int(headDim), C.int(nHeads), C.float(theta), C.int(stride), C.int(seqLen))
}
func (m *Metal) FusedAttention(q, k, v, out, scores *Tensor, dim, kvDim, headDim, nHeads, nKVHeads, seqLen int) {
	C.mtl_fused_attn(MtlBufPtr(q), MtlBufPtr(k), MtlBufPtr(v), MtlBufPtr(out), MtlBufPtr(scores),
		C.int(dim), C.int(kvDim), C.int(headDim), C.int(nHeads), C.int(nKVHeads), C.int(seqLen))
}
func (m *Metal) FusedAttentionBwdQ(dOut, q, k, v, scores, dQ, dK, dV *Tensor,
	dim, kvDim, headDim, nHeads, nKVHeads, seqLen, qLen int) {
	C.mtl_fused_attention_bwd_q(MtlBufPtr(dOut), MtlBufPtr(q), MtlBufPtr(k), MtlBufPtr(v), MtlBufPtr(scores),
		MtlBufPtr(dQ), MtlBufPtr(dK), MtlBufPtr(dV),
		C.int(dim), C.int(kvDim), C.int(headDim), C.int(nHeads), C.int(nKVHeads), C.int(seqLen), C.int(qLen))
}
func (m *Metal) FusedSiLUGateMul(gate, up, out *Tensor, n int) {
	C.mtl_fused_silu_gate_mul(MtlBufPtr(gate), MtlBufPtr(up), MtlBufPtr(out), C.int(n))
}
func (m *Metal) SiLUGateBackward(dOut, gatePre, upOut, gateAct, dGatePre, dUp *Tensor) {
	C.mtl_silu_gate_backward_gpu(MtlBufPtr(dOut), MtlBufPtr(gatePre), MtlBufPtr(upOut),
		MtlBufPtr(gateAct), MtlBufPtr(dGatePre), MtlBufPtr(dUp), C.int(dOut.Size))
}
func (m *Metal) FusedAddInPlace(a, b *Tensor, n int) {
	C.mtl_fused_add_inplace(MtlBufPtr(a), MtlBufPtr(b), C.int(n))
}
func (m *Metal) FusedCopy(dst, src *Tensor, n int) {
	C.mtl_fused_copy(MtlBufPtr(dst), MtlBufPtr(src), C.int(n))
}
func (m *Metal) CELoss(logits, targets, losses *Tensor, seqLen, vocabSize int) {
	C.mtl_ce_loss(MtlBufPtr(logits), MtlBufPtr(targets), MtlBufPtr(losses), C.int(seqLen), C.int(vocabSize))
}
func (m *Metal) AdamWT(param, grad, mState, vState *Tensor, lr, wd float32, step int) {
	bc1 := C.float(1.0 - math.Pow(0.9, float64(step)))
	bc2 := C.float(1.0 - math.Pow(0.95, float64(step)))
	C.mtl_adamw_gpu(MtlBufPtr(param), MtlBufPtr(grad), MtlBufPtr(mState), MtlBufPtr(vState),
		C.float(lr), C.float(0.9), C.float(0.95), bc1, bc2, C.float(1e-8), C.float(wd), C.int(param.Size))
}
func (m *Metal) DNARungGPU(d1, g1, m1, v1, d2, g2, m2, v2 *Tensor,
	bb1, gly1, hb1, hb2, gly2, bb2, bondStr, lr, beta1, beta2, bc1, bc2, eps, wd float32, n int) {
	C.mtl_dna_rung_gpu(MtlBufPtr(d1), MtlBufPtr(g1), MtlBufPtr(m1), MtlBufPtr(v1),
		MtlBufPtr(d2), MtlBufPtr(g2), MtlBufPtr(m2), MtlBufPtr(v2),
		C.float(bb1), C.float(gly1), C.float(hb1), C.float(hb2), C.float(gly2), C.float(bb2),
		C.float(bondStr), C.float(lr), C.float(beta1), C.float(beta2),
		C.float(bc1), C.float(bc2), C.float(eps), C.float(wd), C.int(n))
}
func (m *Metal) GradNormSqGPU(grad, out *Tensor, n int) {
	C.mtl_grad_norm_sq(MtlBufPtr(grad), MtlBufPtr(out), C.int(n))
}

// WarmCache is a single MTLBuffer in unified memory that holds all optimizer state
// (momentum + velocity for every parameter). Both CPU and GPU access the same physical
// pages — the CPU (helix) computes rung geometry and reads/writes m/v via []float32
// slices; the GPU kernel reads/writes the same m/v via buffer offsets. No copy.
type WarmCache struct {
	buf      C.MTLBufferRef
	nFloats  int
	sharedF  []float32
}

// NewWarmCache allocates a single unified-memory buffer large enough to hold
// nFloats float32 values. Returns a WarmCache whose Slice method yields
// CPU-visible []float32 windows that the GPU also reads/writes.
func (m *Metal) NewWarmCache(nFloats int) *WarmCache {
	buf := C.mtl_alloc(C.size_t(nFloats * 4))
	C.mtl_zero(buf, C.size_t(nFloats*4))

	ptr := C.mtl_shared_ptr(buf)
	shared := (*[1 << 30]float32)(ptr)[:nFloats:nFloats]

	return &WarmCache{
		buf:     buf,
		nFloats: nFloats,
		sharedF: shared,
	}
}

// Slice returns a CPU-visible []float32 view into the warm cache starting at
// float offset 'off' with length 'n'. This slice is backed by the MTLBuffer's
// shared memory — writes from Go are visible to the GPU and vice versa.
func (wc *WarmCache) Slice(off, n int) []float32 {
	return wc.sharedF[off : off+n]
}

// ByteOffset returns the byte offset for a given float index, for passing
// to the GPU kernel dispatch.
func (wc *WarmCache) ByteOffset(floatIdx int) int {
	return floatIdx * 4
}

// BufPtr returns the raw MTLBuffer pointer for use with kernel dispatch.
func (wc *WarmCache) BufPtr() unsafe.Pointer {
	return unsafe.Pointer(wc.buf)
}

// Release frees the underlying MTLBuffer.
func (wc *WarmCache) Release() {
	if wc.buf != nil {
		C.mtl_free(wc.buf)
		wc.buf = nil
		wc.sharedF = nil
	}
}

// SharedSlice returns a CPU-visible []float32 view of a Tensor's underlying
// MTLBuffer. On Apple Silicon this IS the GPU memory — unified architecture.
// The returned slice is valid as long as the Tensor is not released.
func (m *Metal) SharedSlice(t *Tensor) []float32 {
	mp := t.device.(*mtlPtr)
	ptr := C.mtl_shared_ptr(mp.buf)
	return (*[1 << 30]float32)(ptr)[:t.Size:t.Size]
}

// DNARungWarm dispatches the paired DNA rung kernel with m/v read from a warm cache
// at the given float offsets. No separate m/v tensor allocations needed.
func (m *Metal) DNARungWarm(d1, g1, d2, g2 *Tensor, wc *WarmCache,
	m1Off, v1Off, m2Off, v2Off int,
	bb1, gly1, hb1, hb2, gly2, bb2, bondStr, lr, beta1, beta2, bc1, bc2, eps, wd float32, n int) {
	C.mtl_dna_rung_warm(MtlBufPtr(d1), MtlBufPtr(g1), MtlBufPtr(d2), MtlBufPtr(g2),
		wc.BufPtr(),
		C.int(wc.ByteOffset(m1Off)), C.int(wc.ByteOffset(v1Off)),
		C.int(wc.ByteOffset(m2Off)), C.int(wc.ByteOffset(v2Off)),
		C.float(bb1), C.float(gly1), C.float(hb1), C.float(hb2), C.float(gly2), C.float(bb2),
		C.float(bondStr), C.float(lr), C.float(beta1), C.float(beta2),
		C.float(bc1), C.float(bc2), C.float(eps), C.float(wd), C.int(n))
}

// AdamWWarm dispatches the AdamW kernel with m/v read from a warm cache at the given
// float offsets. Single-strand update for unpaired parameters.
func (m *Metal) AdamWWarm(param, grad *Tensor, wc *WarmCache, mOff, vOff int,
	lr, beta1, beta2, bc1, bc2, eps, wd float32, n int) {
	C.mtl_adamw_warm(MtlBufPtr(param), MtlBufPtr(grad),
		wc.BufPtr(),
		C.int(wc.ByteOffset(mOff)), C.int(wc.ByteOffset(vOff)),
		C.float(lr), C.float(beta1), C.float(beta2),
		C.float(bc1), C.float(bc2), C.float(eps), C.float(wd), C.int(n))
}

func (m *Metal) FusedZeroScalar(buf *Tensor) { C.mtl_fused_zero_scalar(MtlBufPtr(buf)) }
func (m *Metal) FusedBarrierBuffers()        { C.mtl_fused_barrier_buffers() }
func (m *Metal) FusedEndAsync()              { C.mtl_fused_end_async() }
func (m *Metal) FusedWait()                  { C.mtl_fused_wait() }

func (m *Metal) FusedGradNormSq(grad, sumSq *Tensor, n int) {
	C.mtl_fused_grad_norm_sq(MtlBufPtr(grad), MtlBufPtr(sumSq), C.int(n))
}

func (m *Metal) ICBExecuteFwd()  { C.mtl_icb_execute_fwd() }
func (m *Metal) ICBExecuteFull() { C.mtl_icb_execute_full() }

type ICBLayerActs struct {
	XIn, Normed, Q, K, V, AttnOut           *Tensor
	XMid, Normed2, GatePre, UpOut, FfnMid   *Tensor
	RmsScale1, RmsScale2, GateAct           *Tensor
}

type ICBLayerInt8 struct {
	Data, Scales, Delta, Mom, Vel, Live *Tensor
	Mask                                *HotRowMask
}

type ICBLayerBwd struct {
	DFfnMid, DGate, DUp, DN2, Dx        *Tensor
	DAttnOut, DQ, DK, DV, DN1           *Tensor
	DWDown, DWGate, DWUp, DWO, DWQ, DWK, DWV *Tensor
}

type ICBLayerWeights struct {
	WQ, WK, WV, WO, Gate, Up, Down ICBLayerInt8
	Norm1, Norm2                    *Tensor
}

func (m *Metal) ICBBuildTraining(
	dim, kvDim, headDim, nHeads, nKVHeads, ffnDim, vocabSize, seqLen, nLayers int,
	hidden, normedFinal, finalNorm, finalScales *Tensor,
	lmMaxLogit, lmSumExp, lmLoss, targetsGPU *Tensor,
	dHidden, dScratch, dEmbed *Tensor,
	gradSumSq, clipScaleBuf, scores *Tensor,
	embed *Tensor,
	embedInt8 ICBLayerInt8,
	acts []ICBLayerActs,
	weights []ICBLayerWeights,
	bwds []ICBLayerBwd,
	lrBuf, bc1Buf, bc2Buf, maxNormBuf *Tensor,
	bb1Buf, gly1Buf, hb1Buf, hb2Buf, gly2Buf, bb2Buf, bondStrBuf *Tensor,
) int {
	nL := nLayers
	mkArr := func(tensors []*Tensor) *unsafe.Pointer {
		arr := (*[64]unsafe.Pointer)(C.malloc(C.size_t(len(tensors) * 8)))
		for i, t := range tensors { arr[i] = MtlBufPtr(t) }
		return &arr[0]
	}
	mkMaskArr := func(masks []*HotRowMask) *unsafe.Pointer {
		arr := (*[64]unsafe.Pointer)(C.malloc(C.size_t(len(masks) * 8)))
		for i, m := range masks { arr[i] = m.BufPtr() }
		return &arr[0]
	}

	// Collect per-layer arrays
	norm1s := make([]*Tensor, nL); norm2s := make([]*Tensor, nL)
	aXIn := make([]*Tensor, nL); aNormed := make([]*Tensor, nL)
	aQ := make([]*Tensor, nL); aK := make([]*Tensor, nL); aV := make([]*Tensor, nL)
	aAttnOut := make([]*Tensor, nL); aXMid := make([]*Tensor, nL)
	aNormed2 := make([]*Tensor, nL); aGatePre := make([]*Tensor, nL)
	aUpOut := make([]*Tensor, nL); aFfnMid := make([]*Tensor, nL)
	aRmsScale1 := make([]*Tensor, nL); aRmsScale2 := make([]*Tensor, nL)
	aGateAct := make([]*Tensor, nL)

	wqD := make([]*Tensor, nL); wqS := make([]*Tensor, nL); wqDl := make([]*Tensor, nL)
	wqM := make([]*Tensor, nL); wqV := make([]*Tensor, nL); wqL := make([]*Tensor, nL); wqMk := make([]*HotRowMask, nL)
	wkD := make([]*Tensor, nL); wkS := make([]*Tensor, nL); wkDl := make([]*Tensor, nL)
	wkM := make([]*Tensor, nL); wkV := make([]*Tensor, nL); wkL := make([]*Tensor, nL); wkMk := make([]*HotRowMask, nL)
	wvD := make([]*Tensor, nL); wvS := make([]*Tensor, nL); wvDl := make([]*Tensor, nL)
	wvM := make([]*Tensor, nL); wvV := make([]*Tensor, nL); wvL := make([]*Tensor, nL); wvMk := make([]*HotRowMask, nL)
	woD := make([]*Tensor, nL); woS := make([]*Tensor, nL); woDl := make([]*Tensor, nL)
	woM := make([]*Tensor, nL); woV := make([]*Tensor, nL); woL := make([]*Tensor, nL); woMk := make([]*HotRowMask, nL)
	gD := make([]*Tensor, nL); gS := make([]*Tensor, nL); gDl := make([]*Tensor, nL)
	gM := make([]*Tensor, nL); gV := make([]*Tensor, nL); gL := make([]*Tensor, nL); gMk := make([]*HotRowMask, nL)
	uD := make([]*Tensor, nL); uS := make([]*Tensor, nL); uDl := make([]*Tensor, nL)
	uM := make([]*Tensor, nL); uV := make([]*Tensor, nL); uL := make([]*Tensor, nL); uMk := make([]*HotRowMask, nL)
	dD := make([]*Tensor, nL); dS := make([]*Tensor, nL); dDl := make([]*Tensor, nL)
	dM := make([]*Tensor, nL); dV2 := make([]*Tensor, nL); dL := make([]*Tensor, nL); dMk := make([]*HotRowMask, nL)

	bDFfn := make([]*Tensor, nL); bDGate := make([]*Tensor, nL); bDUp := make([]*Tensor, nL)
	bDN2 := make([]*Tensor, nL); bDx := make([]*Tensor, nL); bDAttn := make([]*Tensor, nL)
	bDQ := make([]*Tensor, nL); bDK := make([]*Tensor, nL); bDV := make([]*Tensor, nL)
	bDN1 := make([]*Tensor, nL); bDWD := make([]*Tensor, nL); bDWG := make([]*Tensor, nL)
	bDWU := make([]*Tensor, nL); bDWO := make([]*Tensor, nL); bDWQ := make([]*Tensor, nL)
	bDWK := make([]*Tensor, nL); bDWV := make([]*Tensor, nL)

	for i := 0; i < nL; i++ {
		norm1s[i] = weights[i].Norm1; norm2s[i] = weights[i].Norm2
		a := acts[i]
		aXIn[i] = a.XIn; aNormed[i] = a.Normed; aQ[i] = a.Q; aK[i] = a.K; aV[i] = a.V
		aAttnOut[i] = a.AttnOut; aXMid[i] = a.XMid; aNormed2[i] = a.Normed2
		aGatePre[i] = a.GatePre; aUpOut[i] = a.UpOut; aFfnMid[i] = a.FfnMid
		aRmsScale1[i] = a.RmsScale1; aRmsScale2[i] = a.RmsScale2; aGateAct[i] = a.GateAct

		w := weights[i]
		wqD[i]=w.WQ.Data; wqS[i]=w.WQ.Scales; wqDl[i]=w.WQ.Delta; wqM[i]=w.WQ.Mom; wqV[i]=w.WQ.Vel; wqL[i]=w.WQ.Live; wqMk[i]=w.WQ.Mask
		wkD[i]=w.WK.Data; wkS[i]=w.WK.Scales; wkDl[i]=w.WK.Delta; wkM[i]=w.WK.Mom; wkV[i]=w.WK.Vel; wkL[i]=w.WK.Live; wkMk[i]=w.WK.Mask
		wvD[i]=w.WV.Data; wvS[i]=w.WV.Scales; wvDl[i]=w.WV.Delta; wvM[i]=w.WV.Mom; wvV[i]=w.WV.Vel; wvL[i]=w.WV.Live; wvMk[i]=w.WV.Mask
		woD[i]=w.WO.Data; woS[i]=w.WO.Scales; woDl[i]=w.WO.Delta; woM[i]=w.WO.Mom; woV[i]=w.WO.Vel; woL[i]=w.WO.Live; woMk[i]=w.WO.Mask
		gD[i]=w.Gate.Data; gS[i]=w.Gate.Scales; gDl[i]=w.Gate.Delta; gM[i]=w.Gate.Mom; gV[i]=w.Gate.Vel; gL[i]=w.Gate.Live; gMk[i]=w.Gate.Mask
		uD[i]=w.Up.Data; uS[i]=w.Up.Scales; uDl[i]=w.Up.Delta; uM[i]=w.Up.Mom; uV[i]=w.Up.Vel; uL[i]=w.Up.Live; uMk[i]=w.Up.Mask
		dD[i]=w.Down.Data; dS[i]=w.Down.Scales; dDl[i]=w.Down.Delta; dM[i]=w.Down.Mom; dV2[i]=w.Down.Vel; dL[i]=w.Down.Live; dMk[i]=w.Down.Mask

		b := bwds[i]
		bDFfn[i]=b.DFfnMid; bDGate[i]=b.DGate; bDUp[i]=b.DUp; bDN2[i]=b.DN2; bDx[i]=b.Dx
		bDAttn[i]=b.DAttnOut; bDQ[i]=b.DQ; bDK[i]=b.DK; bDV[i]=b.DV; bDN1[i]=b.DN1
		bDWD[i]=b.DWDown; bDWG[i]=b.DWGate; bDWU[i]=b.DWUp; bDWO[i]=b.DWO
		bDWQ[i]=b.DWQ; bDWK[i]=b.DWK; bDWV[i]=b.DWV
	}

	p := C.ICBBuildParams{
		dim: C.int(dim), kvDim: C.int(kvDim), headDim: C.int(headDim),
		nHeads: C.int(nHeads), nKVHeads: C.int(nKVHeads), ffnDim: C.int(ffnDim),
		vocabSize: C.int(vocabSize), seqLen: C.int(seqLen), nLayers: C.int(nLayers),
		hidden: MtlBufPtr(hidden), normedFinal: MtlBufPtr(normedFinal),
		finalNorm: MtlBufPtr(finalNorm), finalScales: MtlBufPtr(finalScales),
		lmMaxLogit: MtlBufPtr(lmMaxLogit), lmSumExp: MtlBufPtr(lmSumExp),
		lmLoss: MtlBufPtr(lmLoss), targetsGPU: MtlBufPtr(targetsGPU),
		dHidden: MtlBufPtr(dHidden), dScratch: MtlBufPtr(dScratch), dEmbed: MtlBufPtr(dEmbed),
		gradSumSq: MtlBufPtr(gradSumSq), clipScaleBuf: MtlBufPtr(clipScaleBuf), scores: MtlBufPtr(scores),
		embed: MtlBufPtr(embed),
		embedData: MtlBufPtr(embedInt8.Data), embedScales: MtlBufPtr(embedInt8.Scales),
		embedDelta: MtlBufPtr(embedInt8.Delta), embedMom: MtlBufPtr(embedInt8.Mom),
		embedVel: MtlBufPtr(embedInt8.Vel), embedMask: embedInt8.Mask.BufPtr(),
		embedLive: MtlBufPtr(embedInt8.Live),
		norm1: mkArr(norm1s), norm2: mkArr(norm2s),
		a_xIn: mkArr(aXIn), a_normed: mkArr(aNormed), a_Q: mkArr(aQ), a_K: mkArr(aK),
		a_V: mkArr(aV), a_attnOut: mkArr(aAttnOut), a_xMid: mkArr(aXMid),
		a_normed2: mkArr(aNormed2), a_gatePre: mkArr(aGatePre), a_upOut: mkArr(aUpOut),
		a_ffnMid: mkArr(aFfnMid), a_rmsScale1: mkArr(aRmsScale1), a_rmsScale2: mkArr(aRmsScale2),
		a_gateAct: mkArr(aGateAct),
		wq_data: mkArr(wqD), wq_scales: mkArr(wqS), wq_delta: mkArr(wqDl),
		wq_mom: mkArr(wqM), wq_vel: mkArr(wqV), wq_live: mkArr(wqL), wq_mask: mkMaskArr(wqMk),
		wk_data: mkArr(wkD), wk_scales: mkArr(wkS), wk_delta: mkArr(wkDl),
		wk_mom: mkArr(wkM), wk_vel: mkArr(wkV), wk_live: mkArr(wkL), wk_mask: mkMaskArr(wkMk),
		wv_data: mkArr(wvD), wv_scales: mkArr(wvS), wv_delta: mkArr(wvDl),
		wv_mom: mkArr(wvM), wv_vel: mkArr(wvV), wv_live: mkArr(wvL), wv_mask: mkMaskArr(wvMk),
		wo_data: mkArr(woD), wo_scales: mkArr(woS), wo_delta: mkArr(woDl),
		wo_mom: mkArr(woM), wo_vel: mkArr(woV), wo_live: mkArr(woL), wo_mask: mkMaskArr(woMk),
		gate_data: mkArr(gD), gate_scales: mkArr(gS), gate_delta: mkArr(gDl),
		gate_mom: mkArr(gM), gate_vel: mkArr(gV), gate_live: mkArr(gL), gate_mask: mkMaskArr(gMk),
		up_data: mkArr(uD), up_scales: mkArr(uS), up_delta: mkArr(uDl),
		up_mom: mkArr(uM), up_vel: mkArr(uV), up_live: mkArr(uL), up_mask: mkMaskArr(uMk),
		down_data: mkArr(dD), down_scales: mkArr(dS), down_delta: mkArr(dDl),
		down_mom: mkArr(dM), down_vel: mkArr(dV2), down_live: mkArr(dL), down_mask: mkMaskArr(dMk),
		b_dFfnMid: mkArr(bDFfn), b_dGate: mkArr(bDGate), b_dUp: mkArr(bDUp),
		b_dN2: mkArr(bDN2), b_dx: mkArr(bDx), b_dAttnOut: mkArr(bDAttn),
		b_dQ: mkArr(bDQ), b_dK: mkArr(bDK), b_dV: mkArr(bDV), b_dN1: mkArr(bDN1),
		b_dWDown: mkArr(bDWD), b_dWGate: mkArr(bDWG), b_dWUp: mkArr(bDWU),
		b_dWO: mkArr(bDWO), b_dWQ: mkArr(bDWQ), b_dWK: mkArr(bDWK), b_dWV: mkArr(bDWV),
		lrBuf: MtlBufPtr(lrBuf), bc1Buf: MtlBufPtr(bc1Buf), bc2Buf: MtlBufPtr(bc2Buf),
		maxNormBuf: MtlBufPtr(maxNormBuf),
		bb1Buf: MtlBufPtr(bb1Buf), gly1Buf: MtlBufPtr(gly1Buf), hb1Buf: MtlBufPtr(hb1Buf),
		hb2Buf: MtlBufPtr(hb2Buf), gly2Buf: MtlBufPtr(gly2Buf), bb2Buf: MtlBufPtr(bb2Buf),
		bondStrBuf: MtlBufPtr(bondStrBuf),
	}
	return int(C.mtl_icb_build_training(&p))
}

func (m *Metal) FusedComputeClipScale(sumSq, clipScale *Tensor, maxNorm float32) {
	C.mtl_fused_compute_clip_scale(MtlBufPtr(sumSq), MtlBufPtr(clipScale), C.float(maxNorm))
}

func (m *Metal) FusedGradClipScale(grad, sumSq *Tensor, maxNorm float32, n int) {
	C.mtl_fused_grad_clip_scale(MtlBufPtr(grad), MtlBufPtr(sumSq), C.float(maxNorm), C.int(n))
}

func (m *Metal) FusedDequantDelta(src, scales, delta, dst *Tensor, n, cols int) {
	C.mtl_fused_dequant_delta(MtlBufPtr(src), MtlBufPtr(scales), MtlBufPtr(delta), MtlBufPtr(dst), C.int(n), C.int(cols))
}

func (m *Metal) FusedLMHeadPass1(hidden, embed, maxBuf, sumExp *Tensor, dim, vocabSize, n int) {
	C.mtl_fused_lm_head_pass1(MtlBufPtr(hidden), MtlBufPtr(embed), MtlBufPtr(maxBuf), MtlBufPtr(sumExp),
		C.int(dim), C.int(vocabSize), C.int(n))
}

func (m *Metal) FusedLMHeadPass2(hidden, embed, maxBuf, sumExp, targets, dHidden, loss *Tensor, dim, vocabSize, n int) {
	C.mtl_fused_lm_head_pass2(MtlBufPtr(hidden), MtlBufPtr(embed), MtlBufPtr(maxBuf), MtlBufPtr(sumExp),
		MtlBufPtr(targets), MtlBufPtr(dHidden), MtlBufPtr(loss),
		C.int(dim), C.int(vocabSize), C.int(n))
}

func (m *Metal) FusedGemmF32TNSparse(a, b, c *Tensor, mask *HotRowMask, M, K, N int) {
	C.mtl_fused_gemm_tn_sparse(MtlBufPtr(a), MtlBufPtr(b), MtlBufPtr(c), mask.BufPtr(), C.int(M), C.int(K), C.int(N))
}

func (m *Metal) FusedDequantDeltaSparse(src, scales, delta, dst *Tensor, mask *HotRowMask, n, cols int) {
	C.mtl_fused_dequant_delta_sparse(MtlBufPtr(src), MtlBufPtr(scales), MtlBufPtr(delta), MtlBufPtr(dst), mask.BufPtr(), C.int(n), C.int(cols))
}

func (m *Metal) FusedNeedle(data, scales, grad, mom, vel *Tensor, mask *HotRowMask, delta *Tensor,
	lr, beta1, beta2, bc1, bc2, eps, wd float32, n, cols int, live, clipBuf *Tensor) {
	C.mtl_fused_needle(MtlBufPtr(data), MtlBufPtr(scales), MtlBufPtr(grad),
		MtlBufPtr(mom), MtlBufPtr(vel), mask.BufPtr(), MtlBufPtr(delta),
		C.float(lr), C.float(beta1), C.float(beta2),
		C.float(bc1), C.float(bc2), C.float(eps), C.float(wd),
		C.int(n), C.int(cols), MtlBufPtr(live), MtlBufPtr(clipBuf))
}

func (m *Metal) FusedNeedlePaired(d1, d2, s1, s2, g1, g2, m1, m2, v1, v2 *Tensor, mask *HotRowMask, delta1, delta2 *Tensor,
	lr, beta1, beta2, bc1, bc2, eps, wd,
	backbone1, glyco1, hbond1, hbond2, glyco2, backbone2, bondStrength float32,
	n, cols int, live1, live2, clipBuf *Tensor) {
	C.mtl_fused_needle_paired(MtlBufPtr(d1), MtlBufPtr(d2), MtlBufPtr(s1), MtlBufPtr(s2),
		MtlBufPtr(g1), MtlBufPtr(g2), MtlBufPtr(m1), MtlBufPtr(m2),
		MtlBufPtr(v1), MtlBufPtr(v2), mask.BufPtr(), MtlBufPtr(delta1), MtlBufPtr(delta2),
		C.float(lr), C.float(beta1), C.float(beta2),
		C.float(bc1), C.float(bc2), C.float(eps), C.float(wd),
		C.float(backbone1), C.float(glyco1), C.float(hbond1),
		C.float(hbond2), C.float(glyco2), C.float(backbone2),
		C.float(bondStrength), C.int(n), C.int(cols),
		MtlBufPtr(live1), MtlBufPtr(live2), MtlBufPtr(clipBuf))
}

type HotRowMask struct {
	buf    C.MTLBufferRef
	nRows  int
	shared []int8
}

func (m *Metal) NewHotRowMask(nRows int) *HotRowMask {
	buf := C.mtl_alloc(C.size_t(nRows))
	C.mtl_zero(buf, C.size_t(nRows))
	ptr := C.mtl_shared_ptr(buf)
	shared := (*[1 << 30]int8)(ptr)[:nRows:nRows]
	return &HotRowMask{buf: buf, nRows: nRows, shared: shared}
}

func (h *HotRowMask) Set(hotRows []int32) {
	for i := range h.shared {
		h.shared[i] = 0
	}
	for _, r := range hotRows {
		if int(r) >= 0 && int(r) < h.nRows {
			h.shared[r] = 1
		}
	}
}

func (h *HotRowMask) BufPtr() unsafe.Pointer { return unsafe.Pointer(h.buf) }

func (h *HotRowMask) Release() {
	if h.buf != nil {
		C.mtl_free(h.buf)
		h.buf = nil
		h.shared = nil
	}
}
