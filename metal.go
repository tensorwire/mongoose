
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

typedef struct {
    float embeddingMultiplier;
    float residualMultiplier;
    float attentionScale;
    float logitsScaling;
    float adamBeta2;
} MongooseArchParams;

int mtl_graph_build_full_arch(int dim, int kvDim, int headDim,
                              int nHeads, int nKVHeads, int ffnDim,
                              int vocabSize, int nLayers, int seqLen,
                              float ropeTheta, int mode,
                              MongooseArchParams arch);
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
                    int vocabSize, int nLayers, int maxSeq,
                    float ropeTheta, float rmsEps);
int mtl_fused_set_arch(float embeddingMultiplier, float residualMultiplier,
                       float attentionScale, float logitsScaling);
int mtl_fused_num_weights(void);
int mtl_fused_set_weight(int idx, const float* data, int nFloats);
int mtl_fused_step(const float* hiddenIn, const float* cosData, const float* sinData,
                   int pos, float* logitsOut);
void mtl_fused_reset_kv(void);
void mtl_fused_reset_kv_slot(int slot);
int mtl_fused_num_slots(void);
int mtl_fused_partial_step(const float* hiddenIn, int pos,
                           int layerStart, int layerEnd,
                           float* hiddenOut, float* logitsOut);
int mtl_fused_partial_step_slot(int slot, const float* hiddenIn, int pos,
                                int layerStart, int layerEnd,
                                float* hiddenOut, float* logitsOut);
int mtl_fused_prefill_tile(void);
int mtl_fused_prefill_batch(int slot, const float* hiddenIn, int B, int basePos,
                            float* logitsOut);
// Streaming inference — ping-pong weight buffers, per-layer dispatch
int mtl_stream_build(void);
void mtl_stream_upload_layer(int set, int layer,
    const float* norm1, const float* wq, const float* wk, const float* wv,
    const float* bq, const float* bk, const float* bv, const float* wo,
    const float* norm2, const float* gate, const float* up, const float* down);
int mtl_stream_step_layer(int set, int layer, int pos);
int mtl_stream_step_final(int pos, float* logitsOut);
void mtl_stream_set_hidden(const float* hiddenIn);

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
void mtl_fused_pre_attn(void* hidden, void* normW, void* wq, void* wk, void* wv, void* Q, void* K, void* V, void* normedOut, void* rmsScale, void* xIn, int dim, int kvDim, int headDim, int nHeads, int nKVHeads, int ffnDim, int seqLen, float ropeTheta, float eps);
void mtl_fused_post_attn(void* hidden, void* attnOut, void* wo, void* normW2, void* gate, void* up, void* down, void* xMid, void* normed2, void* rmsScale2, void* gatePre, void* upOut, void* ffnMid, int dim, int kvDim, int headDim, int nHeads, int nKVHeads, int ffnDim, int seqLen, float ropeTheta, float eps);
int mtl_fused_train_available(void);
void mtl_fused_lm_head_pass1(void* hidden, void* embed, void* maxBuf, void* sumExp, int dim, int vocabSize, int n);
void mtl_fused_lm_head_pass2(void* hidden, void* embed, void* maxBuf, void* sumExp, void* targets, void* dHidden, void* loss, int dim, int vocabSize, int n);
void mtl_lm_head_forward_grad(void* hidden, void* embed, void* maxBuf, void* sumExp, void* targets, void* dHidden, void* loss, int dim, int vocabSize, int n);
void mtl_softmax_ce_grad(void* logits, void* targets, void* losses, void* grad, int seqLen, int vocabSize, float invN);
void mtl_fused_gemm_tn_sparse(void* a, void* b, void* c, void* mask, int M, int K, int N);
void mtl_fused_end_async(void);
void mtl_fused_wait(void);
void mtl_fused_needle(void* data, void* scales, void* grad, void* mom, void* vel, void* mask, void* delta, float lr, float beta1, float beta2, float bc1, float bc2, float eps, float wd, int n, int cols, void* live, void* clipBuf);
void mtl_fused_needle_paired(void* d1, void* d2, void* s1, void* s2, void* g1, void* g2, void* m1, void* m2, void* v1, void* v2, void* mask, void* delta1, void* delta2, float lr, float beta1, float beta2, float bc1, float bc2, float eps, float wd, float backbone1, float glyco1, float hbond1, float hbond2, float glyco2, float backbone2, float bondStrength, int n, int cols, void* live1, void* live2, void* clipBuf);

*/
import "C"
import _ "unsafe"

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
	ret := (C.mtl_init)()
	if ret != 0 {
		log.Printf("WARN mongoose => Metal init failed (code %d)", ret)
		return nil
	}

	name := (C.GoString)((C.mtl_device_name)())
	m := &Metal{deviceName: name, pool: make(map[int][]C.MTLBufferRef)}

	if (C.mtl_init_compute)() == 0 {
		log.Printf("[mongoose] Metal initialized: %s (compute kernels ready)", name)
	} else {
		log.Printf("[mongoose] Metal initialized: %s (compute kernels failed, CPU fallback)", name)
	}

	return m
}

func (m *Metal) Name() string { return fmt.Sprintf("metal/%s", m.deviceName) }
func (m *Metal) Close()       {}
func (m *Metal) BeginBatch()  { (C.mtl_begin_batch)() }
func (m *Metal) EndBatch()    { (C.mtl_end_batch)() }
func (m *Metal) Sync()        { (C.mtl_graph_sync)() }
func MtlComputeReady() bool   { return (C.mtl_compute_ready)() == 1 }

func (m *Metal) MatMul(a, b []float32, rows, k, n int) []float32 {
	bufA := m.poolGet(len(a))
	bufB := m.poolGet(len(b))
	bufC := m.poolGet(rows * n)

	func() {
		_cgo0 := bufA
		_cgoIndex1 := &a
		_cgo1 := unsafe.Pointer(&(*_cgoIndex1)[0])
		var _cgo2 C.size_t = C.size_t(len(a) * 4)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, *_cgoIndex1)
		C.mtl_upload(_cgo0, _cgo1, _cgo2)
	}()
	func() {
		_cgo0 := bufB
		_cgoIndex1 := &b
		_cgo1 := unsafe.Pointer(&(*_cgoIndex1)[0])
		var _cgo2 C.size_t = C.size_t(len(b) * 4)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, *_cgoIndex1)
		C.mtl_upload(_cgo0, _cgo1, _cgo2)
	}()
	func() C.int {
		_cgo0 := bufA
		_cgo1 := bufB
		_cgo2 := bufC
		var _cgo3 C.int = C.int(rows)
		var _cgo4 C.int = C.int(k)
		var _cgo5 C.int = C.int(n)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		return C.mtl_sgemm(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5)
	}()

	out := make([]float32, rows*n)
	func() {
		_cgoIndex0 := &out
		_cgo0 := unsafe.Pointer(&(*_cgoIndex0)[0])
		_cgo1 := bufC
		var _cgo2 C.size_t = C.size_t(rows * n * 4)
		_cgoCheckPointer(_cgo0, *_cgoIndex0)
		_cgoCheckPointer(_cgo1, nil)
		C.mtl_download(_cgo0, _cgo1, _cgo2)
	}()

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
	return uint64((C.mtl_recommended_max_working_set_size)())
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
	func() {
		_cgo0 := bufA
		_cgoIndex1 := &A
		_cgo1 := unsafe.Pointer(&(*_cgoIndex1)[0])
		var _cgo2 C.size_t = C.size_t(rows * k * 4)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, *_cgoIndex1)
		C.mtl_upload(_cgo0, _cgo1, _cgo2)
	}()
	func() {
		_cgo0 := bufB
		_cgoIndex1 := &B
		_cgo1 := unsafe.Pointer(&(*_cgoIndex1)[0])
		var _cgo2 C.size_t = C.size_t(n * k * 4)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, *_cgoIndex1)
		C.mtl_upload(_cgo0, _cgo1, _cgo2)
	}()
	func() C.int {
		_cgo0 := bufA
		_cgo1 := bufB
		_cgo2 := bufC
		var _cgo3 C.int = C.int(rows)
		var _cgo4 C.int = C.int(k)
		var _cgo5 C.int = C.int(n)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		return C.mtl_sgemm_transB(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5)
	}()
	func() {
		_cgoIndex0 := &out
		_cgo0 := unsafe.Pointer(&(*_cgoIndex0)[0])
		_cgo1 := bufC
		var _cgo2 C.size_t = C.size_t(rows * n * 4)
		_cgoCheckPointer(_cgo0, *_cgoIndex0)
		_cgoCheckPointer(_cgo1, nil)
		C.mtl_download(_cgo0, _cgo1, _cgo2)
	}()
	m.poolPut(rows*k, bufA)
	m.poolPut(n*k, bufB)
	m.poolPut(rows*n, bufC)
}

func (m *Metal) MatMulInto(out, A, B []float32, rows, k, n int) {
	bufA := m.poolGet(rows * k)
	bufB := m.poolGet(k * n)
	bufC := m.poolGet(rows * n)
	func() {
		_cgo0 := bufA
		_cgoIndex1 := &A
		_cgo1 := unsafe.Pointer(&(*_cgoIndex1)[0])
		var _cgo2 C.size_t = C.size_t(rows * k * 4)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, *_cgoIndex1)
		C.mtl_upload(_cgo0, _cgo1, _cgo2)
	}()
	func() {
		_cgo0 := bufB
		_cgoIndex1 := &B
		_cgo1 := unsafe.Pointer(&(*_cgoIndex1)[0])
		var _cgo2 C.size_t = C.size_t(k * n * 4)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, *_cgoIndex1)
		C.mtl_upload(_cgo0, _cgo1, _cgo2)
	}()
	func() C.int {
		_cgo0 := bufA
		_cgo1 := bufB
		_cgo2 := bufC
		var _cgo3 C.int = C.int(rows)
		var _cgo4 C.int = C.int(k)
		var _cgo5 C.int = C.int(n)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		return C.mtl_sgemm(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5)
	}()
	func() {
		_cgoIndex0 := &out
		_cgo0 := unsafe.Pointer(&(*_cgoIndex0)[0])
		_cgo1 := bufC
		var _cgo2 C.size_t = C.size_t(rows * n * 4)
		_cgoCheckPointer(_cgo0, *_cgoIndex0)
		_cgoCheckPointer(_cgo1, nil)
		C.mtl_download(_cgo0, _cgo1, _cgo2)
	}()
	m.poolPut(rows*k, bufA)
	m.poolPut(k*n, bufB)
	m.poolPut(rows*n, bufC)
}

func (m *Metal) MatMulAddInto(G, A, B []float32, rows, k, n int) {
	bufA := m.poolGet(rows * k)
	bufB := m.poolGet(k * n)
	bufC := m.poolGet(rows * n)
	func() {
		_cgo0 := bufA
		_cgoIndex1 := &A
		_cgo1 := unsafe.Pointer(&(*_cgoIndex1)[0])
		var _cgo2 C.size_t = C.size_t(rows * k * 4)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, *_cgoIndex1)
		C.mtl_upload(_cgo0, _cgo1, _cgo2)
	}()
	func() {
		_cgo0 := bufB
		_cgoIndex1 := &B
		_cgo1 := unsafe.Pointer(&(*_cgoIndex1)[0])
		var _cgo2 C.size_t = C.size_t(k * n * 4)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, *_cgoIndex1)
		C.mtl_upload(_cgo0, _cgo1, _cgo2)
	}()
	func() C.int {
		_cgo0 := bufA
		_cgo1 := bufB
		_cgo2 := bufC
		var _cgo3 C.int = C.int(rows)
		var _cgo4 C.int = C.int(k)
		var _cgo5 C.int = C.int(n)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		return C.mtl_sgemm_transA(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5)
	}()
	tmp := make([]float32, rows*n)
	func() {
		_cgoIndex0 := &tmp
		_cgo0 := unsafe.Pointer(&(*_cgoIndex0)[0])
		_cgo1 := bufC
		var _cgo2 C.size_t = C.size_t(rows * n * 4)
		_cgoCheckPointer(_cgo0, *_cgoIndex0)
		_cgoCheckPointer(_cgo1, nil)
		C.mtl_download(_cgo0, _cgo1, _cgo2)
	}()
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
	func() {
		_cgo0 := bufA
		_cgoIndex1 := &A
		_cgo1 := unsafe.Pointer(&(*_cgoIndex1)[0])
		var _cgo2 C.size_t = C.size_t(rows * k * 4)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, *_cgoIndex1)
		C.mtl_upload(_cgo0, _cgo1, _cgo2)
	}()
	func() {
		_cgo0 := bufB
		_cgoIndex1 := &B
		_cgo1 := unsafe.Pointer(&(*_cgoIndex1)[0])
		var _cgo2 C.size_t = C.size_t(k * n * 4)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, *_cgoIndex1)
		C.mtl_upload(_cgo0, _cgo1, _cgo2)
	}()
	func() C.int {
		_cgo0 := bufA
		_cgo1 := bufB
		_cgo2 := bufC
		var _cgo3 C.int = C.int(rows)
		var _cgo4 C.int = C.int(k)
		var _cgo5 C.int = C.int(n)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		return C.mtl_sgemm_transA(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5)
	}()
	func() {
		_cgoIndex0 := &out
		_cgo0 := unsafe.Pointer(&(*_cgoIndex0)[0])
		_cgo1 := bufC
		var _cgo2 C.size_t = C.size_t(rows * n * 4)
		_cgoCheckPointer(_cgo0, *_cgoIndex0)
		_cgoCheckPointer(_cgo1, nil)
		C.mtl_download(_cgo0, _cgo1, _cgo2)
	}()
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
	return int((C.mtl_graph_build_full)(
		C.int(dim), C.int(kvDim), C.int(headDim),
		C.int(nHeads), C.int(nKVHeads), C.int(ffnDim),
		C.int(vocabSize), C.int(nLayers), C.int(seqLen),
		C.float(ropeTheta), C.int(mode)))
}

// BuildFullGraphArch builds the training graph with architecture-specific
// scalars. Zero fields fall back to Llama defaults, so ArchParams{} is
// equivalent to BuildFullGraph.
//
// Granite requires all of these — see ArchParams. Its tensor layout is
// identical to Llama's, so a graph built without them trains a different
// function and yields plausible garbage instead of an error.
func (m *Metal) BuildFullGraphArch(dim, kvDim, headDim, nHeads, nKVHeads, ffnDim,
	vocabSize, nLayers, seqLen int, ropeTheta float64, mode int, arch ArchParams) int {

	var c C.MongooseArchParams
	c.embeddingMultiplier = C.float(arch.EmbeddingMultiplier)
	c.residualMultiplier = C.float(arch.ResidualMultiplier)
	c.attentionScale = C.float(arch.AttentionScale)
	c.logitsScaling = C.float(arch.LogitsScaling)
	c.adamBeta2 = C.float(arch.AdamBeta2)

	return int((C.mtl_graph_build_full_arch)(
		C.int(dim), C.int(kvDim), C.int(headDim),
		C.int(nHeads), C.int(nKVHeads), C.int(ffnDim),
		C.int(vocabSize), C.int(nLayers), C.int(seqLen),
		C.float(ropeTheta), C.int(mode), c))
}

func (m *Metal) GraphTrainStepAdam(tokens, targets []int32, lr float32) float32 {
	return float32((C.mtl_graph_train_step)(
		(*C.int)(unsafe.Pointer(&tokens[0])),
		(*C.int)(unsafe.Pointer(&targets[0])),
		C.int(len(tokens)),
		nil, nil, 0, C.float(lr), 1))
}

func (m *Metal) GraphFullBuilt() bool  { return (C.mtl_graph_full_built)() == 1 }
func (m *Metal) GraphNumWeights() int  { return int((C.mtl_graph_num_weights)()) }
func (m *Metal) GraphNumDiffable() int { return int((C.mtl_graph_num_diffable)()) }

func (m *Metal) GraphSetVariable(varIdx int, data []float32) int {
	return int((C.mtl_graph_set_variable)(C.int(varIdx), (*C.float)(unsafe.Pointer(&data[0])), C.int(len(data))))
}

func (m *Metal) GraphReadVariable(varIdx int, dst []float32) int {
	return int((C.mtl_graph_read_variable)(C.int(varIdx), (*C.float)(unsafe.Pointer(&dst[0])), C.int(len(dst))))
}

func (m *Metal) GraphApplyWeights(varIdx int, data []float32) int {
	return int((C.mtl_graph_apply_weights)(C.int(varIdx), (*C.float)(unsafe.Pointer(&data[0])), C.int(len(data))))
}

func (m *Metal) GraphAccumAdamStep(lr float32, accumScale float32) int {
	return int((C.mtl_graph_accum_adam_step)(C.float(lr), C.float(accumScale)))
}

func (m *Metal) GraphTrainStepAccum(tokens, targets []int32) float32 {
	return float32((C.mtl_graph_train_step)(
		(*C.int)(unsafe.Pointer(&tokens[0])),
		(*C.int)(unsafe.Pointer(&targets[0])),
		C.int(len(tokens)),
		nil, nil, 0, 0, 2))
}

// --- Inference Graph ---

func (m *Metal) BuildInferGraph(dim, kvDim, headDim, nHeads, nKVHeads, ffnDim, vocabSize, nLayers int, ropeTheta float64) int {
	return int((C.mtl_infer_build)(C.int(dim), C.int(kvDim), C.int(headDim),
		C.int(nHeads), C.int(nKVHeads), C.int(ffnDim),
		C.int(vocabSize), C.int(nLayers), C.float(ropeTheta)))
}

func (m *Metal) InferNumWeights() int {
	return int((C.mtl_infer_num_weights)())
}

func (m *Metal) InferSetWeight(idx int, data []float32) int {
	return int((C.mtl_infer_set_weight)(C.int(idx), (*C.float)(unsafe.Pointer(&data[0])), C.int(len(data))))
}

func (m *Metal) InferForwardA(hidden []float32, cosSlice, sinSlice []float32, qOut, kOut, vOut []float32, layer int) int {
	return int((C.mtl_infer_forward)(
		(*C.float)(unsafe.Pointer(&hidden[0])),
		(*C.float)(unsafe.Pointer(&cosSlice[0])),
		(*C.float)(unsafe.Pointer(&sinSlice[0])),
		(*C.float)(unsafe.Pointer(&qOut[0])),
		(*C.float)(unsafe.Pointer(&kOut[0])),
		(*C.float)(unsafe.Pointer(&vOut[0])),
		nil, nil, C.int(layer)))
}

func (m *Metal) InferForwardB(hidden []float32, attnOut []float32, layer int) int {
	return int((C.mtl_infer_forward_b)(
		(*C.float)(unsafe.Pointer(&hidden[0])),
		(*C.float)(unsafe.Pointer(&attnOut[0])),
		C.int(layer)))
}

func (m *Metal) InferLogits(hidden []float32, logitsOut []float32) int {
	return int((C.mtl_infer_forward)(
		(*C.float)(unsafe.Pointer(&hidden[0])),
		nil, nil, nil, nil, nil, nil,
		(*C.float)(unsafe.Pointer(&logitsOut[0])),
		C.int(10000))) // layer >= nLayers triggers final path
}

func (m *Metal) BuildFused(dim, kvDim, headDim, nHeads, nKVHeads, ffnDim, vocabSize, nLayers, maxSeq int, ropeTheta, rmsEps float64) int {
	return int((C.mtl_fused_build)(C.int(dim), C.int(kvDim), C.int(headDim),
		C.int(nHeads), C.int(nKVHeads), C.int(ffnDim),
		C.int(vocabSize), C.int(nLayers), C.int(maxSeq),
		C.float(ropeTheta), C.float(rmsEps)))
}

// FusedSetArch applies architecture scalar multipliers to the fused inference
// path. Call after BuildFused and before the first forward.
//
// A zero field means "Llama default", so FusedSetArch(ArchParams{}) — or never
// calling it at all — leaves the forward pass byte-identical to before. Granite
// requires all four; see ArchParams.
//
// AdamBeta2 is training-only and ignored here.
func (m *Metal) FusedSetArch(arch ArchParams) int {
	return int((C.mtl_fused_set_arch)(
		C.float(arch.EmbeddingMultiplier),
		C.float(arch.ResidualMultiplier),
		C.float(arch.AttentionScale),
		C.float(arch.LogitsScaling)))
}

func (m *Metal) FusedNumWeights() int {
	return int((C.mtl_fused_num_weights)())
}

func (m *Metal) FusedSetWeight(idx int, data []float32) int {
	return int((C.mtl_fused_set_weight)(C.int(idx), (*C.float)(unsafe.Pointer(&data[0])), C.int(len(data))))
}

func (m *Metal) FusedStep(hidden []float32, cosSlice, sinSlice []float32, pos int, logitsOut []float32) int {
	return int((C.mtl_fused_step)(
		(*C.float)(unsafe.Pointer(&hidden[0])),
		(*C.float)(unsafe.Pointer(&cosSlice[0])),
		(*C.float)(unsafe.Pointer(&sinSlice[0])),
		C.int(pos),
		(*C.float)(unsafe.Pointer(&logitsOut[0]))))
}

func (m *Metal) FusedResetKV() {
	(C.mtl_fused_reset_kv)()
}

func (m *Metal) FusedResetKVSlot(slot int) {
	(C.mtl_fused_reset_kv_slot)(C.int(slot))
}

func (m *Metal) FusedNumSlots() int {
	return int((C.mtl_fused_num_slots)())
}

func (m *Metal) FusedPartialStep(hiddenIn []float32, pos, layerStart, layerEnd int, hiddenOut, logitsOut []float32) int {
	var hOut, lOut *C.float
	if hiddenOut != nil {
		hOut = (*C.float)(unsafe.Pointer(&hiddenOut[0]))
	}
	if logitsOut != nil {
		lOut = (*C.float)(unsafe.Pointer(&logitsOut[0]))
	}
	return int((C.mtl_fused_partial_step)(
		(*C.float)(unsafe.Pointer(&hiddenIn[0])), C.int(pos),
		C.int(layerStart), C.int(layerEnd), hOut, lOut))
}

func (m *Metal) FusedPartialStepSlot(slot int, hiddenIn []float32, pos, layerStart, layerEnd int, hiddenOut, logitsOut []float32) int {
	var hOut, lOut *C.float
	if hiddenOut != nil {
		hOut = (*C.float)(unsafe.Pointer(&hiddenOut[0]))
	}
	if logitsOut != nil {
		lOut = (*C.float)(unsafe.Pointer(&logitsOut[0]))
	}
	return int((C.mtl_fused_partial_step_slot)(C.int(slot),
		(*C.float)(unsafe.Pointer(&hiddenIn[0])), C.int(pos),
		C.int(layerStart), C.int(layerEnd), hOut, lOut))
}

// FusedPrefillTile is the maximum number of prompt tokens FusedPrefillBatch
// accepts in one call.
func (m *Metal) FusedPrefillTile() int {
	return int((C.mtl_fused_prefill_tile)())
}

// FusedPrefillBatch runs B consecutive prompt tokens through the model in a
// single pass, appending them to slot's KV cache from basePos.
//
// Prefilling token-by-token is memory-bound on weights: every token re-reads
// the whole model. Batching reads each weight once per tile instead of once per
// token, which is what makes a long prompt cost seconds rather than minutes.
//
// hiddenIn is B rows of dim embeddings, contiguous. logitsOut may be nil —
// pass it only for the final tile of a prompt, since only the last token's
// logits are sampled and computing them per token adds a full vocab-sized
// matvec for nothing.
func (m *Metal) FusedPrefillBatch(slot int, hiddenIn []float32, B, basePos int, logitsOut []float32) int {
	if B <= 0 || len(hiddenIn) == 0 {
		return -1
	}
	var lOut *C.float
	if logitsOut != nil {
		lOut = (*C.float)(unsafe.Pointer(&logitsOut[0]))
	}
	return int((C.mtl_fused_prefill_batch)(C.int(slot),
		(*C.float)(unsafe.Pointer(&hiddenIn[0])), C.int(B), C.int(basePos), lOut))
}

func (m *Metal) StreamBuild() int {
	return int((C.mtl_stream_build)())
}

func (m *Metal) StreamUploadLayer(set, layer int,
	norm1, wq, wk, wv []float32,
	bq, bk, bv []float32,
	wo, norm2, gate, up, down []float32) {
	var bqP, bkP, bvP *C.float
	if bq != nil {
		bqP = (*C.float)(unsafe.Pointer(&bq[0]))
	}
	if bk != nil {
		bkP = (*C.float)(unsafe.Pointer(&bk[0]))
	}
	if bv != nil {
		bvP = (*C.float)(unsafe.Pointer(&bv[0]))
	}
	(C.mtl_stream_upload_layer)(C.int(set), C.int(layer),
		(*C.float)(unsafe.Pointer(&norm1[0])),
		(*C.float)(unsafe.Pointer(&wq[0])),
		(*C.float)(unsafe.Pointer(&wk[0])),
		(*C.float)(unsafe.Pointer(&wv[0])),
		bqP, bkP, bvP,
		(*C.float)(unsafe.Pointer(&wo[0])),
		(*C.float)(unsafe.Pointer(&norm2[0])),
		(*C.float)(unsafe.Pointer(&gate[0])),
		(*C.float)(unsafe.Pointer(&up[0])),
		(*C.float)(unsafe.Pointer(&down[0])))
}

func (m *Metal) StreamStepLayer(set, layer, pos int) int {
	return int((C.mtl_stream_step_layer)(C.int(set), C.int(layer), C.int(pos)))
}

func (m *Metal) StreamStepFinal(pos int, logitsOut []float32) int {
	return int((C.mtl_stream_step_final)(C.int(pos), (*C.float)(unsafe.Pointer(&logitsOut[0]))))
}

func (m *Metal) StreamSetHidden(hidden []float32) {
	(C.mtl_stream_set_hidden)((*C.float)(unsafe.Pointer(&hidden[0])))
}

func (m *Metal) BuildFusedInfer(dim, kvDim, headDim, nHeads, nKVHeads, ffnDim, vocabSize, nLayers, maxSeq int, ropeTheta float64) int {
	return int((C.mtl_fused_infer_build)(C.int(dim), C.int(kvDim), C.int(headDim),
		C.int(nHeads), C.int(nKVHeads), C.int(ffnDim),
		C.int(vocabSize), C.int(nLayers), C.int(maxSeq), C.float(ropeTheta)))
}

func (m *Metal) FusedInferNumWeights() int {
	return int((C.mtl_fused_infer_num_weights)())
}

func (m *Metal) FusedInferSetWeight(idx int, data []float32) int {
	return int((C.mtl_fused_infer_set_weight)(C.int(idx), (*C.float)(unsafe.Pointer(&data[0])), C.int(len(data))))
}

func (m *Metal) FusedInferStep(hidden []float32, cosSlice, sinSlice []float32, pos int, logitsOut []float32) int {
	return int((C.mtl_fused_infer_step)(
		(*C.float)(unsafe.Pointer(&hidden[0])),
		(*C.float)(unsafe.Pointer(&cosSlice[0])),
		(*C.float)(unsafe.Pointer(&sinSlice[0])),
		C.int(pos),
		(*C.float)(unsafe.Pointer(&logitsOut[0]))))
}

func (m *Metal) FusedInferReset() int {
	return int((C.mtl_fused_infer_reset)())
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
	return (C.mtl_alloc)(C.size_t(sizeFloats * 4))
}

func (m *Metal) poolPut(sizeFloats int, buf C.MTLBufferRef) {
	m.poolMu.Lock()
	if len(m.pool[sizeFloats]) >= 8 {
		m.poolMu.Unlock()
		func() { _cgo0 := buf; _cgoCheckPointer(_cgo0, nil); C.mtl_free(_cgo0) }()
		return
	}
	m.pool[sizeFloats] = append(m.pool[sizeFloats], buf)
	m.poolMu.Unlock()
}

// === Fused Training Compute Kernels ===

func (m *Metal) FusedBegin()              { (C.mtl_fused_begin)() }
func (m *Metal) FusedEnd()                { (C.mtl_fused_end)() }
func (m *Metal) FusedBeginSlot(slot int)  { (C.mtl_fused_begin_slot)(C.int(slot)) }
func (m *Metal) FusedEndSlot(slot int)    { (C.mtl_fused_end_slot)(C.int(slot)) }
func (m *Metal) FusedSetSlot(slot int)    { (C.mtl_fused_set_slot)(C.int(slot)) }
func (m *Metal) FusedCommitSlot(slot int) { (C.mtl_fused_commit_slot)(C.int(slot)) }
func (m *Metal) FusedWaitSlot(slot int)   { (C.mtl_fused_wait_slot)(C.int(slot)) }
func (m *Metal) FusedSyncAll()            { (C.mtl_fused_sync_all)() }

func (m *Metal) FusedGemmBT(a, b, c *Tensor, M, K, N int) {
	func() {
		_cgo0 := MtlBufPtr(a)
		_cgo1 := MtlBufPtr(b)
		_cgo2 := MtlBufPtr(c)
		var _cgo3 C.int = C.int(M)
		var _cgo4 C.int = C.int(K)
		var _cgo5 C.int = C.int(N)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		C.mtl_fused_gemm_bt(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5)
	}()
}
func (m *Metal) FusedGemmNN(a, b, c *Tensor, M, K, N int) {
	func() {
		_cgo0 := MtlBufPtr(a)
		_cgo1 := MtlBufPtr(b)
		_cgo2 := MtlBufPtr(c)
		var _cgo3 C.int = C.int(M)
		var _cgo4 C.int = C.int(K)
		var _cgo5 C.int = C.int(N)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		C.mtl_fused_gemm_nn(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5)
	}()
}
func (m *Metal) FusedGemmTN(a, b, c *Tensor, M, K, N int) {
	func() {
		_cgo0 := MtlBufPtr(a)
		_cgo1 := MtlBufPtr(b)
		_cgo2 := MtlBufPtr(c)
		var _cgo3 C.int = C.int(M)
		var _cgo4 C.int = C.int(K)
		var _cgo5 C.int = C.int(N)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		C.mtl_fused_gemm_tn(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5)
	}()
}
func (m *Metal) FusedGemmF32BT(a, b, c *Tensor, M, K, N int) {
	func() {
		_cgo0 := MtlBufPtr(a)
		_cgo1 := MtlBufPtr(b)
		_cgo2 := MtlBufPtr(c)
		var _cgo3 C.int = C.int(M)
		var _cgo4 C.int = C.int(K)
		var _cgo5 C.int = C.int(N)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		C.mtl_fused_gemm_f32_bt(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5)
	}()
}
func (m *Metal) FusedGemmF32NN(a, b, c *Tensor, M, K, N int) {
	func() {
		_cgo0 := MtlBufPtr(a)
		_cgo1 := MtlBufPtr(b)
		_cgo2 := MtlBufPtr(c)
		var _cgo3 C.int = C.int(M)
		var _cgo4 C.int = C.int(K)
		var _cgo5 C.int = C.int(N)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		C.mtl_fused_gemm_f32_nn(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5)
	}()
}
func (m *Metal) FusedGemmF32TN(a, b, c *Tensor, M, K, N int) {
	func() {
		_cgo0 := MtlBufPtr(a)
		_cgo1 := MtlBufPtr(b)
		_cgo2 := MtlBufPtr(c)
		var _cgo3 C.int = C.int(M)
		var _cgo4 C.int = C.int(K)
		var _cgo5 C.int = C.int(N)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		C.mtl_fused_gemm_f32_tn(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5)
	}()
}
func (m *Metal) FusedRMSNorm(x, w, scale *Tensor, seqLen, dim int) {
	func() {
		_cgo0 := MtlBufPtr(x)
		_cgo1 := MtlBufPtr(w)
		_cgo2 := MtlBufPtr(scale)
		var _cgo3 C.int = C.int(seqLen)
		var _cgo4 C.int = C.int(dim)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		C.mtl_fused_rmsnorm(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4)
	}()
}
func (m *Metal) FusedRMSNormBwd(dOut, xIn, w, scale, dx *Tensor, seqLen, dim int) {
	func() {
		_cgo0 := MtlBufPtr(dOut)
		_cgo1 := MtlBufPtr(xIn)
		_cgo2 := MtlBufPtr(w)
		_cgo3 := MtlBufPtr(scale)
		_cgo4 := MtlBufPtr(dx)
		var _cgo5 C.int = C.int(seqLen)
		var _cgo6 C.int = C.int(dim)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		_cgoCheckPointer(_cgo3, nil)
		_cgoCheckPointer(_cgo4, nil)
		C.mtl_fused_rmsnorm_bwd(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5, _cgo6)
	}()
}
func (m *Metal) FusedRoPE(x *Tensor, headDim, nHeads int, theta float32, stride, seqLen int) {
	func() {
		_cgo0 := MtlBufPtr(x)
		var _cgo1 C.int = C.int(headDim)
		var _cgo2 C.int = C.int(nHeads)
		var _cgo3 C.float = C.float(theta)
		var _cgo4 C.int = C.int(stride)
		var _cgo5 C.int = C.int(seqLen)
		_cgoCheckPointer(_cgo0, nil)
		C.mtl_fused_rope(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5)
	}()
}
func (m *Metal) FusedAttention(q, k, v, out, scores *Tensor, dim, kvDim, headDim, nHeads, nKVHeads, seqLen int) {
	func() {
		_cgo0 := MtlBufPtr(q)
		_cgo1 := MtlBufPtr(k)
		_cgo2 := MtlBufPtr(v)
		_cgo3 := MtlBufPtr(out)
		_cgo4 := MtlBufPtr(scores)
		var _cgo5 C.int = C.int(dim)
		var _cgo6 C.int = C.int(kvDim)
		var _cgo7 C.int = C.int(headDim)
		var _cgo8 C.int = C.int(nHeads)
		var _cgo9 C.int = C.int(nKVHeads)
		var _cgo10 C.int = C.int(seqLen)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		_cgoCheckPointer(_cgo3, nil)
		_cgoCheckPointer(_cgo4, nil)
		C.mtl_fused_attn(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5, _cgo6, _cgo7, _cgo8, _cgo9, _cgo10)
	}()
}
func (m *Metal) FusedAttentionBwdQ(dOut, q, k, v, scores, dQ, dK, dV *Tensor,
	dim, kvDim, headDim, nHeads, nKVHeads, seqLen, qLen int) {
	func() {
		_cgo0 := MtlBufPtr(dOut)
		_cgo1 := MtlBufPtr(q)
		_cgo2 := MtlBufPtr(k)
		_cgo3 := MtlBufPtr(v)
		_cgo4 := MtlBufPtr(scores)
		_cgo5 := MtlBufPtr(dQ)
		_cgo6 := MtlBufPtr(dK)
		_cgo7 := MtlBufPtr(dV)
		var _cgo8 C.int = C.int(dim)
		var _cgo9 C.int = C.int(kvDim)
		var _cgo10 C.int = C.int(headDim)
		var _cgo11 C.int = C.int(nHeads)
		var _cgo12 C.int = C.int(nKVHeads)
		var _cgo13 C.int = C.int(seqLen)
		var _cgo14 C.int = C.int(qLen)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		_cgoCheckPointer(_cgo3, nil)
		_cgoCheckPointer(_cgo4, nil)
		_cgoCheckPointer(_cgo5, nil)
		_cgoCheckPointer(_cgo6, nil)
		_cgoCheckPointer(_cgo7, nil)
		C.mtl_fused_attention_bwd_q(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5, _cgo6, _cgo7, _cgo8, _cgo9, _cgo10, _cgo11, _cgo12, _cgo13, _cgo14)
	}()
}
func (m *Metal) FusedSiLUGateMul(gate, up, out *Tensor, n int) {
	func() {
		_cgo0 := MtlBufPtr(gate)
		_cgo1 := MtlBufPtr(up)
		_cgo2 := MtlBufPtr(out)
		var _cgo3 C.int = C.int(n)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		C.mtl_fused_silu_gate_mul(_cgo0, _cgo1, _cgo2, _cgo3)
	}()
}
func (m *Metal) SiLUGateBackward(dOut, gatePre, upOut, gateAct, dGatePre, dUp *Tensor) {
	func() {
		_cgo0 := MtlBufPtr(dOut)
		_cgo1 := MtlBufPtr(gatePre)
		_cgo2 := MtlBufPtr(upOut)
		_cgo3 := MtlBufPtr(gateAct)
		_cgo4 := MtlBufPtr(dGatePre)
		_cgo5 := MtlBufPtr(dUp)
		var _cgo6 C.int = C.int(dOut.Size)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		_cgoCheckPointer(_cgo3, nil)
		_cgoCheckPointer(_cgo4, nil)
		_cgoCheckPointer(_cgo5, nil)
		C.mtl_silu_gate_backward_gpu(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5, _cgo6)
	}()
}
func (m *Metal) FusedAddInPlace(a, b *Tensor, n int) {
	func() {
		_cgo0 := MtlBufPtr(a)
		_cgo1 := MtlBufPtr(b)
		var _cgo2 C.int = C.int(n)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		C.mtl_fused_add_inplace(_cgo0, _cgo1, _cgo2)
	}()
}
func (m *Metal) FusedCopy(dst, src *Tensor, n int) {
	func() {
		_cgo0 := MtlBufPtr(dst)
		_cgo1 := MtlBufPtr(src)
		var _cgo2 C.int = C.int(n)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		C.mtl_fused_copy(_cgo0, _cgo1, _cgo2)
	}()
}
func (m *Metal) CELoss(logits, targets, losses *Tensor, seqLen, vocabSize int) {
	func() {
		_cgo0 := MtlBufPtr(logits)
		_cgo1 := MtlBufPtr(targets)
		_cgo2 := MtlBufPtr(losses)
		var _cgo3 C.int = C.int(seqLen)
		var _cgo4 C.int = C.int(vocabSize)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		C.mtl_ce_loss(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4)
	}()
}
func (m *Metal) AdamWT(param, grad, mState, vState *Tensor, lr, wd float32, step int) {
	bc1 := C.float(1.0 - math.Pow(0.9, float64(step)))
	bc2 := C.float(1.0 - math.Pow(0.95, float64(step)))
	func() {
		_cgo0 := MtlBufPtr(param)
		_cgo1 := MtlBufPtr(grad)
		_cgo2 := MtlBufPtr(mState)
		_cgo3 := MtlBufPtr(vState)
		var _cgo4 C.float = C.float(lr)
		var _cgo5 C.float = C.float(0.9)
		var _cgo6 C.float = C.float(0.95)
		var _cgo7 C.float = bc1
		var _cgo8 C.float = bc2
		var _cgo9 C.float = C.float(1e-8)
		var _cgo10 C.float = C.float(wd)
		var _cgo11 C.int = C.int(param.Size)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		_cgoCheckPointer(_cgo3, nil)
		C.mtl_adamw_gpu(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5, _cgo6, _cgo7, _cgo8, _cgo9, _cgo10, _cgo11)
	}()
}
func (m *Metal) DNARungGPU(d1, g1, m1, v1, d2, g2, m2, v2 *Tensor,
	bb1, gly1, hb1, hb2, gly2, bb2, bondStr, lr, beta1, beta2, bc1, bc2, eps, wd float32, n int) {
	func() {
		_cgo0 := MtlBufPtr(d1)
		_cgo1 := MtlBufPtr(g1)
		_cgo2 := MtlBufPtr(m1)
		_cgo3 := MtlBufPtr(v1)
		_cgo4 := MtlBufPtr(d2)
		_cgo5 := MtlBufPtr(g2)
		_cgo6 := MtlBufPtr(m2)
		_cgo7 := MtlBufPtr(v2)
		var _cgo8 C.float = C.float(bb1)
		var _cgo9 C.float = C.float(gly1)
		var _cgo10 C.float = C.float(hb1)
		var _cgo11 C.float = C.float(hb2)
		var _cgo12 C.float = C.float(gly2)
		var _cgo13 C.float = C.float(bb2)
		var _cgo14 C.float = C.float(bondStr)
		var _cgo15 C.float = C.float(lr)
		var _cgo16 C.float = C.float(beta1)
		var _cgo17 C.float = C.float(beta2)
		var _cgo18 C.float = C.float(bc1)
		var _cgo19 C.float = C.float(bc2)
		var _cgo20 C.float = C.float(eps)
		var _cgo21 C.float = C.float(wd)
		var _cgo22 C.int = C.int(n)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		_cgoCheckPointer(_cgo3, nil)
		_cgoCheckPointer(_cgo4, nil)
		_cgoCheckPointer(_cgo5, nil)
		_cgoCheckPointer(_cgo6, nil)
		_cgoCheckPointer(_cgo7, nil)
		C.mtl_dna_rung_gpu(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5, _cgo6, _cgo7, _cgo8, _cgo9, _cgo10, _cgo11, _cgo12, _cgo13, _cgo14, _cgo15, _cgo16, _cgo17, _cgo18, _cgo19, _cgo20, _cgo21, _cgo22)
	}()
}
func (m *Metal) GradNormSqGPU(grad, out *Tensor, n int) {
	func() {
		_cgo0 := MtlBufPtr(grad)
		_cgo1 := MtlBufPtr(out)
		var _cgo2 C.int = C.int(n)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		C.mtl_grad_norm_sq(_cgo0, _cgo1, _cgo2)
	}()
}

// WarmCache is a single MTLBuffer in unified memory that holds all optimizer state
// (momentum + velocity for every parameter). Both CPU and GPU access the same physical
// pages — the CPU (helix) computes rung geometry and reads/writes m/v via []float32
// slices; the GPU kernel reads/writes the same m/v via buffer offsets. No copy.
type WarmCache struct {
	buf     C.MTLBufferRef
	nFloats int
	sharedF []float32
}

// NewWarmCache allocates a single unified-memory buffer large enough to hold
// nFloats float32 values. Returns a WarmCache whose Slice method yields
// CPU-visible []float32 windows that the GPU also reads/writes.
func (m *Metal) NewWarmCache(nFloats int) *WarmCache {
	buf := (C.mtl_alloc)(C.size_t(nFloats * 4))
	func() {
		_cgo0 := buf
		var _cgo1 C.size_t = C.size_t(nFloats * 4)
		_cgoCheckPointer(_cgo0, nil)
		C.mtl_zero(_cgo0, _cgo1)
	}()

	ptr := func() unsafe.Pointer { _cgo0 := buf; _cgoCheckPointer(_cgo0, nil); return C.mtl_shared_ptr(_cgo0) }()
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
		func() { _cgo0 := wc.buf; _cgoCheckPointer(_cgo0, nil); C.mtl_free(_cgo0) }()
		wc.buf = nil
		wc.sharedF = nil
	}
}

// SharedSlice returns a CPU-visible []float32 view of a Tensor's underlying
// MTLBuffer. On Apple Silicon this IS the GPU memory — unified architecture.
// The returned slice is valid as long as the Tensor is not released.
func (m *Metal) SharedSlice(t *Tensor) []float32 {
	mp := t.device.(*mtlPtr)
	ptr := func() unsafe.Pointer { _cgo0 := mp.buf; _cgoCheckPointer(_cgo0, nil); return C.mtl_shared_ptr(_cgo0) }()
	return (*[1 << 30]float32)(ptr)[:t.Size:t.Size]
}

// DNARungWarm dispatches the paired DNA rung kernel with m/v read from a warm cache
// at the given float offsets. No separate m/v tensor allocations needed.
func (m *Metal) DNARungWarm(d1, g1, d2, g2 *Tensor, wc *WarmCache,
	m1Off, v1Off, m2Off, v2Off int,
	bb1, gly1, hb1, hb2, gly2, bb2, bondStr, lr, beta1, beta2, bc1, bc2, eps, wd float32, n int) {
	func() {
		_cgo0 := MtlBufPtr(d1)
		_cgo1 := MtlBufPtr(g1)
		_cgo2 := MtlBufPtr(d2)
		_cgo3 := MtlBufPtr(g2)
		_cgo4 := wc.BufPtr()
		var _cgo5 C.int = C.int(wc.ByteOffset(m1Off))
		var _cgo6 C.int = C.int(wc.ByteOffset(v1Off))
		var _cgo7 C.int = C.int(wc.ByteOffset(m2Off))
		var _cgo8 C.int = C.int(wc.ByteOffset(v2Off))
		var _cgo9 C.float = C.float(bb1)
		var _cgo10 C.float = C.float(gly1)
		var _cgo11 C.float = C.float(hb1)
		var _cgo12 C.float = C.float(hb2)
		var _cgo13 C.float = C.float(gly2)
		var _cgo14 C.float = C.float(bb2)
		var _cgo15 C.float = C.float(bondStr)
		var _cgo16 C.float = C.float(lr)
		var _cgo17 C.float = C.float(beta1)
		var _cgo18 C.float = C.float(beta2)
		var _cgo19 C.float = C.float(bc1)
		var _cgo20 C.float = C.float(bc2)
		var _cgo21 C.float = C.float(eps)
		var _cgo22 C.float = C.float(wd)
		var _cgo23 C.int = C.int(n)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		_cgoCheckPointer(_cgo3, nil)
		_cgoCheckPointer(_cgo4, nil)
		C.mtl_dna_rung_warm(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5, _cgo6, _cgo7, _cgo8, _cgo9, _cgo10, _cgo11, _cgo12, _cgo13, _cgo14, _cgo15, _cgo16, _cgo17, _cgo18, _cgo19, _cgo20, _cgo21, _cgo22, _cgo23)
	}()
}

// AdamWWarm dispatches the AdamW kernel with m/v read from a warm cache at the given
// float offsets. Single-strand update for unpaired parameters.
func (m *Metal) AdamWWarm(param, grad *Tensor, wc *WarmCache, mOff, vOff int,
	lr, beta1, beta2, bc1, bc2, eps, wd float32, n int) {
	func() {
		_cgo0 := MtlBufPtr(param)
		_cgo1 := MtlBufPtr(grad)
		_cgo2 := wc.BufPtr()
		var _cgo3 C.int = C.int(wc.ByteOffset(mOff))
		var _cgo4 C.int = C.int(wc.ByteOffset(vOff))
		var _cgo5 C.float = C.float(lr)
		var _cgo6 C.float = C.float(beta1)
		var _cgo7 C.float = C.float(beta2)
		var _cgo8 C.float = C.float(bc1)
		var _cgo9 C.float = C.float(bc2)
		var _cgo10 C.float = C.float(eps)
		var _cgo11 C.float = C.float(wd)
		var _cgo12 C.int = C.int(n)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		C.mtl_adamw_warm(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5, _cgo6, _cgo7, _cgo8, _cgo9, _cgo10, _cgo11, _cgo12)
	}()
}

func (m *Metal) FusedZeroScalar(buf *Tensor) {
	func() { _cgo0 := MtlBufPtr(buf); _cgoCheckPointer(_cgo0, nil); C.mtl_fused_zero_scalar(_cgo0) }()
}
func (m *Metal) FusedBarrierBuffers() { (C.mtl_fused_barrier_buffers)() }
func (m *Metal) FusedEndAsync()       { (C.mtl_fused_end_async)() }
func (m *Metal) FusedWait()           { (C.mtl_fused_wait)() }

func (m *Metal) FusedGradNormSq(grad, sumSq *Tensor, n int) {
	func() {
		_cgo0 := MtlBufPtr(grad)
		_cgo1 := MtlBufPtr(sumSq)
		var _cgo2 C.int = C.int(n)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		C.mtl_fused_grad_norm_sq(_cgo0, _cgo1, _cgo2)
	}()
}

func (m *Metal) ICBExecuteFwd()  { (C.mtl_icb_execute_fwd)() }
func (m *Metal) ICBExecuteFull() { (C.mtl_icb_execute_full)() }

type ICBLayerActs struct {
	XIn, Normed, Q, K, V, AttnOut         *Tensor
	XMid, Normed2, GatePre, UpOut, FfnMid *Tensor
	RmsScale1, RmsScale2, GateAct         *Tensor
}

type ICBLayerInt8 struct {
	Data, Scales, Delta, Mom, Vel, Live *Tensor
	Mask                                *HotRowMask
}

type ICBLayerBwd struct {
	DFfnMid, DGate, DUp, DN2, Dx             *Tensor
	DAttnOut, DQ, DK, DV, DN1                *Tensor
	DWDown, DWGate, DWUp, DWO, DWQ, DWK, DWV *Tensor
}

type ICBLayerWeights struct {
	WQ, WK, WV, WO, Gate, Up, Down ICBLayerInt8
	Norm1, Norm2                   *Tensor
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
		arr := (*[64]unsafe.Pointer)((C.malloc)(C.size_t(len(tensors) * 8)))
		for i, t := range tensors {
			arr[i] = MtlBufPtr(t)
		}
		return &arr[0]
	}
	mkMaskArr := func(masks []*HotRowMask) *unsafe.Pointer {
		arr := (*[64]unsafe.Pointer)((C.malloc)(C.size_t(len(masks) * 8)))
		for i, m := range masks {
			arr[i] = m.BufPtr()
		}
		return &arr[0]
	}

	// Collect per-layer arrays
	norm1s := make([]*Tensor, nL)
	norm2s := make([]*Tensor, nL)
	aXIn := make([]*Tensor, nL)
	aNormed := make([]*Tensor, nL)
	aQ := make([]*Tensor, nL)
	aK := make([]*Tensor, nL)
	aV := make([]*Tensor, nL)
	aAttnOut := make([]*Tensor, nL)
	aXMid := make([]*Tensor, nL)
	aNormed2 := make([]*Tensor, nL)
	aGatePre := make([]*Tensor, nL)
	aUpOut := make([]*Tensor, nL)
	aFfnMid := make([]*Tensor, nL)
	aRmsScale1 := make([]*Tensor, nL)
	aRmsScale2 := make([]*Tensor, nL)
	aGateAct := make([]*Tensor, nL)

	wqD := make([]*Tensor, nL)
	wqS := make([]*Tensor, nL)
	wqDl := make([]*Tensor, nL)
	wqM := make([]*Tensor, nL)
	wqV := make([]*Tensor, nL)
	wqL := make([]*Tensor, nL)
	wqMk := make([]*HotRowMask, nL)
	wkD := make([]*Tensor, nL)
	wkS := make([]*Tensor, nL)
	wkDl := make([]*Tensor, nL)
	wkM := make([]*Tensor, nL)
	wkV := make([]*Tensor, nL)
	wkL := make([]*Tensor, nL)
	wkMk := make([]*HotRowMask, nL)
	wvD := make([]*Tensor, nL)
	wvS := make([]*Tensor, nL)
	wvDl := make([]*Tensor, nL)
	wvM := make([]*Tensor, nL)
	wvV := make([]*Tensor, nL)
	wvL := make([]*Tensor, nL)
	wvMk := make([]*HotRowMask, nL)
	woD := make([]*Tensor, nL)
	woS := make([]*Tensor, nL)
	woDl := make([]*Tensor, nL)
	woM := make([]*Tensor, nL)
	woV := make([]*Tensor, nL)
	woL := make([]*Tensor, nL)
	woMk := make([]*HotRowMask, nL)
	gD := make([]*Tensor, nL)
	gS := make([]*Tensor, nL)
	gDl := make([]*Tensor, nL)
	gM := make([]*Tensor, nL)
	gV := make([]*Tensor, nL)
	gL := make([]*Tensor, nL)
	gMk := make([]*HotRowMask, nL)
	uD := make([]*Tensor, nL)
	uS := make([]*Tensor, nL)
	uDl := make([]*Tensor, nL)
	uM := make([]*Tensor, nL)
	uV := make([]*Tensor, nL)
	uL := make([]*Tensor, nL)
	uMk := make([]*HotRowMask, nL)
	dD := make([]*Tensor, nL)
	dS := make([]*Tensor, nL)
	dDl := make([]*Tensor, nL)
	dM := make([]*Tensor, nL)
	dV2 := make([]*Tensor, nL)
	dL := make([]*Tensor, nL)
	dMk := make([]*HotRowMask, nL)

	bDFfn := make([]*Tensor, nL)
	bDGate := make([]*Tensor, nL)
	bDUp := make([]*Tensor, nL)
	bDN2 := make([]*Tensor, nL)
	bDx := make([]*Tensor, nL)
	bDAttn := make([]*Tensor, nL)
	bDQ := make([]*Tensor, nL)
	bDK := make([]*Tensor, nL)
	bDV := make([]*Tensor, nL)
	bDN1 := make([]*Tensor, nL)
	bDWD := make([]*Tensor, nL)
	bDWG := make([]*Tensor, nL)
	bDWU := make([]*Tensor, nL)
	bDWO := make([]*Tensor, nL)
	bDWQ := make([]*Tensor, nL)
	bDWK := make([]*Tensor, nL)
	bDWV := make([]*Tensor, nL)

	for i := 0; i < nL; i++ {
		norm1s[i] = weights[i].Norm1
		norm2s[i] = weights[i].Norm2
		a := acts[i]
		aXIn[i] = a.XIn
		aNormed[i] = a.Normed
		aQ[i] = a.Q
		aK[i] = a.K
		aV[i] = a.V
		aAttnOut[i] = a.AttnOut
		aXMid[i] = a.XMid
		aNormed2[i] = a.Normed2
		aGatePre[i] = a.GatePre
		aUpOut[i] = a.UpOut
		aFfnMid[i] = a.FfnMid
		aRmsScale1[i] = a.RmsScale1
		aRmsScale2[i] = a.RmsScale2
		aGateAct[i] = a.GateAct

		w := weights[i]
		wqD[i] = w.WQ.Data
		wqS[i] = w.WQ.Scales
		wqDl[i] = w.WQ.Delta
		wqM[i] = w.WQ.Mom
		wqV[i] = w.WQ.Vel
		wqL[i] = w.WQ.Live
		wqMk[i] = w.WQ.Mask
		wkD[i] = w.WK.Data
		wkS[i] = w.WK.Scales
		wkDl[i] = w.WK.Delta
		wkM[i] = w.WK.Mom
		wkV[i] = w.WK.Vel
		wkL[i] = w.WK.Live
		wkMk[i] = w.WK.Mask
		wvD[i] = w.WV.Data
		wvS[i] = w.WV.Scales
		wvDl[i] = w.WV.Delta
		wvM[i] = w.WV.Mom
		wvV[i] = w.WV.Vel
		wvL[i] = w.WV.Live
		wvMk[i] = w.WV.Mask
		woD[i] = w.WO.Data
		woS[i] = w.WO.Scales
		woDl[i] = w.WO.Delta
		woM[i] = w.WO.Mom
		woV[i] = w.WO.Vel
		woL[i] = w.WO.Live
		woMk[i] = w.WO.Mask
		gD[i] = w.Gate.Data
		gS[i] = w.Gate.Scales
		gDl[i] = w.Gate.Delta
		gM[i] = w.Gate.Mom
		gV[i] = w.Gate.Vel
		gL[i] = w.Gate.Live
		gMk[i] = w.Gate.Mask
		uD[i] = w.Up.Data
		uS[i] = w.Up.Scales
		uDl[i] = w.Up.Delta
		uM[i] = w.Up.Mom
		uV[i] = w.Up.Vel
		uL[i] = w.Up.Live
		uMk[i] = w.Up.Mask
		dD[i] = w.Down.Data
		dS[i] = w.Down.Scales
		dDl[i] = w.Down.Delta
		dM[i] = w.Down.Mom
		dV2[i] = w.Down.Vel
		dL[i] = w.Down.Live
		dMk[i] = w.Down.Mask

		b := bwds[i]
		bDFfn[i] = b.DFfnMid
		bDGate[i] = b.DGate
		bDUp[i] = b.DUp
		bDN2[i] = b.DN2
		bDx[i] = b.Dx
		bDAttn[i] = b.DAttnOut
		bDQ[i] = b.DQ
		bDK[i] = b.DK
		bDV[i] = b.DV
		bDN1[i] = b.DN1
		bDWD[i] = b.DWDown
		bDWG[i] = b.DWGate
		bDWU[i] = b.DWUp
		bDWO[i] = b.DWO
		bDWQ[i] = b.DWQ
		bDWK[i] = b.DWK
		bDWV[i] = b.DWV
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
		embed:     MtlBufPtr(embed),
		embedData: MtlBufPtr(embedInt8.Data), embedScales: MtlBufPtr(embedInt8.Scales),
		embedDelta: MtlBufPtr(embedInt8.Delta), embedMom: MtlBufPtr(embedInt8.Mom),
		embedVel: MtlBufPtr(embedInt8.Vel), embedMask: embedInt8.Mask.BufPtr(),
		embedLive: MtlBufPtr(embedInt8.Live),
		norm1:     mkArr(norm1s), norm2: mkArr(norm2s),
		a_xIn: mkArr(aXIn), a_normed: mkArr(aNormed), a_Q: mkArr(aQ), a_K: mkArr(aK),
		a_V: mkArr(aV), a_attnOut: mkArr(aAttnOut), a_xMid: mkArr(aXMid),
		a_normed2: mkArr(aNormed2), a_gatePre: mkArr(aGatePre), a_upOut: mkArr(aUpOut),
		a_ffnMid: mkArr(aFfnMid), a_rmsScale1: mkArr(aRmsScale1), a_rmsScale2: mkArr(aRmsScale2),
		a_gateAct: mkArr(aGateAct),
		wq_data:   mkArr(wqD), wq_scales: mkArr(wqS), wq_delta: mkArr(wqDl),
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
		bb1Buf:     MtlBufPtr(bb1Buf), gly1Buf: MtlBufPtr(gly1Buf), hb1Buf: MtlBufPtr(hb1Buf),
		hb2Buf: MtlBufPtr(hb2Buf), gly2Buf: MtlBufPtr(gly2Buf), bb2Buf: MtlBufPtr(bb2Buf),
		bondStrBuf: MtlBufPtr(bondStrBuf),
	}
	return int(func() C.int {
		_cgoBase0 := &p
		_cgo0 := _cgoBase0
		_cgoCheckPointer(_cgoBase0, 0 == 0)
		return C.mtl_icb_build_training(_cgo0)
	}())
}

func (m *Metal) FusedComputeClipScale(sumSq, clipScale *Tensor, maxNorm float32) {
	func() {
		_cgo0 := MtlBufPtr(sumSq)
		_cgo1 := MtlBufPtr(clipScale)
		var _cgo2 C.float = C.float(maxNorm)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		C.mtl_fused_compute_clip_scale(_cgo0, _cgo1, _cgo2)
	}()
}

func (m *Metal) FusedGradClipScale(grad, sumSq *Tensor, maxNorm float32, n int) {
	func() {
		_cgo0 := MtlBufPtr(grad)
		_cgo1 := MtlBufPtr(sumSq)
		var _cgo2 C.float = C.float(maxNorm)
		var _cgo3 C.int = C.int(n)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		C.mtl_fused_grad_clip_scale(_cgo0, _cgo1, _cgo2, _cgo3)
	}()
}

func (m *Metal) FusedDequantDelta(src, scales, delta, dst *Tensor, n, cols int) {
	func() {
		_cgo0 := MtlBufPtr(src)
		_cgo1 := MtlBufPtr(scales)
		_cgo2 := MtlBufPtr(delta)
		_cgo3 := MtlBufPtr(dst)
		var _cgo4 C.int = C.int(n)
		var _cgo5 C.int = C.int(cols)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		_cgoCheckPointer(_cgo3, nil)
		C.mtl_fused_dequant_delta(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5)
	}()
}

func (m *Metal) FusedTrainAvailable() bool { return (C.mtl_fused_train_available)() == 1 }

func (m *Metal) FusedPreAttn(hidden, normW, wq, wk, wv, Q, K, V, normedOut, rmsScale, xIn *Tensor,
	dim, kvDim, headDim, nHeads, nKVHeads, ffnDim, seqLen int, ropeTheta, eps float32) {
	func() {
		_cgo0 := MtlBufPtr(hidden)
		_cgo1 := MtlBufPtr(normW)
		_cgo2 := MtlBufPtr(wq)
		_cgo3 := MtlBufPtr(wk)
		_cgo4 := MtlBufPtr(wv)
		_cgo5 := MtlBufPtr(Q)
		_cgo6 := MtlBufPtr(K)
		_cgo7 := MtlBufPtr(V)
		_cgo8 := MtlBufPtr(normedOut)
		_cgo9 := MtlBufPtr(rmsScale)
		_cgo10 := MtlBufPtr(xIn)
		var _cgo11 C.int = C.int(dim)
		var _cgo12 C.int = C.int(kvDim)
		var _cgo13 C.int = C.int(headDim)
		var _cgo14 C.int = C.int(nHeads)
		var _cgo15 C.int = C.int(nKVHeads)
		var _cgo16 C.int = C.int(ffnDim)
		var _cgo17 C.int = C.int(seqLen)
		var _cgo18 C.float = C.float(ropeTheta)
		var _cgo19 C.float = C.float(eps)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		_cgoCheckPointer(_cgo3, nil)
		_cgoCheckPointer(_cgo4, nil)
		_cgoCheckPointer(_cgo5, nil)
		_cgoCheckPointer(_cgo6, nil)
		_cgoCheckPointer(_cgo7, nil)
		_cgoCheckPointer(_cgo8, nil)
		_cgoCheckPointer(_cgo9, nil)
		_cgoCheckPointer(_cgo10, nil)
		C.mtl_fused_pre_attn(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5, _cgo6, _cgo7, _cgo8, _cgo9, _cgo10, _cgo11, _cgo12, _cgo13, _cgo14, _cgo15, _cgo16, _cgo17, _cgo18, _cgo19)
	}()
}

func (m *Metal) FusedPostAttn(hidden, attnOut, wo, normW2, gate, up, down, xMid, normed2, rmsScale2, gatePre, upOut, ffnMid *Tensor,
	dim, kvDim, headDim, nHeads, nKVHeads, ffnDim, seqLen int, ropeTheta, eps float32) {
	func() {
		_cgo0 := MtlBufPtr(hidden)
		_cgo1 := MtlBufPtr(attnOut)
		_cgo2 := MtlBufPtr(wo)
		_cgo3 := MtlBufPtr(normW2)
		_cgo4 := MtlBufPtr(gate)
		_cgo5 := MtlBufPtr(up)
		_cgo6 := MtlBufPtr(down)
		_cgo7 := MtlBufPtr(xMid)
		_cgo8 := MtlBufPtr(normed2)
		_cgo9 := MtlBufPtr(rmsScale2)
		_cgo10 := MtlBufPtr(gatePre)
		_cgo11 := MtlBufPtr(upOut)
		_cgo12 := MtlBufPtr(ffnMid)
		var _cgo13 C.int = C.int(dim)
		var _cgo14 C.int = C.int(kvDim)
		var _cgo15 C.int = C.int(headDim)
		var _cgo16 C.int = C.int(nHeads)
		var _cgo17 C.int = C.int(nKVHeads)
		var _cgo18 C.int = C.int(ffnDim)
		var _cgo19 C.int = C.int(seqLen)
		var _cgo20 C.float = C.float(ropeTheta)
		var _cgo21 C.float = C.float(eps)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		_cgoCheckPointer(_cgo3, nil)
		_cgoCheckPointer(_cgo4, nil)
		_cgoCheckPointer(_cgo5, nil)
		_cgoCheckPointer(_cgo6, nil)
		_cgoCheckPointer(_cgo7, nil)
		_cgoCheckPointer(_cgo8, nil)
		_cgoCheckPointer(_cgo9, nil)
		_cgoCheckPointer(_cgo10, nil)
		_cgoCheckPointer(_cgo11, nil)
		_cgoCheckPointer(_cgo12, nil)
		C.mtl_fused_post_attn(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5, _cgo6, _cgo7, _cgo8, _cgo9, _cgo10, _cgo11, _cgo12, _cgo13, _cgo14, _cgo15, _cgo16, _cgo17, _cgo18, _cgo19, _cgo20, _cgo21)
	}()
}

func (m *Metal) FusedLMHeadPass1(hidden, embed, maxBuf, sumExp *Tensor, dim, vocabSize, n int) {
	func() {
		_cgo0 := MtlBufPtr(hidden)
		_cgo1 := MtlBufPtr(embed)
		_cgo2 := MtlBufPtr(maxBuf)
		_cgo3 := MtlBufPtr(sumExp)
		var _cgo4 C.int = C.int(dim)
		var _cgo5 C.int = C.int(vocabSize)
		var _cgo6 C.int = C.int(n)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		_cgoCheckPointer(_cgo3, nil)
		C.mtl_fused_lm_head_pass1(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5, _cgo6)
	}()
}

func (m *Metal) FusedLMHeadPass2(hidden, embed, maxBuf, sumExp, targets, dHidden, loss *Tensor, dim, vocabSize, n int) {
	func() {
		_cgo0 := MtlBufPtr(hidden)
		_cgo1 := MtlBufPtr(embed)
		_cgo2 := MtlBufPtr(maxBuf)
		_cgo3 := MtlBufPtr(sumExp)
		_cgo4 := MtlBufPtr(targets)
		_cgo5 := MtlBufPtr(dHidden)
		_cgo6 := MtlBufPtr(loss)
		var _cgo7 C.int = C.int(dim)
		var _cgo8 C.int = C.int(vocabSize)
		var _cgo9 C.int = C.int(n)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		_cgoCheckPointer(_cgo3, nil)
		_cgoCheckPointer(_cgo4, nil)
		_cgoCheckPointer(_cgo5, nil)
		_cgoCheckPointer(_cgo6, nil)
		C.mtl_fused_lm_head_pass2(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5, _cgo6, _cgo7, _cgo8, _cgo9)
	}()
}

func (m *Metal) LMHeadForwardGrad(hidden, embed, maxBuf, sumExp, targets, dHidden, loss *Tensor, dim, vocabSize, n int) {
	func() {
		_cgo0 := MtlBufPtr(hidden)
		_cgo1 := MtlBufPtr(embed)
		_cgo2 := MtlBufPtr(maxBuf)
		_cgo3 := MtlBufPtr(sumExp)
		_cgo4 := MtlBufPtr(targets)
		_cgo5 := MtlBufPtr(dHidden)
		_cgo6 := MtlBufPtr(loss)
		var _cgo7 C.int = C.int(dim)
		var _cgo8 C.int = C.int(vocabSize)
		var _cgo9 C.int = C.int(n)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		_cgoCheckPointer(_cgo3, nil)
		_cgoCheckPointer(_cgo4, nil)
		_cgoCheckPointer(_cgo5, nil)
		_cgoCheckPointer(_cgo6, nil)
		C.mtl_lm_head_forward_grad(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5, _cgo6, _cgo7, _cgo8, _cgo9)
	}()
}

func (m *Metal) SoftmaxCEGrad(logits, targets, losses, grad *Tensor, seqLen, vocabSize int, invN float32) {
	func() {
		_cgo0 := MtlBufPtr(logits)
		_cgo1 := MtlBufPtr(targets)
		_cgo2 := MtlBufPtr(losses)
		_cgo3 := MtlBufPtr(grad)
		var _cgo4 C.int = C.int(seqLen)
		var _cgo5 C.int = C.int(vocabSize)
		var _cgo6 C.float = C.float(invN)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		_cgoCheckPointer(_cgo3, nil)
		C.mtl_softmax_ce_grad(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5, _cgo6)
	}()
}

func (m *Metal) FusedGemmF32TNSparse(a, b, c *Tensor, mask *HotRowMask, M, K, N int) {
	func() {
		_cgo0 := MtlBufPtr(a)
		_cgo1 := MtlBufPtr(b)
		_cgo2 := MtlBufPtr(c)
		_cgo3 := mask.BufPtr()
		var _cgo4 C.int = C.int(M)
		var _cgo5 C.int = C.int(K)
		var _cgo6 C.int = C.int(N)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		_cgoCheckPointer(_cgo3, nil)
		C.mtl_fused_gemm_tn_sparse(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5, _cgo6)
	}()
}

func (m *Metal) FusedDequantDeltaSparse(src, scales, delta, dst *Tensor, mask *HotRowMask, n, cols int) {
	func() {
		_cgo0 := MtlBufPtr(src)
		_cgo1 := MtlBufPtr(scales)
		_cgo2 := MtlBufPtr(delta)
		_cgo3 := MtlBufPtr(dst)
		_cgo4 := mask.BufPtr()
		var _cgo5 C.int = C.int(n)
		var _cgo6 C.int = C.int(cols)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		_cgoCheckPointer(_cgo3, nil)
		_cgoCheckPointer(_cgo4, nil)
		C.mtl_fused_dequant_delta_sparse(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5, _cgo6)
	}()
}

func (m *Metal) FusedNeedle(data, scales, grad, mom, vel *Tensor, mask *HotRowMask, delta *Tensor,
	lr, beta1, beta2, bc1, bc2, eps, wd float32, n, cols int, live, clipBuf *Tensor) {
	func() {
		_cgo0 := MtlBufPtr(data)
		_cgo1 := MtlBufPtr(scales)
		_cgo2 := MtlBufPtr(grad)
		_cgo3 := MtlBufPtr(mom)
		_cgo4 := MtlBufPtr(vel)
		_cgo5 := mask.BufPtr()
		_cgo6 := MtlBufPtr(delta)
		var _cgo7 C.float = C.float(lr)
		var _cgo8 C.float = C.float(beta1)
		var _cgo9 C.float = C.float(beta2)
		var _cgo10 C.float = C.float(bc1)
		var _cgo11 C.float = C.float(bc2)
		var _cgo12 C.float = C.float(eps)
		var _cgo13 C.float = C.float(wd)
		var _cgo14 C.int = C.int(n)
		var _cgo15 C.int = C.int(cols)
		_cgo16 := MtlBufPtr(live)
		_cgo17 := MtlBufPtr(clipBuf)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		_cgoCheckPointer(_cgo3, nil)
		_cgoCheckPointer(_cgo4, nil)
		_cgoCheckPointer(_cgo5, nil)
		_cgoCheckPointer(_cgo6, nil)
		_cgoCheckPointer(_cgo16, nil)
		_cgoCheckPointer(_cgo17, nil)
		C.mtl_fused_needle(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5, _cgo6, _cgo7, _cgo8, _cgo9, _cgo10, _cgo11, _cgo12, _cgo13, _cgo14, _cgo15, _cgo16, _cgo17)
	}()
}

func (m *Metal) FusedNeedlePaired(d1, d2, s1, s2, g1, g2, m1, m2, v1, v2 *Tensor, mask *HotRowMask, delta1, delta2 *Tensor,
	lr, beta1, beta2, bc1, bc2, eps, wd,
	backbone1, glyco1, hbond1, hbond2, glyco2, backbone2, bondStrength float32,
	n, cols int, live1, live2, clipBuf *Tensor) {
	func() {
		_cgo0 := MtlBufPtr(d1)
		_cgo1 := MtlBufPtr(d2)
		_cgo2 := MtlBufPtr(s1)
		_cgo3 := MtlBufPtr(s2)
		_cgo4 := MtlBufPtr(g1)
		_cgo5 := MtlBufPtr(g2)
		_cgo6 := MtlBufPtr(m1)
		_cgo7 := MtlBufPtr(m2)
		_cgo8 := MtlBufPtr(v1)
		_cgo9 := MtlBufPtr(v2)
		_cgo10 := mask.BufPtr()
		_cgo11 := MtlBufPtr(delta1)
		_cgo12 := MtlBufPtr(delta2)
		var _cgo13 C.float = C.float(lr)
		var _cgo14 C.float = C.float(beta1)
		var _cgo15 C.float = C.float(beta2)
		var _cgo16 C.float = C.float(bc1)
		var _cgo17 C.float = C.float(bc2)
		var _cgo18 C.float = C.float(eps)
		var _cgo19 C.float = C.float(wd)
		var _cgo20 C.float = C.float(backbone1)
		var _cgo21 C.float = C.float(glyco1)
		var _cgo22 C.float = C.float(hbond1)
		var _cgo23 C.float = C.float(hbond2)
		var _cgo24 C.float = C.float(glyco2)
		var _cgo25 C.float = C.float(backbone2)
		var _cgo26 C.float = C.float(bondStrength)
		var _cgo27 C.int = C.int(n)
		var _cgo28 C.int = C.int(cols)
		_cgo29 := MtlBufPtr(live1)
		_cgo30 := MtlBufPtr(live2)
		_cgo31 := MtlBufPtr(clipBuf)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		_cgoCheckPointer(_cgo3, nil)
		_cgoCheckPointer(_cgo4, nil)
		_cgoCheckPointer(_cgo5, nil)
		_cgoCheckPointer(_cgo6, nil)
		_cgoCheckPointer(_cgo7, nil)
		_cgoCheckPointer(_cgo8, nil)
		_cgoCheckPointer(_cgo9, nil)
		_cgoCheckPointer(_cgo10, nil)
		_cgoCheckPointer(_cgo11, nil)
		_cgoCheckPointer(_cgo12, nil)
		_cgoCheckPointer(_cgo29, nil)
		_cgoCheckPointer(_cgo30, nil)
		_cgoCheckPointer(_cgo31, nil)
		C.mtl_fused_needle_paired(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5, _cgo6, _cgo7, _cgo8, _cgo9, _cgo10, _cgo11, _cgo12, _cgo13, _cgo14, _cgo15, _cgo16, _cgo17, _cgo18, _cgo19, _cgo20, _cgo21, _cgo22, _cgo23, _cgo24, _cgo25, _cgo26, _cgo27, _cgo28, _cgo29, _cgo30, _cgo31)
	}()
}

type HotRowMask struct {
	buf    C.MTLBufferRef
	nRows  int
	shared []int8
}

func (m *Metal) NewHotRowMask(nRows int) *HotRowMask {
	buf := (C.mtl_alloc)(C.size_t(nRows))
	func() {
		_cgo0 := buf
		var _cgo1 C.size_t = C.size_t(nRows)
		_cgoCheckPointer(_cgo0, nil)
		C.mtl_zero(_cgo0, _cgo1)
	}()
	ptr := func() unsafe.Pointer { _cgo0 := buf; _cgoCheckPointer(_cgo0, nil); return C.mtl_shared_ptr(_cgo0) }()
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
		func() { _cgo0 := h.buf; _cgoCheckPointer(_cgo0, nil); C.mtl_free(_cgo0) }()
		h.buf = nil
		h.shared = nil
	}
}
