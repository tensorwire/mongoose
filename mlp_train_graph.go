//go:build linux && cgo

package mongoose

/*
#include <cuda_runtime.h>
#include <cublas_v2.h>

extern cublasHandle_t tw_cublas_handle;
extern void* tw_gpu_alloc(size_t bytes);
extern void* tw_get_kernel_lib();

static cudaStream_t mlp_graph_stream = NULL;
static cudaGraphExec_t mlp_graph_exec = NULL;

static cudaStream_t mlp_create_stream() {
    cudaStream_t s;
    if (cudaStreamCreate(&s) != cudaSuccess) return NULL;
    return s;
}

static void mlp_set_cublas_stream(cudaStream_t s) {
    cublasSetStream(tw_cublas_handle, s);
}

static void mlp_restore_cublas_stream() {
    cublasSetStream(tw_cublas_handle, 0);
}

static int mlp_begin_capture(cudaStream_t s) {
    cudaGetLastError();
    return (int)cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed);
}

static void* mlp_end_capture(cudaStream_t s) {
    cudaGraph_t graph;
    cudaError_t err = cudaStreamEndCapture(s, &graph);
    if (err != cudaSuccess) return NULL;
    cudaGraphExec_t exec;
    err = cudaGraphInstantiate(&exec, graph, 0);
    cudaGraphDestroy(graph);
    if (err != cudaSuccess) return NULL;
    return (void*)exec;
}

static int mlp_launch_graph(void* exec, cudaStream_t s) {
    return (int)cudaGraphLaunch((cudaGraphExec_t)exec, s);
}

static int mlp_stream_sync(cudaStream_t s) {
    return (int)cudaStreamSynchronize(s);
}

static void mlp_destroy_graph(void* exec) {
    if (exec) cudaGraphExecDestroy((cudaGraphExec_t)exec);
}

// Device-pointer sgemm on a specific stream (for graph capture)
static int mlp_sgemm_transB_stream(const float* dA, const float* dB, float* dC,
    int m, int k, int n, cudaStream_t stream) {
    cublasSetStream(tw_cublas_handle, stream);
    float alpha = 1.0f, beta = 0.0f;
    int ret = (int)cublasGemmEx(tw_cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
        &alpha, dB, CUDA_R_32F, k, dA, CUDA_R_32F, k,
        &beta, dC, CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return ret;
}

// C = A^T @ B on stream (for weight gradient: dW = dOut^T @ input)
// Row-major: A=[m,k] (dOut=[B,OutDim]), B=[m,n] (input=[B,InDim]), C=[k,n] (dW=[OutDim,InDim])
// Col-major: A_col=[k,m], B_col=[n,m], C_col=[n,k]
// cuBLAS: C_col = B_col @ A_col^T → GemmEx(OP_N, OP_T, n, k, m, B, lda=n, A, ldb=k, C, ldc=n)
static int mlp_sgemm_transA_stream(const float* dA, const float* dB, float* dC,
    int m, int k, int n, cudaStream_t stream) {
    cublasSetStream(tw_cublas_handle, stream);
    float alpha = 1.0f, beta = 0.0f;
    int ret = (int)cublasGemmEx(tw_cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_T, n, k, m,
        &alpha, dB, CUDA_R_32F, n, dA, CUDA_R_32F, k,
        &beta, dC, CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return ret;
}

// C = A @ B on stream (for input gradient: dInput = dOut @ W)
static int mlp_sgemm_stream(const float* dA, const float* dB, float* dC,
    int m, int k, int n, cudaStream_t stream) {
    cublasSetStream(tw_cublas_handle, stream);
    float alpha = 1.0f, beta = 0.0f;
    int ret = (int)cublasGemmEx(tw_cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
        &alpha, dB, CUDA_R_32F, n, dA, CUDA_R_32F, k,
        &beta, dC, CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return ret;
}

// Load kernel symbols via dlsym from the kernel library.
// tw_get_kernel_lib() is defined in kernels.go and returns the dlopen'd handle.
#include <dlfcn.h>

typedef void (*fn_void_ff_ii_s)(float*, const float*, int, int, void*);
typedef void (*fn_void_cff_ii_s)(const float*, float*, int, int, void*);
typedef void (*fn_bn_fwd)(const float*, float*, const float*, const float*, float*, float*, float*, float*, int, int, float, float, void*);
typedef void (*fn_bn_bwd)(const float*, const float*, const float*, const float*, const float*, float*, float*, float*, int, int, void*);
typedef void (*fn_bce)(const float*, const float*, float*, float*, int, void*);
typedef void (*fn_rmean)(const float*, float*, int, void*);
typedef void (*fn_drop_f)(float*, float*, unsigned long long, unsigned long long, float, int, void*);
typedef void (*fn_drop_b)(float*, const float*, int, void*);
typedef void (*fn_adamw_k)(float*, const float*, float*, float*, float, float, float, float, float, float, int, void*);
typedef void (*fn_relu_k)(float*, int, void*);
typedef void (*fn_relu_bk)(float*, const float*, const float*, int, void*);

static fn_void_ff_ii_s _mlp_bias_add = NULL;
static fn_void_cff_ii_s _mlp_reduce_mean_rows = NULL;
static fn_bn_fwd _mlp_bn_fwd = NULL;
static fn_bn_bwd _mlp_bn_bwd = NULL;
static fn_bce _mlp_bce = NULL;
static fn_rmean _mlp_rmean = NULL;
static fn_drop_f _mlp_drop_f = NULL;
static fn_drop_b _mlp_drop_b = NULL;
static fn_adamw_k _mlp_adamw = NULL;
static fn_relu_k _mlp_relu = NULL;
static fn_relu_bk _mlp_relu_bk = NULL;

static int mlp_load_kernels() {
    void* lib = tw_get_kernel_lib();
    if (!lib) return 0;
    _mlp_bias_add = (fn_void_ff_ii_s)dlsym(lib, "mongoose_bias_add");
    _mlp_reduce_mean_rows = (fn_void_cff_ii_s)dlsym(lib, "mongoose_reduce_mean_rows");
    _mlp_bn_fwd = (fn_bn_fwd)dlsym(lib, "mongoose_batchnorm_fwd");
    _mlp_bn_bwd = (fn_bn_bwd)dlsym(lib, "mongoose_batchnorm_bwd");
    _mlp_bce = (fn_bce)dlsym(lib, "mongoose_bce_with_logits");
    _mlp_rmean = (fn_rmean)dlsym(lib, "mongoose_reduce_mean");
    _mlp_drop_f = (fn_drop_f)dlsym(lib, "mongoose_dropout_fwd");
    _mlp_drop_b = (fn_drop_b)dlsym(lib, "mongoose_dropout_bwd");
    _mlp_adamw = (fn_adamw_k)dlsym(lib, "mongoose_adamw");
    _mlp_relu = (fn_relu_k)dlsym(lib, "mongoose_relu");
    _mlp_relu_bk = (fn_relu_bk)dlsym(lib, "mongoose_relu_backward");
    return (_mlp_bias_add && _mlp_bn_fwd && _mlp_bce && _mlp_adamw && _mlp_relu) ? 1 : 0;
}

static void mlp_bias_add(float* o, const float* b, int B, int D, cudaStream_t s) { if(_mlp_bias_add) _mlp_bias_add(o,b,B,D,s); }
static void mlp_reduce_mean_rows(const float* i, float* o, int B, int D, cudaStream_t s) { if(_mlp_reduce_mean_rows) _mlp_reduce_mean_rows(i,o,B,D,s); }
static void mlp_batchnorm_fwd(const float* x, float* o, const float* g, const float* b,
    float* rm, float* rv, float* sm, float* si, int B, int D, float mom, float eps, cudaStream_t s) {
    if(_mlp_bn_fwd) _mlp_bn_fwd(x,o,g,b,rm,rv,sm,si,B,D,mom,eps,s);
}
static void mlp_batchnorm_bwd(const float* dO, const float* x, const float* sm, const float* si,
    const float* g, float* dx, float* dg, float* db, int B, int D, cudaStream_t s) {
    if(_mlp_bn_bwd) _mlp_bn_bwd(dO,x,sm,si,g,dx,dg,db,B,D,s);
}
static void mlp_bce_with_logits(const float* l, const float* t, float* g, float* lo, int B, cudaStream_t s) { if(_mlp_bce) _mlp_bce(l,t,g,lo,B,s); }
static void mlp_reduce_mean(const float* i, float* o, int n, cudaStream_t s) { if(_mlp_rmean) _mlp_rmean(i,o,n,s); }
static void mlp_dropout_fwd(float* x, float* m, unsigned long long seed, unsigned long long ctr, float p, int n, cudaStream_t s) { if(_mlp_drop_f) _mlp_drop_f(x,m,seed,ctr,p,n,s); }
static void mlp_dropout_bwd(float* dx, const float* m, int n, cudaStream_t s) { if(_mlp_drop_b) _mlp_drop_b(dx,m,n,s); }
static void mlp_adamw(float* p, const float* g, float* m, float* v,
    float lr, float wd, float b1, float b2, float bc1, float bc2, int n, cudaStream_t s) {
    if(_mlp_adamw) _mlp_adamw(p,g,m,v,lr,wd,b1,b2,bc1,bc2,n,s);
}
static void mlp_relu(void* x, int n, cudaStream_t s) { if(_mlp_relu) _mlp_relu((float*)x,n,s); }
static void mlp_relu_backward(float* o, const float* dO, const float* i, int n, cudaStream_t s) { if(_mlp_relu_bk) _mlp_relu_bk(o,dO,i,n,s); }

// Async H2D memcpy on stream
static void mlp_upload_async(void* dst, const void* src, int bytes, cudaStream_t s) {
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, s);
}

// Async D2H memcpy on stream
static void mlp_download_async(void* dst, const void* src, int bytes, cudaStream_t s) {
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, s);
}

// Allocate pinned host memory
static void* mlp_alloc_pinned(int bytes) {
    void* p;
    if (cudaHostAlloc(&p, bytes, cudaHostAllocDefault) != cudaSuccess) return NULL;
    return p;
}

static void mlp_free_pinned(void* p) {
    if (p) cudaFreeHost(p);
}
*/
import "C"
import (
	"fmt"
	"log"
	"math"
	"unsafe"
)

// MLPTrainGraph holds a captured CUDA graph for fused MLP training.
// One capture → many replays. Zero CPU round-trips in the hot loop.
type MLPTrainGraph struct {
	eng       *CUDA
	mlp       *MLP
	stream    C.cudaStream_t
	graphExec unsafe.Pointer
	captured  bool

	batchSize  int
	nFeatures  int
	nLayers    int

	// Pre-allocated GPU buffers
	inputBuf  unsafe.Pointer // [batchSize, nFeatures]
	targetBuf unsafe.Pointer // [batchSize]

	// Per-layer forward buffers
	layerOut      []unsafe.Pointer // [batchSize, outDim] — output of linear+bias
	layerBNOut    []unsafe.Pointer // [batchSize, outDim] — after BN
	layerPreReLU  []unsafe.Pointer // [batchSize, outDim] — BN output before ReLU (for backward)
	layerMask     []unsafe.Pointer // [batchSize, outDim] — dropout masks
	saveMean      []unsafe.Pointer // [outDim]
	saveInvStd    []unsafe.Pointer // [outDim]

	// Per-layer backward buffers
	layerDX     []unsafe.Pointer // [batchSize, inDim] — input gradient
	layerDW     []unsafe.Pointer // [outDim, inDim] — weight gradient
	layerDB     []unsafe.Pointer // [outDim] — bias gradient
	layerDGamma []unsafe.Pointer // [outDim] — BN gamma gradient
	layerDBeta  []unsafe.Pointer // [outDim] — BN beta gradient

	// Loss buffers
	gradBuf   unsafe.Pointer // [batchSize] — BCE gradient
	lossBuf   unsafe.Pointer // [batchSize] — per-sample loss
	lossScalar unsafe.Pointer // [1] — reduced loss (device)

	// Pinned host staging
	pinnedInput  unsafe.Pointer // [batchSize * nFeatures]
	pinnedTarget unsafe.Pointer // [batchSize]
	pinnedLoss   unsafe.Pointer // [1]

	// Adam optimizer state (GPU)
	mW, vW []unsafe.Pointer // per-layer weight momentum/velocity
	mB, vB []unsafe.Pointer // per-layer bias momentum/velocity
	mG, vG []unsafe.Pointer // per-layer BN gamma momentum/velocity
	mBt, vBt []unsafe.Pointer // per-layer BN beta momentum/velocity

	dropoutCounter uint64
	step           int
	captureLR      float32
	captureBC1     float32
	captureBC2     float32
}

func gpuAlloc(n int) unsafe.Pointer {
	return C.tw_gpu_alloc(C.size_t(n * 4))
}

func gpuZero(ptr unsafe.Pointer, n int) {
	C.cudaMemset(ptr, 0, C.size_t(n*4))
}

// NewMLPTrainGraph pre-allocates all GPU buffers for fused training.
func NewMLPTrainGraph(eng *CUDA, mlp *MLP, batchSize int) *MLPTrainGraph {
	nLayers := len(mlp.Layers)
	nFeatures := mlp.Layers[0].InDim

	g := &MLPTrainGraph{
		eng:       eng,
		mlp:       mlp,
		batchSize: batchSize,
		nFeatures: nFeatures,
		nLayers:   nLayers,
	}

	g.stream = C.mlp_create_stream()
	if g.stream == nil {
		log.Fatal("[MLPTrainGraph] failed to create CUDA stream")
	}
	if C.mlp_load_kernels() == 0 {
		log.Fatal("[MLPTrainGraph] MLP training kernels not loaded — compile kernels first")
	}

	// Staging buffers
	g.inputBuf = gpuAlloc(batchSize * nFeatures)
	g.targetBuf = gpuAlloc(batchSize)
	g.gradBuf = gpuAlloc(batchSize)
	g.lossBuf = gpuAlloc(batchSize)
	g.lossScalar = gpuAlloc(1)
	gpuZero(g.inputBuf, batchSize*nFeatures)
	gpuZero(g.targetBuf, batchSize)
	gpuZero(g.gradBuf, batchSize)
	gpuZero(g.lossBuf, batchSize)
	gpuZero(g.lossScalar, 1)

	// Pinned host memory for async transfers
	g.pinnedInput = C.mlp_alloc_pinned(C.int(batchSize * nFeatures * 4))
	g.pinnedTarget = C.mlp_alloc_pinned(C.int(batchSize * 4))
	g.pinnedLoss = C.mlp_alloc_pinned(C.int(4))

	// Per-layer buffers
	g.layerOut = make([]unsafe.Pointer, nLayers)
	g.layerBNOut = make([]unsafe.Pointer, nLayers)
	g.layerPreReLU = make([]unsafe.Pointer, nLayers)
	g.layerMask = make([]unsafe.Pointer, nLayers)
	g.saveMean = make([]unsafe.Pointer, nLayers)
	g.saveInvStd = make([]unsafe.Pointer, nLayers)
	g.layerDX = make([]unsafe.Pointer, nLayers)
	g.layerDW = make([]unsafe.Pointer, nLayers)
	g.layerDB = make([]unsafe.Pointer, nLayers)
	g.layerDGamma = make([]unsafe.Pointer, nLayers)
	g.layerDBeta = make([]unsafe.Pointer, nLayers)
	g.mW = make([]unsafe.Pointer, nLayers)
	g.vW = make([]unsafe.Pointer, nLayers)
	g.mB = make([]unsafe.Pointer, nLayers)
	g.vB = make([]unsafe.Pointer, nLayers)
	g.mG = make([]unsafe.Pointer, nLayers)
	g.vG = make([]unsafe.Pointer, nLayers)
	g.mBt = make([]unsafe.Pointer, nLayers)
	g.vBt = make([]unsafe.Pointer, nLayers)

	for i, l := range mlp.Layers {
		n := batchSize * l.OutDim
		g.layerOut[i] = gpuAlloc(n)
		g.layerBNOut[i] = gpuAlloc(n)
		g.layerPreReLU[i] = gpuAlloc(n)
		g.layerMask[i] = gpuAlloc(n)
		g.saveMean[i] = gpuAlloc(l.OutDim)
		g.saveInvStd[i] = gpuAlloc(l.OutDim)
		g.layerDX[i] = gpuAlloc(batchSize * l.InDim)
		g.layerDW[i] = gpuAlloc(l.OutDim * l.InDim)
		g.layerDB[i] = gpuAlloc(l.OutDim)
		g.layerDGamma[i] = gpuAlloc(l.OutDim)
		g.layerDBeta[i] = gpuAlloc(l.OutDim)
		// Zero ALL working buffers — cudaMalloc does not zero memory
		gpuZero(g.layerOut[i], n)
		gpuZero(g.layerBNOut[i], n)
		gpuZero(g.layerPreReLU[i], n)
		gpuZero(g.layerMask[i], n)
		gpuZero(g.saveMean[i], l.OutDim)
		gpuZero(g.saveInvStd[i], l.OutDim)
		gpuZero(g.layerDX[i], batchSize*l.InDim)
		gpuZero(g.layerDW[i], l.OutDim*l.InDim)
		gpuZero(g.layerDB[i], l.OutDim)
		gpuZero(g.layerDGamma[i], l.OutDim)
		gpuZero(g.layerDBeta[i], l.OutDim)
		g.mW[i] = gpuAlloc(l.OutDim * l.InDim)
		g.vW[i] = gpuAlloc(l.OutDim * l.InDim)
		g.mB[i] = gpuAlloc(l.OutDim)
		g.vB[i] = gpuAlloc(l.OutDim)
		gpuZero(g.mW[i], l.OutDim*l.InDim)
		gpuZero(g.vW[i], l.OutDim*l.InDim)
		gpuZero(g.mB[i], l.OutDim)
		gpuZero(g.vB[i], l.OutDim)
		if l.BNGamma != nil {
			g.mG[i] = gpuAlloc(l.OutDim)
			g.vG[i] = gpuAlloc(l.OutDim)
			g.mBt[i] = gpuAlloc(l.OutDim)
			g.vBt[i] = gpuAlloc(l.OutDim)
			gpuZero(g.mG[i], l.OutDim)
			gpuZero(g.vG[i], l.OutDim)
			gpuZero(g.mBt[i], l.OutDim)
			gpuZero(g.vBt[i], l.OutDim)
		}
	}

	// Upload initial weights to GPU layer buffers
	for _, l := range mlp.Layers {
		if l.gW == nil {
			mlp.ToGPU(AsTensorEngine(eng))
			break
		}
	}

	// Ensure all allocations and uploads are complete before any kernel launch
	C.cudaDeviceSynchronize()

	log.Printf("[MLPTrainGraph] allocated %d layers, batch=%d, features=%d", nLayers, batchSize, nFeatures)
	return g
}

// CaptureWithParams records the full training step as a CUDA graph with specific optimizer params.
// Call once initially, then call RecaptureWithParams each epoch to update lr/bias correction.
func (g *MLPTrainGraph) CaptureWithParams(lr, bc1, bc2 float32) {
	s := g.stream
	g.captureLR = lr
	g.captureBC1 = bc1
	g.captureBC2 = bc2

	// Upload dummy data so buffers are populated
	dummy := make([]float32, g.batchSize*g.nFeatures)
	dummyT := make([]float32, g.batchSize)
	copy((*[1 << 30]float32)(g.pinnedInput)[:len(dummy)], dummy)
	copy((*[1 << 30]float32)(g.pinnedTarget)[:len(dummyT)], dummyT)
	C.mlp_upload_async(g.inputBuf, g.pinnedInput, C.int(len(dummy)*4), s)
	C.mlp_upload_async(g.targetBuf, g.pinnedTarget, C.int(len(dummyT)*4), s)
	C.mlp_stream_sync(s)

	// Destroy previous graph if re-capturing
	if g.graphExec != nil {
		C.mlp_destroy_graph(g.graphExec)
		g.graphExec = nil
	}

	// Set cuBLAS to capture stream
	C.mlp_set_cublas_stream(s)

	// Begin capture
	ret := C.mlp_begin_capture(s)
	if ret != 0 {
		log.Fatal("[MLPTrainGraph] cudaStreamBeginCapture failed")
	}

	// Record the full step (forward + backward + optimizer) into the graph
	g.recordStep(s)

	// End capture
	exec := C.mlp_end_capture(s)
	if exec == nil {
		log.Fatal("[MLPTrainGraph] cudaStreamEndCapture/Instantiate failed")
	}
	g.graphExec = exec
	g.captured = true

	// Restore cuBLAS to default stream
	C.mlp_restore_cublas_stream()
}

// Capture is a convenience wrapper with default lr.
func (g *MLPTrainGraph) Capture() {
	g.CaptureWithParams(0.0003, 0.1, 0.001)
	log.Printf("[MLPTrainGraph] graph captured: %d layers, batch=%d", g.nLayers, g.batchSize)
}

// LaunchGraph replays the captured graph. Call UploadBatch first.
// Returns the loss value.
func (g *MLPTrainGraph) LaunchGraph(lr float32) float32 {
	if !g.captured {
		log.Fatal("[MLPTrainGraph] call Capture() before LaunchGraph()")
	}
	s := g.stream

	// The graph reads from inputBuf/targetBuf which were filled by UploadBatch.
	// Launch the captured graph
	ret := C.mlp_launch_graph(g.graphExec, s)
	if ret != 0 {
		log.Printf("[MLPTrainGraph] graph launch failed: %d", ret)
		return 0
	}

	// Download loss (async D2H was NOT in the graph — do it now)
	C.mlp_download_async(g.pinnedLoss, g.lossScalar, C.int(4), s)
	C.mlp_stream_sync(s)

	return *(*float32)(g.pinnedLoss)
}

// UploadBatch copies features and targets to pinned memory, then async H2D.
func (g *MLPTrainGraph) UploadBatch(features, targets []float32) {
	// Copy to pinned memory
	copy((*[1 << 30]float32)(g.pinnedInput)[:len(features)], features)
	copy((*[1 << 30]float32)(g.pinnedTarget)[:len(targets)], targets)

	// Async upload to GPU
	C.mlp_upload_async(g.inputBuf, g.pinnedInput, C.int(len(features)*4), g.stream)
	C.mlp_upload_async(g.targetBuf, g.pinnedTarget, C.int(len(targets)*4), g.stream)
}

// recordStep records the kernel sequence for graph capture or direct execution.
// When called during capture, kernels are recorded but not executed.
// When called directly (RunStep), kernels execute immediately on the stream.
func (g *MLPTrainGraph) recordStep(s C.cudaStream_t) {
	B := g.batchSize
	mlp := g.mlp

	bc1 := g.captureBC1
	bc2 := g.captureBC2
	lr := g.captureLR

	// === FORWARD ===
	curInput := g.inputBuf

	for i := range mlp.Layers {
		l := &mlp.Layers[i]
		isLast := i == len(mlp.Layers)-1
		n := B * l.OutDim

		// Linear: out = input @ W^T
		C.mlp_sgemm_transB_stream(
			(*C.float)(curInput), (*C.float)(l.gW.DevicePtr()),
			(*C.float)(g.layerOut[i]),
			C.int(B), C.int(l.InDim), C.int(l.OutDim), s)

		// Bias add
		C.mlp_bias_add((*C.float)(g.layerOut[i]), (*C.float)(l.gB.DevicePtr()), C.int(B), C.int(l.OutDim), s)

		// BatchNorm
		if l.BNGamma != nil && l.gBNGamma != nil {
			C.mlp_batchnorm_fwd(
				(*C.float)(g.layerOut[i]), (*C.float)(g.layerBNOut[i]),
				(*C.float)(l.gBNGamma.DevicePtr()), (*C.float)(l.gBNBeta.DevicePtr()),
				(*C.float)(l.gRunMean.DevicePtr()), (*C.float)(l.gRunVar.DevicePtr()),
				(*C.float)(g.saveMean[i]), (*C.float)(g.saveInvStd[i]),
				C.int(B), C.int(l.OutDim), C.float(mlp.Config.BNMomentum), C.float(1e-5), s)
		} else {
			C.cudaMemcpyAsync(g.layerBNOut[i], g.layerOut[i], C.size_t(n*4), C.cudaMemcpyDeviceToDevice, s)
		}

		// Save pre-ReLU BN output for backward (BUG 6 fix: ReLU backward needs post-BN, pre-ReLU values)
		if !isLast {
			C.cudaMemcpyAsync(g.layerPreReLU[i], g.layerBNOut[i], C.size_t(n*4), C.cudaMemcpyDeviceToDevice, s)
		}

		// ReLU (hidden layers only) — modifies layerBNOut in-place
		if !isLast {
			C.mlp_relu(g.layerBNOut[i], C.int(n), s)
		}

		// Dropout (hidden layers, training)
		if !isLast && mlp.Config.Dropout > 0 {
			C.mlp_dropout_fwd((*C.float)(g.layerBNOut[i]), (*C.float)(g.layerMask[i]),
				C.ulonglong(42), C.ulonglong(g.dropoutCounter+uint64(i)),
				C.float(mlp.Config.Dropout), C.int(n), s)
		}

		curInput = g.layerBNOut[i]
	}

	// === LOSS ===
	lastOut := g.layerBNOut[g.nLayers-1]
	C.mlp_bce_with_logits((*C.float)(lastOut), (*C.float)(g.targetBuf),
		(*C.float)(g.gradBuf), (*C.float)(g.lossBuf), C.int(B), s)
	C.mlp_reduce_mean((*C.float)(g.lossBuf), (*C.float)(g.lossScalar), C.int(B), s)

	// === BACKWARD ===
	curGrad := g.gradBuf

	for i := g.nLayers - 1; i >= 0; i-- {
		l := &mlp.Layers[i]
		isLast := i == g.nLayers-1
		n := B * l.OutDim

		// Dropout backward
		if !isLast && mlp.Config.Dropout > 0 {
			C.mlp_dropout_bwd((*C.float)(curGrad), (*C.float)(g.layerMask[i]), C.int(n), s)
		}

		// ReLU backward: use layerPreReLU (post-BN, pre-ReLU) for correct mask
		if !isLast {
			C.mlp_relu_backward((*C.float)(curGrad), (*C.float)(curGrad), (*C.float)(g.layerPreReLU[i]), C.int(n), s)
		}

		// BatchNorm backward — dGamma/dBeta go to SEPARATE buffers (not saveMean/saveInvStd)
		if l.BNGamma != nil && l.gBNGamma != nil {
			C.mlp_batchnorm_bwd(
				(*C.float)(curGrad), (*C.float)(g.layerOut[i]),
				(*C.float)(g.saveMean[i]), (*C.float)(g.saveInvStd[i]),
				(*C.float)(l.gBNGamma.DevicePtr()),
				(*C.float)(g.layerDX[i]),
				(*C.float)(g.layerDGamma[i]),
				(*C.float)(g.layerDBeta[i]),
				C.int(B), C.int(l.OutDim), s)
			curGrad = g.layerDX[i]
		}

		// Weight gradient: dW = dOut^T @ input
		var inputPtr unsafe.Pointer
		if i == 0 {
			inputPtr = g.inputBuf
		} else {
			inputPtr = g.layerBNOut[i-1]
		}
		C.mlp_sgemm_transA_stream(
			(*C.float)(curGrad), (*C.float)(inputPtr),
			(*C.float)(g.layerDW[i]),
			C.int(B), C.int(l.OutDim), C.int(l.InDim), s)

		// Bias gradient: dB = mean(dOut, axis=0)
		C.mlp_reduce_mean_rows((*C.float)(curGrad), (*C.float)(g.layerDB[i]), C.int(B), C.int(l.OutDim), s)

		// Input gradient: dInput = dOut @ W
		if i > 0 {
			C.mlp_sgemm_stream(
				(*C.float)(curGrad), (*C.float)(l.gW.DevicePtr()),
				(*C.float)(g.layerDX[i]),
				C.int(B), C.int(l.OutDim), C.int(l.InDim), s)
			curGrad = g.layerDX[i]
		}

		// === OPTIMIZER (AdamW) ===
		wd := C.float(mlp.Config.WeightDecay)
		if wd == 0 {
			wd = C.float(0.0005)
		}

		nW := l.OutDim * l.InDim
		C.mlp_adamw((*C.float)(l.gW.DevicePtr()), (*C.float)(g.layerDW[i]),
			(*C.float)(g.mW[i]), (*C.float)(g.vW[i]),
			C.float(lr), wd, C.float(0.9), C.float(0.999),
			C.float(bc1), C.float(bc2), C.int(nW), s)

		C.mlp_adamw((*C.float)(l.gB.DevicePtr()), (*C.float)(g.layerDB[i]),
			(*C.float)(g.mB[i]), (*C.float)(g.vB[i]),
			C.float(lr), wd, C.float(0.9), C.float(0.999),
			C.float(bc1), C.float(bc2), C.int(l.OutDim), s)

		// BN gamma/beta optimizer (BUG 5 fix: these were never updated)
		if l.BNGamma != nil && l.gBNGamma != nil {
			C.mlp_adamw((*C.float)(l.gBNGamma.DevicePtr()), (*C.float)(g.layerDGamma[i]),
				(*C.float)(g.mG[i]), (*C.float)(g.vG[i]),
				C.float(lr), C.float(0), C.float(0.9), C.float(0.999),
				C.float(bc1), C.float(bc2), C.int(l.OutDim), s)

			C.mlp_adamw((*C.float)(l.gBNBeta.DevicePtr()), (*C.float)(g.layerDBeta[i]),
				(*C.float)(g.mBt[i]), (*C.float)(g.vBt[i]),
				C.float(lr), C.float(0), C.float(0.9), C.float(0.999),
				C.float(bc1), C.float(bc2), C.int(l.OutDim), s)
		}
	}

}

// RunStep executes a full forward + backward + optimizer step directly on stream (no graph).
// This is the recommended path — it avoids the frozen-counter problem with captured graphs.
func (g *MLPTrainGraph) RunStep(lr float32) float32 {
	s := g.stream
	g.step++
	g.dropoutCounter++
	bc1 := 1.0 - float32(math.Pow(0.9, float64(g.step)))
	bc2 := 1.0 - float32(math.Pow(0.999, float64(g.step)))
	g.captureLR = lr
	g.captureBC1 = bc1
	g.captureBC2 = bc2
	C.mlp_set_cublas_stream(s)
	g.recordStep(s)
	C.mlp_download_async(g.pinnedLoss, g.lossScalar, C.int(4), s)
	C.mlp_stream_sync(s)
	C.mlp_restore_cublas_stream()
	return *(*float32)(g.pinnedLoss)
}

// DiagnosticStep runs one forward pass with diagnostics, printing where NaN first appears.
// Call after UploadBatch. Does NOT run backward or optimizer.
func (g *MLPTrainGraph) DiagnosticStep() {
	s := g.stream
	B := g.batchSize
	mlp := g.mlp

	C.mlp_set_cublas_stream(s)

	// Check input
	g.dumpBuf("input", g.inputBuf, B*g.nFeatures, s)

	curInput := g.inputBuf
	for i := range mlp.Layers {
		l := &mlp.Layers[i]
		isLast := i == len(mlp.Layers)-1
		n := B * l.OutDim

		// Check weights
		g.dumpBuf(fmt.Sprintf("L%d.W", i), l.gW.DevicePtr(), l.OutDim*l.InDim, s)
		g.dumpBuf(fmt.Sprintf("L%d.B", i), l.gB.DevicePtr(), l.OutDim, s)

		// Matmul
		C.mlp_sgemm_transB_stream(
			(*C.float)(curInput), (*C.float)(l.gW.DevicePtr()),
			(*C.float)(g.layerOut[i]),
			C.int(B), C.int(l.InDim), C.int(l.OutDim), s)
		g.dumpBuf(fmt.Sprintf("L%d.matmul", i), g.layerOut[i], n, s)

		// Bias
		C.mlp_bias_add((*C.float)(g.layerOut[i]), (*C.float)(l.gB.DevicePtr()), C.int(B), C.int(l.OutDim), s)
		g.dumpBuf(fmt.Sprintf("L%d.bias", i), g.layerOut[i], n, s)

		// BN
		if l.BNGamma != nil && l.gBNGamma != nil {
			C.mlp_batchnorm_fwd(
				(*C.float)(g.layerOut[i]), (*C.float)(g.layerBNOut[i]),
				(*C.float)(l.gBNGamma.DevicePtr()), (*C.float)(l.gBNBeta.DevicePtr()),
				(*C.float)(l.gRunMean.DevicePtr()), (*C.float)(l.gRunVar.DevicePtr()),
				(*C.float)(g.saveMean[i]), (*C.float)(g.saveInvStd[i]),
				C.int(B), C.int(l.OutDim), C.float(mlp.Config.BNMomentum), C.float(1e-5), s)
			g.dumpBuf(fmt.Sprintf("L%d.bn", i), g.layerBNOut[i], n, s)
		} else {
			C.cudaMemcpyAsync(g.layerBNOut[i], g.layerOut[i], C.size_t(n*4), C.cudaMemcpyDeviceToDevice, s)
		}

		if !isLast {
			C.mlp_relu(g.layerBNOut[i], C.int(n), s)
			g.dumpBuf(fmt.Sprintf("L%d.relu", i), g.layerBNOut[i], n, s)
		}

		curInput = g.layerBNOut[i]
	}

	// Loss
	lastOut := g.layerBNOut[g.nLayers-1]
	C.mlp_bce_with_logits((*C.float)(lastOut), (*C.float)(g.targetBuf),
		(*C.float)(g.gradBuf), (*C.float)(g.lossBuf), C.int(B), s)
	g.dumpBuf("bce.grad", g.gradBuf, B, s)
	g.dumpBuf("bce.loss", g.lossBuf, B, s)

	C.mlp_reduce_mean((*C.float)(g.lossBuf), (*C.float)(g.lossScalar), C.int(B), s)
	g.dumpBuf("loss_scalar", g.lossScalar, 1, s)

	// === BACKWARD ===
	log.Println("[DIAG] --- BACKWARD ---")
	curGrad := g.gradBuf

	bc1 := float32(1.0 - math.Pow(0.9, 1.0))
	bc2 := float32(1.0 - math.Pow(0.999, 1.0))
	lr := g.captureLR
	if lr == 0 {
		lr = 0.0003
	}
	log.Printf("[DIAG] optimizer params: lr=%.6f bc1=%.6f bc2=%.6f", lr, bc1, bc2)

	for i := g.nLayers - 1; i >= 0; i-- {
		l := &mlp.Layers[i]
		isLast := i == g.nLayers-1
		n := B * l.OutDim

		g.dumpBuf(fmt.Sprintf("L%d.grad_in", i), curGrad, n, s)

		if !isLast && mlp.Config.Dropout > 0 {
			C.mlp_dropout_bwd((*C.float)(curGrad), (*C.float)(g.layerMask[i]), C.int(n), s)
			g.dumpBuf(fmt.Sprintf("L%d.drop_bwd", i), curGrad, n, s)
		}

		if !isLast {
			C.mlp_relu_backward((*C.float)(curGrad), (*C.float)(curGrad), (*C.float)(g.layerPreReLU[i]), C.int(n), s)
			g.dumpBuf(fmt.Sprintf("L%d.relu_bwd", i), curGrad, n, s)
		}

		if l.BNGamma != nil && l.gBNGamma != nil {
			C.mlp_batchnorm_bwd(
				(*C.float)(curGrad), (*C.float)(g.layerOut[i]),
				(*C.float)(g.saveMean[i]), (*C.float)(g.saveInvStd[i]),
				(*C.float)(l.gBNGamma.DevicePtr()),
				(*C.float)(g.layerDX[i]),
				(*C.float)(g.layerDGamma[i]),
				(*C.float)(g.layerDBeta[i]),
				C.int(B), C.int(l.OutDim), s)
			g.dumpBuf(fmt.Sprintf("L%d.bn_bwd_dx", i), g.layerDX[i], B*l.OutDim, s)
			g.dumpBuf(fmt.Sprintf("L%d.dGamma", i), g.layerDGamma[i], l.OutDim, s)
			g.dumpBuf(fmt.Sprintf("L%d.dBeta", i), g.layerDBeta[i], l.OutDim, s)
			curGrad = g.layerDX[i]
		}

		var inputPtr unsafe.Pointer
		if i == 0 {
			inputPtr = g.inputBuf
		} else {
			inputPtr = g.layerBNOut[i-1]
		}
		C.mlp_sgemm_transA_stream(
			(*C.float)(curGrad), (*C.float)(inputPtr),
			(*C.float)(g.layerDW[i]),
			C.int(B), C.int(l.OutDim), C.int(l.InDim), s)
		g.dumpBuf(fmt.Sprintf("L%d.dW", i), g.layerDW[i], l.OutDim*l.InDim, s)

		C.mlp_reduce_mean_rows((*C.float)(curGrad), (*C.float)(g.layerDB[i]), C.int(B), C.int(l.OutDim), s)
		g.dumpBuf(fmt.Sprintf("L%d.dB", i), g.layerDB[i], l.OutDim, s)

		if i > 0 {
			C.mlp_sgemm_stream(
				(*C.float)(curGrad), (*C.float)(l.gW.DevicePtr()),
				(*C.float)(g.layerDX[i]),
				C.int(B), C.int(l.OutDim), C.int(l.InDim), s)
			g.dumpBuf(fmt.Sprintf("L%d.dInput", i), g.layerDX[i], B*l.InDim, s)
			curGrad = g.layerDX[i]
		}

		// Optimizer
		wd := C.float(mlp.Config.WeightDecay)
		if wd == 0 {
			wd = C.float(0.0005)
		}

		nW := l.OutDim * l.InDim
		C.mlp_adamw((*C.float)(l.gW.DevicePtr()), (*C.float)(g.layerDW[i]),
			(*C.float)(g.mW[i]), (*C.float)(g.vW[i]),
			C.float(lr), wd, C.float(0.9), C.float(0.999),
			C.float(bc1), C.float(bc2), C.int(nW), s)
		g.dumpBuf(fmt.Sprintf("L%d.W_post", i), l.gW.DevicePtr(), nW, s)

		C.mlp_adamw((*C.float)(l.gB.DevicePtr()), (*C.float)(g.layerDB[i]),
			(*C.float)(g.mB[i]), (*C.float)(g.vB[i]),
			C.float(lr), wd, C.float(0.9), C.float(0.999),
			C.float(bc1), C.float(bc2), C.int(l.OutDim), s)
		g.dumpBuf(fmt.Sprintf("L%d.B_post", i), l.gB.DevicePtr(), l.OutDim, s)

		if l.BNGamma != nil && l.gBNGamma != nil {
			C.mlp_adamw((*C.float)(l.gBNGamma.DevicePtr()), (*C.float)(g.layerDGamma[i]),
				(*C.float)(g.mG[i]), (*C.float)(g.vG[i]),
				C.float(lr), C.float(0), C.float(0.9), C.float(0.999),
				C.float(bc1), C.float(bc2), C.int(l.OutDim), s)
			g.dumpBuf(fmt.Sprintf("L%d.Gam_post", i), l.gBNGamma.DevicePtr(), l.OutDim, s)

			C.mlp_adamw((*C.float)(l.gBNBeta.DevicePtr()), (*C.float)(g.layerDBeta[i]),
				(*C.float)(g.mBt[i]), (*C.float)(g.vBt[i]),
				C.float(lr), C.float(0), C.float(0.9), C.float(0.999),
				C.float(bc1), C.float(bc2), C.int(l.OutDim), s)
			g.dumpBuf(fmt.Sprintf("L%d.Bet_post", i), l.gBNBeta.DevicePtr(), l.OutDim, s)
		}
	}

	log.Println("[DIAG] --- DIAGNOSTIC COMPLETE ---")
	C.mlp_restore_cublas_stream()
}

func (g *MLPTrainGraph) dumpBuf(label string, ptr unsafe.Pointer, n int, s C.cudaStream_t) {
	host := make([]float32, n)
	C.mlp_download_async(unsafe.Pointer(&host[0]), ptr, C.int(n*4), s)
	C.mlp_stream_sync(s)

	nans, infs, zeros := 0, 0, 0
	var mn, mx float32
	mn, mx = host[0], host[0]
	for _, v := range host {
		if v != v { nans++ } // NaN != NaN
		if v > 1e30 || v < -1e30 { infs++ }
		if v == 0 { zeros++ }
		if v == v && v < mn { mn = v }
		if v == v && v > mx { mx = v }
	}
	status := "OK"
	if nans > 0 { status = "*** NaN ***" }
	if infs > 0 { status = "*** INF ***" }
	log.Printf("[DIAG] %-15s n=%-8d min=%12.6f max=%12.6f zeros=%d nans=%d infs=%d %s",
		label, n, mn, mx, zeros, nans, infs, status)
}

// DownloadWeights copies trained weights from GPU back to MLP CPU slices.
func (g *MLPTrainGraph) DownloadWeights() {
	g.mlp.ToCPU()
}

// Destroy frees all GPU resources.
func (g *MLPTrainGraph) Destroy() {
	if g.graphExec != nil {
		C.mlp_destroy_graph(g.graphExec)
	}
	if g.stream != nil {
		C.cudaStreamDestroy(g.stream)
	}
	C.mlp_free_pinned(g.pinnedInput)
	C.mlp_free_pinned(g.pinnedTarget)
	C.mlp_free_pinned(g.pinnedLoss)
	// GPU buffers freed when process exits (arena cleanup)
}

func init() {
	_ = fmt.Sprint // keep fmt import
}
