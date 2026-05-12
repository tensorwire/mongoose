//go:build darwin && cgo

package mongoose

/*
#cgo LDFLAGS: -framework Metal -framework Foundation -framework MetalPerformanceShaders
#include <stdlib.h>

typedef void* MTLBufferRef;

extern int mtl_mlp_init_path(const char* path);
extern int mtl_mlp_ready(void);
extern void mtl_mlp_train_step(
    int nLayers, const int* inDims, const int* outDims, const int* hasBN,
    void** W, void** bias,
    void** gamma, void** beta, void** runMean, void** runVar,
    void** bnMean, void** bnVar,
    void** mW, void** vW, void** mB, void** vB,
    void** mG, void** vG, void** mBt, void** vBt,
    void** act, void** preBN, void** preReLU, void** masks,
    void** dW, void** dB, void** dGamma, void** dBeta,
    void* gradBuf, void* lossBuf, void* lossScalar,
    void* input, void* targets,
    int B, float lr, float wd, float beta1Val, float beta2Val,
    float bc1, float bc2, float bnMomentum, float epsVal, float dropoutP,
    unsigned int dropSeed, unsigned int dropCounter
);

extern MTLBufferRef mtl_alloc(unsigned long bytes);
extern void* mtl_shared_ptr(MTLBufferRef buf);

// Test dispatch helpers
extern void mtl_mlp_gemm_bt(void* a, void* b, void* c, int M, int K, int N);
extern void mtl_mlp_gemm_tn(void* a, void* b, void* c, int M, int K, int N);
extern void mtl_mlp_gemm_nn(void* a, void* b, void* c, int M, int K, int N);
extern void mtl_mlp_bias_add(void* out, void* bias, int B, int D);
extern void mtl_mlp_bn_forward(void* x, void* mean, void* var, void* gamma, void* beta,
    void* runMean, void* runVar, int B, int D, float momentum);
extern void mtl_mlp_bn_backward(void* dOut, void* x, void* mean, void* var,
    void* gamma, void* dGamma, void* dBeta, int B, int D);
extern void mtl_mlp_bce(void* logits, void* targets, void* grad, void* lossBuf,
    void* lossScalar, int B);
extern void mtl_mlp_adamw_step(void* param, void* grad, void* m, void* v,
    float lr, float beta1, float beta2, float bc1, float bc2, float eps, float wd, int n);
extern void mtl_mlp_dropout_fwd(void* x, void* mask, int n, float p,
    unsigned int seed, unsigned int counter);
extern void mtl_mlp_dropout_bwd(void* dx, void* mask, int n);
*/
import "C"
import (
	"log"
	"math"
	"os"
	"path/filepath"
	"unsafe"
)

type MLPMetal struct {
	eng       *Metal
	mlp       *MLP
	batchSize int
	nLayers   int

	inDims  []C.int
	outDims []C.int
	hasBN   []C.int

	// Per-layer Metal buffer refs stored as unsafe.Pointer (= void* = MTLBufferRef)
	W, bias                          []unsafe.Pointer
	gamma, beta, runMean, runVar     []unsafe.Pointer
	bnMean, bnVar                    []unsafe.Pointer
	mW, vW, mB, vB                  []unsafe.Pointer
	mG, vG, mBt, vBt                []unsafe.Pointer
	act, preBN, preReLU, masks       []unsafe.Pointer
	dW, dB, dGamma, dBeta           []unsafe.Pointer

	gradBuf    unsafe.Pointer
	lossBuf    unsafe.Pointer
	lossScalar unsafe.Pointer
	inputBuf   unsafe.Pointer
	targetBuf  unsafe.Pointer

	inputShared  []float32
	targetShared []float32
	lossShared   []float32

	step           int
	dropoutCounter uint32
}

func mtlBufAlloc(nFloats int) unsafe.Pointer {
	return unsafe.Pointer(C.mtl_alloc(C.ulong(nFloats * 4)))
}

func mtlBufSharedPtr(ref unsafe.Pointer) unsafe.Pointer {
	return C.mtl_shared_ptr(C.MTLBufferRef(ref))
}

func mtlBufZeros(nFloats int) unsafe.Pointer {
	ref := mtlBufAlloc(nFloats)
	ptr := (*[1 << 30]float32)(mtlBufSharedPtr(ref))[:nFloats]
	for i := range ptr {
		ptr[i] = 0
	}
	return ref
}

func mtlBufFromHost(data []float32) unsafe.Pointer {
	ref := mtlBufAlloc(len(data))
	ptr := (*[1 << 30]float32)(mtlBufSharedPtr(ref))[:len(data)]
	copy(ptr, data)
	return ref
}

func findMLPMetallib() string {
	const name = "mlp_train.metallib"
	paths := []string{
		filepath.Join(".", name),
		filepath.Join("kernels", name),
	}
	if exe, err := os.Executable(); err == nil {
		dir := filepath.Dir(exe)
		paths = append(paths,
			filepath.Join(dir, name),
			filepath.Join(dir, "kernels", name),
		)
	}
	if gopath := os.Getenv("GOPATH"); gopath != "" {
		paths = append(paths, filepath.Join(gopath, "src/github.com/tensorwire/mongoose/kernels", name))
	}
	if home := os.Getenv("HOME"); home != "" {
		paths = append(paths,
			filepath.Join(home, "go/src/github.com/tensorwire/mongoose/kernels", name),
			filepath.Join(home, "tensorwire", name),
		)
	}
	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			if abs, err := filepath.Abs(p); err == nil {
				return abs
			}
			return p
		}
	}
	return ""
}

func NewMLPMetal(eng *Metal, mlp *MLP, batchSize int) *MLPMetal {
	path := findMLPMetallib()
	if path == "" {
		log.Println("[MLPMetal] mlp_train.metallib not found")
		return nil
	}
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	if C.mtl_mlp_init_path(cPath) == 0 {
		log.Println("[MLPMetal] mlp_train.metallib failed to load")
		return nil
	}

	nLayers := len(mlp.Layers)
	m := &MLPMetal{
		eng: eng, mlp: mlp, batchSize: batchSize, nLayers: nLayers,
		inDims: make([]C.int, nLayers), outDims: make([]C.int, nLayers), hasBN: make([]C.int, nLayers),
	}

	alloc := func(name string, n int) []unsafe.Pointer {
		s := make([]unsafe.Pointer, n)
		return s
	}
	m.W = alloc("W", nLayers)
	m.bias = alloc("bias", nLayers)
	m.gamma = alloc("gamma", nLayers)
	m.beta = alloc("beta", nLayers)
	m.runMean = alloc("runMean", nLayers)
	m.runVar = alloc("runVar", nLayers)
	m.bnMean = alloc("bnMean", nLayers)
	m.bnVar = alloc("bnVar", nLayers)
	m.mW = alloc("mW", nLayers)
	m.vW = alloc("vW", nLayers)
	m.mB = alloc("mB", nLayers)
	m.vB = alloc("vB", nLayers)
	m.mG = alloc("mG", nLayers)
	m.vG = alloc("vG", nLayers)
	m.mBt = alloc("mBt", nLayers)
	m.vBt = alloc("vBt", nLayers)
	m.act = alloc("act", nLayers)
	m.preBN = alloc("preBN", nLayers)
	m.preReLU = alloc("preReLU", nLayers)
	m.masks = alloc("masks", nLayers)
	m.dW = alloc("dW", nLayers)
	m.dB = alloc("dB", nLayers)
	m.dGamma = alloc("dGamma", nLayers)
	m.dBeta = alloc("dBeta", nLayers)

	for i, l := range mlp.Layers {
		nW := l.OutDim * l.InDim
		n := batchSize * l.OutDim

		m.inDims[i] = C.int(l.InDim)
		m.outDims[i] = C.int(l.OutDim)
		if l.BNGamma != nil {
			m.hasBN[i] = 1
		}

		m.W[i] = mtlBufFromHost(l.W)
		m.bias[i] = mtlBufFromHost(l.B)
		m.mW[i] = mtlBufZeros(nW)
		m.vW[i] = mtlBufZeros(nW)
		m.mB[i] = mtlBufZeros(l.OutDim)
		m.vB[i] = mtlBufZeros(l.OutDim)
		m.act[i] = mtlBufZeros(n)
		m.preBN[i] = mtlBufZeros(n)
		m.preReLU[i] = mtlBufZeros(n)
		m.masks[i] = mtlBufZeros(n)
		m.dW[i] = mtlBufZeros(nW)
		m.dB[i] = mtlBufZeros(l.OutDim)

		if l.BNGamma != nil {
			m.gamma[i] = mtlBufFromHost(l.BNGamma)
			m.beta[i] = mtlBufFromHost(l.BNBeta)
			m.runMean[i] = mtlBufFromHost(l.BNMean)
			m.runVar[i] = mtlBufFromHost(l.BNVar)
			m.bnMean[i] = mtlBufZeros(l.OutDim)
			m.bnVar[i] = mtlBufZeros(l.OutDim)
			m.mG[i] = mtlBufZeros(l.OutDim)
			m.vG[i] = mtlBufZeros(l.OutDim)
			m.mBt[i] = mtlBufZeros(l.OutDim)
			m.vBt[i] = mtlBufZeros(l.OutDim)
			m.dGamma[i] = mtlBufZeros(l.OutDim)
			m.dBeta[i] = mtlBufZeros(l.OutDim)
		} else {
			m.gamma[i] = mtlBufZeros(1)
			m.beta[i] = mtlBufZeros(1)
			m.runMean[i] = mtlBufZeros(1)
			m.runVar[i] = mtlBufZeros(1)
			m.bnMean[i] = mtlBufZeros(1)
			m.bnVar[i] = mtlBufZeros(1)
			m.mG[i] = mtlBufZeros(1)
			m.vG[i] = mtlBufZeros(1)
			m.mBt[i] = mtlBufZeros(1)
			m.vBt[i] = mtlBufZeros(1)
			m.dGamma[i] = mtlBufZeros(1)
			m.dBeta[i] = mtlBufZeros(1)
		}
	}

	nFeatures := mlp.Layers[0].InDim
	m.gradBuf = mtlBufZeros(batchSize)
	m.lossBuf = mtlBufZeros(batchSize)
	m.lossScalar = mtlBufZeros(1)
	m.inputBuf = mtlBufAlloc(batchSize * nFeatures)
	m.targetBuf = mtlBufAlloc(batchSize)

	m.inputShared = (*[1 << 30]float32)(mtlBufSharedPtr(m.inputBuf))[:batchSize*nFeatures]
	m.targetShared = (*[1 << 30]float32)(mtlBufSharedPtr(m.targetBuf))[:batchSize]
	m.lossShared = (*[1 << 30]float32)(mtlBufSharedPtr(m.lossScalar))[:1]

	log.Printf("[MLPMetal] allocated %d layers, batch=%d", nLayers, batchSize)
	return m
}

func (m *MLPMetal) UploadBatch(features, targets []float32) {
	copy(m.inputShared, features)
	copy(m.targetShared, targets)
}

func (m *MLPMetal) TrainStep(lr float32) float32 {
	m.step++
	m.dropoutCounter++
	bc1 := float32(1.0 - math.Pow(0.9, float64(m.step)))
	bc2 := float32(1.0 - math.Pow(0.999, float64(m.step)))
	wd := m.mlp.Config.WeightDecay

	C.mtl_mlp_train_step(
		C.int(m.nLayers), (*C.int)(&m.inDims[0]), (*C.int)(&m.outDims[0]), (*C.int)(&m.hasBN[0]),
		&m.W[0], &m.bias[0],
		&m.gamma[0], &m.beta[0], &m.runMean[0], &m.runVar[0],
		&m.bnMean[0], &m.bnVar[0],
		&m.mW[0], &m.vW[0], &m.mB[0], &m.vB[0],
		&m.mG[0], &m.vG[0], &m.mBt[0], &m.vBt[0],
		&m.act[0], &m.preBN[0], &m.preReLU[0], &m.masks[0],
		&m.dW[0], &m.dB[0], &m.dGamma[0], &m.dBeta[0],
		m.gradBuf, m.lossBuf, m.lossScalar,
		m.inputBuf, m.targetBuf,
		C.int(m.batchSize), C.float(lr), C.float(wd), C.float(0.9), C.float(0.999),
		C.float(bc1), C.float(bc2),
		C.float(m.mlp.Config.BNMomentum), C.float(1e-8), C.float(m.mlp.Config.Dropout),
		C.uint(42), C.uint(m.dropoutCounter),
	)

	return m.lossShared[0]
}

func (m *MLPMetal) DownloadWeights() {
	for i, l := range m.mlp.Layers {
		nW := l.OutDim * l.InDim
		wPtr := (*[1 << 30]float32)(mtlBufSharedPtr(m.W[i]))[:nW]
		copy(l.W, wPtr)
		bPtr := (*[1 << 30]float32)(mtlBufSharedPtr(m.bias[i]))[:l.OutDim]
		copy(l.B, bPtr)
		if l.BNGamma != nil {
			copy(l.BNGamma, (*[1 << 30]float32)(mtlBufSharedPtr(m.gamma[i]))[:l.OutDim])
			copy(l.BNBeta, (*[1 << 30]float32)(mtlBufSharedPtr(m.beta[i]))[:l.OutDim])
			copy(l.BNMean, (*[1 << 30]float32)(mtlBufSharedPtr(m.runMean[i]))[:l.OutDim])
			copy(l.BNVar, (*[1 << 30]float32)(mtlBufSharedPtr(m.runVar[i]))[:l.OutDim])
		}
	}
}

func (m *MLPMetal) Destroy() {}

func MtlGemmBT(a, b, c unsafe.Pointer, M, K, N int) {
	C.mtl_mlp_gemm_bt(a, b, c, C.int(M), C.int(K), C.int(N))
}

func MtlGemmTN(a, b, c unsafe.Pointer, M, K, N int) {
	C.mtl_mlp_gemm_tn(a, b, c, C.int(M), C.int(K), C.int(N))
}

func MtlGemmNN(a, b, c unsafe.Pointer, M, K, N int) {
	C.mtl_mlp_gemm_nn(a, b, c, C.int(M), C.int(K), C.int(N))
}

func MtlBiasAdd(out, bias unsafe.Pointer, B, D int) {
	C.mtl_mlp_bias_add(out, bias, C.int(B), C.int(D))
}

func MtlBNForward(x, mean, vr, gamma, beta, runMean, runVar unsafe.Pointer, B, D int, momentum float32) {
	C.mtl_mlp_bn_forward(x, mean, vr, gamma, beta, runMean, runVar, C.int(B), C.int(D), C.float(momentum))
}

func MtlBNBackward(dOut, x, mean, vr, gamma, dGamma, dBeta unsafe.Pointer, B, D int) {
	C.mtl_mlp_bn_backward(dOut, x, mean, vr, gamma, dGamma, dBeta, C.int(B), C.int(D))
}

func MtlBCE(logits, targets, grad, lossBuf, lossScalar unsafe.Pointer, B int) {
	C.mtl_mlp_bce(logits, targets, grad, lossBuf, lossScalar, C.int(B))
}

func MtlAdamWStep(param, grad, m, v unsafe.Pointer, lr, beta1, beta2, bc1, bc2, eps, wd float32, n int) {
	C.mtl_mlp_adamw_step(param, grad, m, v, C.float(lr), C.float(beta1), C.float(beta2),
		C.float(bc1), C.float(bc2), C.float(eps), C.float(wd), C.int(n))
}

func MtlDropoutFwd(x, mask unsafe.Pointer, n int, p float32, seed, counter uint32) {
	C.mtl_mlp_dropout_fwd(x, mask, C.int(n), C.float(p), C.uint(seed), C.uint(counter))
}

func MtlDropoutBwd(dx, mask unsafe.Pointer, n int) {
	C.mtl_mlp_dropout_bwd(dx, mask, C.int(n))
}

func MtlBufAlloc(nFloats int) unsafe.Pointer  { return mtlBufAlloc(nFloats) }
func MtlBufFromHost(data []float32) unsafe.Pointer { return mtlBufFromHost(data) }
func MtlBufZeros(nFloats int) unsafe.Pointer   { return mtlBufZeros(nFloats) }
func MtlBufSharedSlice(ref unsafe.Pointer, n int) []float32 {
	return (*[1 << 30]float32)(mtlBufSharedPtr(ref))[:n]
}
func MtlMLPInit() bool {
	path := findMLPMetallib()
	if path == "" {
		return false
	}
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	return C.mtl_mlp_init_path(cPath) != 0
}
