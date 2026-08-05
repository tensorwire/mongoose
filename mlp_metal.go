
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
import _ "unsafe"
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
	W, bias                      []unsafe.Pointer
	gamma, beta, runMean, runVar []unsafe.Pointer
	bnMean, bnVar                []unsafe.Pointer
	mW, vW, mB, vB               []unsafe.Pointer
	mG, vG, mBt, vBt             []unsafe.Pointer
	act, preBN, preReLU, masks   []unsafe.Pointer
	dW, dB, dGamma, dBeta        []unsafe.Pointer

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
	return unsafe.Pointer((C.mtl_alloc)(C.ulong(nFloats * 4)))
}

func mtlBufSharedPtr(ref unsafe.Pointer) unsafe.Pointer {
	return func() unsafe.Pointer {
		_cgo0 := C.MTLBufferRef(ref)
		_cgoCheckPointer(_cgo0, nil)
		return C.mtl_shared_ptr(_cgo0)
	}()
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
	cPath := (C.CString)(path)
	defer func() func() {
		_cgo0 := unsafe.Pointer(cPath)
		return func() { _cgoCheckPointer(_cgo0, nil); C.free(_cgo0) }
	}()()
	if (C.mtl_mlp_init_path)(cPath) == 0 {
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

	func() {
		var _cgo0 C.int = C.int(m.nLayers)
		var _cgo1 *C.int = (*C.int)(&m.inDims[0])
		var _cgo2 *C.int = (*C.int)(&m.outDims[0])
		var _cgo3 *C.int = (*C.int)(&m.hasBN[0])
		_cgoIndex4 := &m.W
		_cgo4 := &(*_cgoIndex4)[0]
		_cgoIndex5 := &m.bias
		_cgo5 := &(*_cgoIndex5)[0]
		_cgoIndex6 := &m.gamma
		_cgo6 := &(*_cgoIndex6)[0]
		_cgoIndex7 := &m.beta
		_cgo7 := &(*_cgoIndex7)[0]
		_cgoIndex8 := &m.runMean
		_cgo8 := &(*_cgoIndex8)[0]
		_cgoIndex9 := &m.runVar
		_cgo9 := &(*_cgoIndex9)[0]
		_cgoIndex10 := &m.bnMean
		_cgo10 := &(*_cgoIndex10)[0]
		_cgoIndex11 := &m.bnVar
		_cgo11 := &(*_cgoIndex11)[0]
		_cgoIndex12 := &m.mW
		_cgo12 := &(*_cgoIndex12)[0]
		_cgoIndex13 := &m.vW
		_cgo13 := &(*_cgoIndex13)[0]
		_cgoIndex14 := &m.mB
		_cgo14 := &(*_cgoIndex14)[0]
		_cgoIndex15 := &m.vB
		_cgo15 := &(*_cgoIndex15)[0]
		_cgoIndex16 := &m.mG
		_cgo16 := &(*_cgoIndex16)[0]
		_cgoIndex17 := &m.vG
		_cgo17 := &(*_cgoIndex17)[0]
		_cgoIndex18 := &m.mBt
		_cgo18 := &(*_cgoIndex18)[0]
		_cgoIndex19 := &m.vBt
		_cgo19 := &(*_cgoIndex19)[0]
		_cgoIndex20 := &m.act
		_cgo20 := &(*_cgoIndex20)[0]
		_cgoIndex21 := &m.preBN
		_cgo21 := &(*_cgoIndex21)[0]
		_cgoIndex22 := &m.preReLU
		_cgo22 := &(*_cgoIndex22)[0]
		_cgoIndex23 := &m.masks
		_cgo23 := &(*_cgoIndex23)[0]
		_cgoIndex24 := &m.dW
		_cgo24 := &(*_cgoIndex24)[0]
		_cgoIndex25 := &m.dB
		_cgo25 := &(*_cgoIndex25)[0]
		_cgoIndex26 := &m.dGamma
		_cgo26 := &(*_cgoIndex26)[0]
		_cgoIndex27 := &m.dBeta
		_cgo27 := &(*_cgoIndex27)[0]
		_cgo28 := m.gradBuf
		_cgo29 := m.lossBuf
		_cgo30 := m.lossScalar
		_cgo31 := m.inputBuf
		_cgo32 := m.targetBuf
		var _cgo33 C.int = C.int(m.batchSize)
		var _cgo34 C.float = C.float(lr)
		var _cgo35 C.float = C.float(wd)
		var _cgo36 C.float = C.float(0.9)
		var _cgo37 C.float = C.float(0.999)
		var _cgo38 C.float = C.float(bc1)
		var _cgo39 C.float = C.float(bc2)
		var _cgo40 C.float = C.float(m.mlp.Config.BNMomentum)
		var _cgo41 C.float = C.float(1e-8)
		var _cgo42 C.float = C.float(m.mlp.Config.Dropout)
		var _cgo43 C.uint = C.uint(42)
		var _cgo44 C.uint = C.uint(m.dropoutCounter)
		_cgoCheckPointer(_cgo4, *_cgoIndex4)
		_cgoCheckPointer(_cgo5, *_cgoIndex5)
		_cgoCheckPointer(_cgo6, *_cgoIndex6)
		_cgoCheckPointer(_cgo7, *_cgoIndex7)
		_cgoCheckPointer(_cgo8, *_cgoIndex8)
		_cgoCheckPointer(_cgo9, *_cgoIndex9)
		_cgoCheckPointer(_cgo10, *_cgoIndex10)
		_cgoCheckPointer(_cgo11, *_cgoIndex11)
		_cgoCheckPointer(_cgo12, *_cgoIndex12)
		_cgoCheckPointer(_cgo13, *_cgoIndex13)
		_cgoCheckPointer(_cgo14, *_cgoIndex14)
		_cgoCheckPointer(_cgo15, *_cgoIndex15)
		_cgoCheckPointer(_cgo16, *_cgoIndex16)
		_cgoCheckPointer(_cgo17, *_cgoIndex17)
		_cgoCheckPointer(_cgo18, *_cgoIndex18)
		_cgoCheckPointer(_cgo19, *_cgoIndex19)
		_cgoCheckPointer(_cgo20, *_cgoIndex20)
		_cgoCheckPointer(_cgo21, *_cgoIndex21)
		_cgoCheckPointer(_cgo22, *_cgoIndex22)
		_cgoCheckPointer(_cgo23, *_cgoIndex23)
		_cgoCheckPointer(_cgo24, *_cgoIndex24)
		_cgoCheckPointer(_cgo25, *_cgoIndex25)
		_cgoCheckPointer(_cgo26, *_cgoIndex26)
		_cgoCheckPointer(_cgo27, *_cgoIndex27)
		_cgoCheckPointer(_cgo28, nil)
		_cgoCheckPointer(_cgo29, nil)
		_cgoCheckPointer(_cgo30, nil)
		_cgoCheckPointer(_cgo31, nil)
		_cgoCheckPointer(_cgo32, nil)
		C.mtl_mlp_train_step(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5, _cgo6, _cgo7, _cgo8, _cgo9, _cgo10, _cgo11, _cgo12, _cgo13, _cgo14, _cgo15, _cgo16, _cgo17, _cgo18, _cgo19, _cgo20, _cgo21, _cgo22, _cgo23, _cgo24, _cgo25, _cgo26, _cgo27, _cgo28, _cgo29, _cgo30, _cgo31, _cgo32, _cgo33, _cgo34, _cgo35, _cgo36, _cgo37, _cgo38, _cgo39, _cgo40, _cgo41, _cgo42, _cgo43, _cgo44)
	}()

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
	func() {
		_cgo0 := a
		_cgo1 := b
		_cgo2 := c
		var _cgo3 C.int = C.int(M)
		var _cgo4 C.int = C.int(K)
		var _cgo5 C.int = C.int(N)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		C.mtl_mlp_gemm_bt(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5)
	}()
}

func MtlGemmTN(a, b, c unsafe.Pointer, M, K, N int) {
	func() {
		_cgo0 := a
		_cgo1 := b
		_cgo2 := c
		var _cgo3 C.int = C.int(M)
		var _cgo4 C.int = C.int(K)
		var _cgo5 C.int = C.int(N)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		C.mtl_mlp_gemm_tn(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5)
	}()
}

func MtlGemmNN(a, b, c unsafe.Pointer, M, K, N int) {
	func() {
		_cgo0 := a
		_cgo1 := b
		_cgo2 := c
		var _cgo3 C.int = C.int(M)
		var _cgo4 C.int = C.int(K)
		var _cgo5 C.int = C.int(N)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		C.mtl_mlp_gemm_nn(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5)
	}()
}

func MtlBiasAdd(out, bias unsafe.Pointer, B, D int) {
	func() {
		_cgo0 := out
		_cgo1 := bias
		var _cgo2 C.int = C.int(B)
		var _cgo3 C.int = C.int(D)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		C.mtl_mlp_bias_add(_cgo0, _cgo1, _cgo2, _cgo3)
	}()
}

func MtlBNForward(x, mean, vr, gamma, beta, runMean, runVar unsafe.Pointer, B, D int, momentum float32) {
	func() {
		_cgo0 := x
		_cgo1 := mean
		_cgo2 := vr
		_cgo3 := gamma
		_cgo4 := beta
		_cgo5 := runMean
		_cgo6 := runVar
		var _cgo7 C.int = C.int(B)
		var _cgo8 C.int = C.int(D)
		var _cgo9 C.float = C.float(momentum)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		_cgoCheckPointer(_cgo3, nil)
		_cgoCheckPointer(_cgo4, nil)
		_cgoCheckPointer(_cgo5, nil)
		_cgoCheckPointer(_cgo6, nil)
		C.mtl_mlp_bn_forward(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5, _cgo6, _cgo7, _cgo8, _cgo9)
	}()
}

func MtlBNBackward(dOut, x, mean, vr, gamma, dGamma, dBeta unsafe.Pointer, B, D int) {
	func() {
		_cgo0 := dOut
		_cgo1 := x
		_cgo2 := mean
		_cgo3 := vr
		_cgo4 := gamma
		_cgo5 := dGamma
		_cgo6 := dBeta
		var _cgo7 C.int = C.int(B)
		var _cgo8 C.int = C.int(D)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		_cgoCheckPointer(_cgo3, nil)
		_cgoCheckPointer(_cgo4, nil)
		_cgoCheckPointer(_cgo5, nil)
		_cgoCheckPointer(_cgo6, nil)
		C.mtl_mlp_bn_backward(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5, _cgo6, _cgo7, _cgo8)
	}()
}

func MtlBCE(logits, targets, grad, lossBuf, lossScalar unsafe.Pointer, B int) {
	func() {
		_cgo0 := logits
		_cgo1 := targets
		_cgo2 := grad
		_cgo3 := lossBuf
		_cgo4 := lossScalar
		var _cgo5 C.int = C.int(B)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		_cgoCheckPointer(_cgo3, nil)
		_cgoCheckPointer(_cgo4, nil)
		C.mtl_mlp_bce(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5)
	}()
}

func MtlAdamWStep(param, grad, m, v unsafe.Pointer, lr, beta1, beta2, bc1, bc2, eps, wd float32, n int) {
	func() {
		_cgo0 := param
		_cgo1 := grad
		_cgo2 := m
		_cgo3 := v
		var _cgo4 C.float = C.float(lr)
		var _cgo5 C.float = C.float(beta1)
		var _cgo6 C.float = C.float(beta2)
		var _cgo7 C.float = C.float(bc1)
		var _cgo8 C.float = C.float(bc2)
		var _cgo9 C.float = C.float(eps)
		var _cgo10 C.float = C.float(wd)
		var _cgo11 C.int = C.int(n)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		_cgoCheckPointer(_cgo3, nil)
		C.mtl_mlp_adamw_step(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5, _cgo6, _cgo7, _cgo8, _cgo9, _cgo10, _cgo11)
	}()
}

func MtlDropoutFwd(x, mask unsafe.Pointer, n int, p float32, seed, counter uint32) {
	func() {
		_cgo0 := x
		_cgo1 := mask
		var _cgo2 C.int = C.int(n)
		var _cgo3 C.float = C.float(p)
		var _cgo4 C.uint = C.uint(seed)
		var _cgo5 C.uint = C.uint(counter)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		C.mtl_mlp_dropout_fwd(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5)
	}()
}

func MtlDropoutBwd(dx, mask unsafe.Pointer, n int) {
	func() {
		_cgo0 := dx
		_cgo1 := mask
		var _cgo2 C.int = C.int(n)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		C.mtl_mlp_dropout_bwd(_cgo0, _cgo1, _cgo2)
	}()
}

func MtlBufAlloc(nFloats int) unsafe.Pointer       { return mtlBufAlloc(nFloats) }
func MtlBufFromHost(data []float32) unsafe.Pointer { return mtlBufFromHost(data) }
func MtlBufZeros(nFloats int) unsafe.Pointer       { return mtlBufZeros(nFloats) }
func MtlBufSharedSlice(ref unsafe.Pointer, n int) []float32 {
	return (*[1 << 30]float32)(mtlBufSharedPtr(ref))[:n]
}
func MtlMLPInit() bool {
	path := findMLPMetallib()
	if path == "" {
		return false
	}
	cPath := (C.CString)(path)
	defer func() func() {
		_cgo0 := unsafe.Pointer(cPath)
		return func() { _cgoCheckPointer(_cgo0, nil); C.free(_cgo0) }
	}()()
	return (C.mtl_mlp_init_path)(cPath) != 0
}
