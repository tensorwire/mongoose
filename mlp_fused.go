//go:build linux && cgo

package mongoose

/*
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>

extern void* tw_gpu_alloc(size_t bytes);
extern void* tw_get_kernel_lib();

// MLPLayerDesc matches the CUDA struct
typedef struct {
    int inDim, outDim;
    int hasBN;
} MLPLayerDesc;

typedef void (*fn_mlp_fused_train)(
    MLPLayerDesc*, int,
    float**, float**,
    float**, float**,
    float**, float**,
    float**, float**,
    float**, float**,
    float**, float**,
    float**, float**,
    float**, float**, float**,
    float**, float**, float**,
    float**, float**,
    float*,
    float*, float*,
    float*, float*, float*,
    float, float, float, float, float, float,
    float, float, float,
    unsigned long long, unsigned long long,
    int, int,
    void*
);

static fn_mlp_fused_train _mlp_fused = NULL;

static int mlp_fused_load() {
    void* lib = tw_get_kernel_lib();
    if (!lib) return 0;
    _mlp_fused = (fn_mlp_fused_train)dlsym(lib, "mongoose_mlp_fused_train");
    return _mlp_fused ? 1 : 0;
}

static void mlp_fused_call(
    MLPLayerDesc* layers, int nLayers,
    float** W, float** bias,
    float** gamma, float** beta,
    float** runMean, float** runVar,
    float** mW, float** vW,
    float** mB, float** vB,
    float** mG, float** vG,
    float** mBt, float** vBt,
    float** act, float** preBN, float** preReLU,
    float** masks, float** dW, float** dB,
    float** dGamma, float** dBeta,
    float* gradBuf,
    float* bnRedSum, float* bnRedSumSq,
    float* input, float* targets, float* lossOut,
    float lr, float wd, float b1, float b2, float bc1, float bc2,
    float bnMom, float eps, float dropP,
    unsigned long long dropSeed, unsigned long long dropCtr,
    int B, int maxDim,
    void* stream
) {
    if (!_mlp_fused) return;
    _mlp_fused(layers, nLayers,
        W, bias, gamma, beta, runMean, runVar,
        mW, vW, mB, vB, mG, vG, mBt, vBt,
        act, preBN, preReLU, masks, dW, dB, dGamma, dBeta,
        gradBuf, bnRedSum, bnRedSumSq,
        input, targets, lossOut,
        lr, wd, b1, b2, bc1, bc2,
        bnMom, eps, dropP,
        dropSeed, dropCtr,
        B, maxDim,
        stream);
}

static void* gpu_alloc(int n) { return tw_gpu_alloc((size_t)n * 4); }
static void gpu_zero(void* p, int n) { cudaMemset(p, 0, (size_t)n * 4); }
static void gpu_upload(void* dst, const void* src, int bytes) { cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice); }
static void gpu_download(void* dst, const void* src, int bytes) { cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost); }


// Device pointer array: allocate array of pointers on GPU
static void* gpu_alloc_ptr_array(int n) {
    void* p = NULL;
    cudaMalloc(&p, n * sizeof(float*));
    return p;
}
static void gpu_upload_ptr_array(void* dst, void* src, int n) {
    cudaMemcpy(dst, src, n * sizeof(float*), cudaMemcpyHostToDevice);
}
*/
import "C"
import (
	"log"
	"math"
	"unsafe"
)

type MLPFused struct {
	eng       *CUDA
	mlp       *MLP
	batchSize int
	nLayers   int
	maxDim    int

	// Device layer descriptors
	dLayers unsafe.Pointer // MLPLayerDesc[]

	// Device pointer arrays (arrays of float* on GPU)
	dW, dBias                          unsafe.Pointer
	dGamma, dBeta                      unsafe.Pointer
	dRunMean, dRunVar                  unsafe.Pointer
	dMW, dVW, dMB, dVB                unsafe.Pointer
	dMG, dVG, dMBt, dVBt              unsafe.Pointer
	dAct, dPreBN, dPreReLU, dMasks    unsafe.Pointer
	dDW, dDB, dDGamma, dDBeta         unsafe.Pointer

	// Scalar device buffers
	dGradBuf    unsafe.Pointer
	dBnRedSum   unsafe.Pointer
	dBnRedSumSq unsafe.Pointer
	dInput      unsafe.Pointer
	dTargets    unsafe.Pointer
	dLossOut    unsafe.Pointer

	// Host-side pointer arrays (for building device arrays)
	hW, hBias             []unsafe.Pointer
	hGamma, hBeta         []unsafe.Pointer
	hRunMean, hRunVar     []unsafe.Pointer
	hMW, hVW, hMB, hVB   []unsafe.Pointer
	hMG, hVG, hMBt, hVBt []unsafe.Pointer
	hAct, hPreBN          []unsafe.Pointer
	hPreReLU, hMasks      []unsafe.Pointer
	hDW, hDB              []unsafe.Pointer
	hDGamma, hDBeta       []unsafe.Pointer

	step           int
	dropoutCounter uint64
}

func NewMLPFused(eng *CUDA, mlp *MLP, batchSize int) *MLPFused {
	nLayers := len(mlp.Layers)

	if C.mlp_fused_load() == 0 {
		log.Fatal("[MLPFused] fused training kernel not loaded — recompile kernels with -rdc=true")
	}

	// Find max dim for BN reduction buffers
	maxDim := 0
	for _, l := range mlp.Layers {
		if l.OutDim > maxDim {
			maxDim = l.OutDim
		}
		if l.InDim > maxDim {
			maxDim = l.InDim
		}
	}

	f := &MLPFused{
		eng:       eng,
		mlp:       mlp,
		batchSize: batchSize,
		nLayers:   nLayers,
		maxDim:    maxDim,
	}

	// Ensure MLP weights are on GPU
	te := AsTensorEngine(eng)
	if mlp.Layers[0].gW == nil {
		mlp.ToGPU(te)
	}

	// Build layer descriptors
	type layerDesc struct {
		inDim, outDim, hasBN int32
	}
	descs := make([]layerDesc, nLayers)
	for i, l := range mlp.Layers {
		bn := int32(0)
		if l.BNGamma != nil {
			bn = 1
		}
		descs[i] = layerDesc{int32(l.InDim), int32(l.OutDim), bn}
	}
	f.dLayers = C.gpu_alloc(C.int(nLayers * 3)) // 3 ints per desc
	C.gpu_upload(f.dLayers, unsafe.Pointer(&descs[0]), C.int(nLayers*12))

	// Allocate per-layer buffers
	f.hW = make([]unsafe.Pointer, nLayers)
	f.hBias = make([]unsafe.Pointer, nLayers)
	f.hGamma = make([]unsafe.Pointer, nLayers)
	f.hBeta = make([]unsafe.Pointer, nLayers)
	f.hRunMean = make([]unsafe.Pointer, nLayers)
	f.hRunVar = make([]unsafe.Pointer, nLayers)
	f.hMW = make([]unsafe.Pointer, nLayers)
	f.hVW = make([]unsafe.Pointer, nLayers)
	f.hMB = make([]unsafe.Pointer, nLayers)
	f.hVB = make([]unsafe.Pointer, nLayers)
	f.hMG = make([]unsafe.Pointer, nLayers)
	f.hVG = make([]unsafe.Pointer, nLayers)
	f.hMBt = make([]unsafe.Pointer, nLayers)
	f.hVBt = make([]unsafe.Pointer, nLayers)
	f.hAct = make([]unsafe.Pointer, nLayers)
	f.hPreBN = make([]unsafe.Pointer, nLayers)
	f.hPreReLU = make([]unsafe.Pointer, nLayers)
	f.hMasks = make([]unsafe.Pointer, nLayers)
	f.hDW = make([]unsafe.Pointer, nLayers)
	f.hDB = make([]unsafe.Pointer, nLayers)
	f.hDGamma = make([]unsafe.Pointer, nLayers)
	f.hDBeta = make([]unsafe.Pointer, nLayers)

	alloc := func(n int) unsafe.Pointer {
		p := C.gpu_alloc(C.int(n))
		C.gpu_zero(p, C.int(n))
		return p
	}

	for i, l := range mlp.Layers {
		nW := l.OutDim * l.InDim
		n := batchSize * l.OutDim

		// Point to existing GPU weight tensors
		f.hW[i] = l.gW.DevicePtr()
		f.hBias[i] = l.gB.DevicePtr()

		if l.BNGamma != nil {
			f.hGamma[i] = l.gBNGamma.DevicePtr()
			f.hBeta[i] = l.gBNBeta.DevicePtr()
			f.hRunMean[i] = l.gRunMean.DevicePtr()
			f.hRunVar[i] = l.gRunVar.DevicePtr()
			f.hMG[i] = alloc(l.OutDim)
			f.hVG[i] = alloc(l.OutDim)
			f.hMBt[i] = alloc(l.OutDim)
			f.hVBt[i] = alloc(l.OutDim)
		}

		f.hMW[i] = alloc(nW)
		f.hVW[i] = alloc(nW)
		f.hMB[i] = alloc(l.OutDim)
		f.hVB[i] = alloc(l.OutDim)

		f.hAct[i] = alloc(n)
		f.hPreBN[i] = alloc(n)
		f.hPreReLU[i] = alloc(n)
		f.hMasks[i] = alloc(n)
		f.hDW[i] = alloc(nW)
		f.hDB[i] = alloc(l.OutDim)
		f.hDGamma[i] = alloc(l.OutDim)
		f.hDBeta[i] = alloc(l.OutDim)
	}

	// Upload pointer arrays to GPU
	uploadPtrs := func(ptrs []unsafe.Pointer) unsafe.Pointer {
		d := C.gpu_alloc_ptr_array(C.int(nLayers))
		C.gpu_upload_ptr_array(d, unsafe.Pointer(&ptrs[0]), C.int(nLayers))
		return d
	}

	f.dW = uploadPtrs(f.hW)
	f.dBias = uploadPtrs(f.hBias)
	f.dGamma = uploadPtrs(f.hGamma)
	f.dBeta = uploadPtrs(f.hBeta)
	f.dRunMean = uploadPtrs(f.hRunMean)
	f.dRunVar = uploadPtrs(f.hRunVar)
	f.dMW = uploadPtrs(f.hMW)
	f.dVW = uploadPtrs(f.hVW)
	f.dMB = uploadPtrs(f.hMB)
	f.dVB = uploadPtrs(f.hVB)
	f.dMG = uploadPtrs(f.hMG)
	f.dVG = uploadPtrs(f.hVG)
	f.dMBt = uploadPtrs(f.hMBt)
	f.dVBt = uploadPtrs(f.hVBt)
	f.dAct = uploadPtrs(f.hAct)
	f.dPreBN = uploadPtrs(f.hPreBN)
	f.dPreReLU = uploadPtrs(f.hPreReLU)
	f.dMasks = uploadPtrs(f.hMasks)
	f.dDW = uploadPtrs(f.hDW)
	f.dDB = uploadPtrs(f.hDB)
	f.dDGamma = uploadPtrs(f.hDGamma)
	f.dDBeta = uploadPtrs(f.hDBeta)

	// Scalar buffers
	nFeatures := mlp.Layers[0].InDim
	f.dGradBuf = alloc(batchSize)
	f.dBnRedSum = alloc(maxDim)
	f.dBnRedSumSq = alloc(maxDim)
	f.dInput = alloc(batchSize * nFeatures)
	f.dTargets = alloc(batchSize)
	f.dLossOut = alloc(1)

	C.cudaDeviceSynchronize()
	log.Printf("[MLPFused] allocated %d layers, batch=%d, maxDim=%d", nLayers, batchSize, maxDim)
	return f
}

func (f *MLPFused) UploadBatch(features, targets []float32) {
	C.gpu_upload(f.dInput, unsafe.Pointer(&features[0]), C.int(len(features)*4))
	C.gpu_upload(f.dTargets, unsafe.Pointer(&targets[0]), C.int(len(targets)*4))
}

func (f *MLPFused) TrainStep(lr float32) float32 {
	f.step++
	f.dropoutCounter++
	bc1 := float32(1.0 - math.Pow(0.9, float64(f.step)))
	bc2 := float32(1.0 - math.Pow(0.999, float64(f.step)))
	wd := f.mlp.Config.WeightDecay
	if wd == 0 {
		wd = 0.0005
	}

	// Zero gradient accumulators before each step
	for i, l := range f.mlp.Layers {
		nW := l.OutDim * l.InDim
		C.gpu_zero(f.hDW[i], C.int(nW))
		C.gpu_zero(f.hDB[i], C.int(l.OutDim))
		if l.BNGamma != nil {
			C.gpu_zero(f.hDGamma[i], C.int(l.OutDim))
			C.gpu_zero(f.hDBeta[i], C.int(l.OutDim))
		}
	}
	C.gpu_zero(f.dLossOut, C.int(1))

	C.mlp_fused_call(
		(*C.MLPLayerDesc)(f.dLayers), C.int(f.nLayers),
		(**C.float)(f.dW), (**C.float)(f.dBias),
		(**C.float)(f.dGamma), (**C.float)(f.dBeta),
		(**C.float)(f.dRunMean), (**C.float)(f.dRunVar),
		(**C.float)(f.dMW), (**C.float)(f.dVW),
		(**C.float)(f.dMB), (**C.float)(f.dVB),
		(**C.float)(f.dMG), (**C.float)(f.dVG),
		(**C.float)(f.dMBt), (**C.float)(f.dVBt),
		(**C.float)(f.dAct), (**C.float)(f.dPreBN), (**C.float)(f.dPreReLU),
		(**C.float)(f.dMasks), (**C.float)(f.dDW), (**C.float)(f.dDB),
		(**C.float)(f.dDGamma), (**C.float)(f.dDBeta),
		(*C.float)(f.dGradBuf),
		(*C.float)(f.dBnRedSum), (*C.float)(f.dBnRedSumSq),
		(*C.float)(f.dInput),
		(*C.float)(f.dTargets),
		(*C.float)(f.dLossOut),
		C.float(lr), C.float(wd), C.float(0.9), C.float(0.999),
		C.float(bc1), C.float(bc2),
		C.float(f.mlp.Config.BNMomentum), C.float(1e-8), C.float(f.mlp.Config.Dropout),
		C.ulonglong(42), C.ulonglong(f.dropoutCounter),
		C.int(f.batchSize), C.int(f.maxDim),
		nil, // default stream
	)

	C.cudaDeviceSynchronize()

	// Read loss from device (L3 loss would need kernel to write to pinned, keep simple for now)
	var loss float32
	C.gpu_download(unsafe.Pointer(&loss), f.dLossOut, C.int(4))
	return loss
}

func (f *MLPFused) DownloadWeights() {
	f.mlp.ToCPU()
	// ToCPU downloads W, B, BNGamma, BNBeta but NOT running stats.
	// Validation uses running_mean/running_var — must download those too.
	for i, l := range f.mlp.Layers {
		if l.BNGamma != nil && f.hRunMean[i] != nil {
			C.gpu_download(unsafe.Pointer(&l.BNMean[0]), f.hRunMean[i], C.int(l.OutDim*4))
			C.gpu_download(unsafe.Pointer(&l.BNVar[0]), f.hRunVar[i], C.int(l.OutDim*4))
		}
	}
}

// DiagCompare runs one forward pass on GPU and CPU with the same input/weights,
// then compares every intermediate output to find where the math diverges.
func (f *MLPFused) DiagCompare(features, targets []float32) {
	B := f.batchSize
	mlp := f.mlp
	nFeatures := mlp.Layers[0].InDim

	// Sync weights from GPU to CPU so both paths use identical weights
	f.DownloadWeights()

	// === CPU forward ===
	cpuLogits := mlp.ForwardLogits(features, B, true)
	cpuLoss, _ := mlp.BCEWithLogitsLoss(cpuLogits, targets, 0)

	// === GPU forward (one step, no optimizer) ===
	f.UploadBatch(features, targets)
	// Run one full step to populate all buffers
	loss := f.TrainStep(0.0003)

	log.Printf("[DIAG] CPU loss=%.6f  GPU loss=%.6f  diff=%.6e", cpuLoss, loss, float64(loss-cpuLoss))

	// Check that CPU and GPU see the same input
	log.Printf("[DIAG] input[0:5] = %.6f %.6f %.6f %.6f %.6f", features[0], features[1], features[2], features[3], features[4])
	// CPU preAct[0] = first element of layer 0 linear+bias output
	if mlp.Layers[0].preAct != nil {
		log.Printf("[DIAG] CPU L0.preAct[0:5] = %.6f %.6f %.6f %.6f %.6f",
			mlp.Layers[0].preAct[0], mlp.Layers[0].preAct[1], mlp.Layers[0].preAct[2], mlp.Layers[0].preAct[3], mlp.Layers[0].preAct[4])
	}
	if mlp.Layers[0].postBN != nil {
		log.Printf("[DIAG] CPU L0.postBN[0:5] = %.6f %.6f %.6f %.6f %.6f",
			mlp.Layers[0].postBN[0], mlp.Layers[0].postBN[1], mlp.Layers[0].postBN[2], mlp.Layers[0].postBN[3], mlp.Layers[0].postBN[4])
	}

	// Download and compare each layer's intermediate outputs
	for i, l := range mlp.Layers {
		n := B * l.OutDim

		// GPU preBN = linear+bias output
		gpuPreBN := make([]float32, n)
		C.gpu_download(unsafe.Pointer(&gpuPreBN[0]), f.hPreBN[i], C.int(n*4))

		// GPU act = post-BN, post-ReLU, post-dropout (final activation)
		gpuAct := make([]float32, n)
		C.gpu_download(unsafe.Pointer(&gpuAct[0]), f.hAct[i], C.int(n*4))

		// CPU equivalents
		cpuPreAct := l.preAct  // linear+bias (pre-BN)
		cpuPostBN := l.postBN  // after BN
		cpuPostAct := l.postAct // after ReLU

		// Compare preBN (linear+bias) — should match preAct
		if cpuPreAct != nil {
			maxDiff, avgDiff, nz := compareSlices(gpuPreBN, cpuPreAct)
			status := "OK"
			if maxDiff > 1e-3 { status = "*** DIVERGED ***" }
			log.Printf("[DIAG] L%d linear+bias  max_diff=%.6e avg_diff=%.6e nonzero=%d/%d %s",
				i, maxDiff, avgDiff, nz, len(cpuPreAct), status)

			// Print first few divergent elements
			if maxDiff > 1e-4 {
				printed := 0
				for j := 0; j < len(gpuPreBN) && printed < 5; j++ {
					d := abs32(gpuPreBN[j] - cpuPreAct[j])
					if d > 1e-4 {
						log.Printf("[DIAG]   elem[%d] gpu=%.6f cpu=%.6f diff=%.6e", j, gpuPreBN[j], cpuPreAct[j], d)
						printed++
					}
				}
			}
		}

		// Compare post-BN and BN stats
		if cpuPostBN != nil {
			gpuPreReLU := make([]float32, n)
			C.gpu_download(unsafe.Pointer(&gpuPreReLU[0]), f.hPreReLU[i], C.int(n*4))

			isLast := i == len(mlp.Layers)-1
			if !isLast && l.BNGamma != nil {
				maxDiff, avgDiff, _ := compareSlices(gpuPreReLU, cpuPostBN)
				status := "OK"
				if maxDiff > 1e-3 { status = "*** DIVERGED ***" }
				log.Printf("[DIAG] L%d post-BN      max_diff=%.6e avg_diff=%.6e %s",
					i, maxDiff, avgDiff, status)
				// Find and print the max-diff element
				if maxDiff > 1e-3 {
					for j := 0; j < len(gpuPreReLU) && j < len(cpuPostBN); j++ {
						d := abs32(gpuPreReLU[j] - cpuPostBN[j])
						if float64(d) >= maxDiff*0.99 {
							row := j / l.OutDim
							col := j % l.OutDim
							log.Printf("[DIAG]   max@[%d,%d] gpu=%.6f cpu=%.6f diff=%.6e",
								row, col, gpuPreReLU[j], cpuPostBN[j], d)
							break
						}
					}
				}

				// Compare BN running stats (GPU vs CPU) to check if BN mean/var are correct
				gpuRunMean := make([]float32, l.OutDim)
				gpuRunVar := make([]float32, l.OutDim)
				C.gpu_download(unsafe.Pointer(&gpuRunMean[0]), f.hRunMean[i], C.int(l.OutDim*4))
				C.gpu_download(unsafe.Pointer(&gpuRunVar[0]), f.hRunVar[i], C.int(l.OutDim*4))
				// CPU running stats were updated by the CPU forward pass
				mDiff, mAvg, _ := compareSlices(gpuRunMean, l.BNMean)
				vDiff, vAvg, _ := compareSlices(gpuRunVar, l.BNVar)
				log.Printf("[DIAG] L%d BN runMean   max_diff=%.6e avg_diff=%.6e", i, mDiff, mAvg)
				log.Printf("[DIAG] L%d BN runVar    max_diff=%.6e avg_diff=%.6e", i, vDiff, vAvg)

				// Also compute CPU mean/var manually for first few features
				if i == 0 {
					for d := 0; d < 3 && d < l.OutDim; d++ {
						var cpuSum float64
						for b := 0; b < B; b++ {
							cpuSum += float64(gpuPreBN[b*l.OutDim+d])
						}
						cpuMean := cpuSum / float64(B)
						var cpuVarSum float64
						for b := 0; b < B; b++ {
							diff := float64(gpuPreBN[b*l.OutDim+d]) - cpuMean
							cpuVarSum += diff * diff
						}
						cpuVar := cpuVarSum / float64(B)
						log.Printf("[DIAG] L0 feat[%d] manual_mean=%.6f manual_var=%.6f gpu_runMean=%.6f gpu_runVar=%.6f",
							d, cpuMean, cpuVar, gpuRunMean[d], gpuRunVar[d])
					}
				}
			}
		}

		// Compare post-activation (post-ReLU for hidden, raw for last)
		if cpuPostAct != nil {
			isLast := i == len(mlp.Layers)-1
			if isLast {
				// Last layer: GPU act = raw logits, CPU postAct = raw logits (ForwardLogits)
				maxDiff, avgDiff, _ := compareSlices(gpuAct, cpuPostAct)
				log.Printf("[DIAG] L%d logits       max_diff=%.6e avg_diff=%.6e",
					i, maxDiff, avgDiff)
			}
		}

		// Compare GPU dW against CPU dW
		gpuDW := make([]float32, l.OutDim*l.InDim)
		C.gpu_download(unsafe.Pointer(&gpuDW[0]), f.hDW[i], C.int(l.OutDim*l.InDim*4))
		if l.DW != nil {
			maxDiff, avgDiff, _ := compareSlices(gpuDW, l.DW)
			status := "OK"
			if maxDiff > 1e-3 { status = "*** DIVERGED ***" }
			log.Printf("[DIAG] L%d dW           max_diff=%.6e avg_diff=%.6e %s",
				i, maxDiff, avgDiff, status)
		}

		_ = nFeatures
	}
}

func compareSlices(a, b []float32) (maxDiff, avgDiff float64, nonzero int) {
	n := len(a)
	if len(b) < n { n = len(b) }
	var sum float64
	for i := 0; i < n; i++ {
		d := float64(abs32(a[i] - b[i]))
		if d > maxDiff { maxDiff = d }
		sum += d
		if a[i] != 0 || b[i] != 0 { nonzero++ }
	}
	if n > 0 { avgDiff = sum / float64(n) }
	return
}

func abs32(x float32) float32 {
	if x < 0 { return -x }
	return x
}

func (f *MLPFused) Destroy() {
	// GPU buffers freed when process exits (arena cleanup)
}
