//go:build linux && cgo

package mongoose

/*
#include <math.h>
#include <cuda_runtime.h>

void tw_gpu_upload(void* dst, const void* src, size_t bytes);
void* tw_gpu_alloc(size_t bytes);
void tw_gpu_zero(void* ptr, size_t bytes);

int tw_q4_cublas_matvec(const float* act, const void* weight_q4,
    void* scratch_fp16, float* out, int N, int K);

// L3 copy stream — separate from compute for true async overlap
static cudaStream_t tw_create_copy_stream() {
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    return s;
}
// Compute stream that synchronizes with the default (NULL) stream.
// Unlike NonBlocking, work on this stream respects implicit NULL-stream barriers.
static cudaStream_t tw_create_blocking_stream() {
    cudaStream_t s;
    cudaStreamCreate(&s);
    return s;
}
static cudaEvent_t tw_create_event() {
    cudaEvent_t e;
    cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
    return e;
}
static void tw_event_record(cudaEvent_t event, cudaStream_t stream) {
    cudaEventRecord(event, stream);
}
static void tw_stream_wait_event(cudaStream_t stream, cudaEvent_t event) {
    cudaStreamWaitEvent(stream, event, 0);
}
static void tw_stream_sync(cudaStream_t stream) {
    cudaStreamSynchronize(stream);
}
static void tw_copy_to_l3_on_stream(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream);
}
*/
import "C"

import (
	"log"
	"math"
	"time"
	"unsafe"
)

// vnodeScratch holds per-vNode scratch buffers for fused inference.
// Each vNode (CUDA stream) gets an independent set so stages can run
// concurrently without data races.
type vnodeScratch struct {
	normed     *Tensor
	Q, K, V    *Tensor
	attnOut    *Tensor
	proj       *Tensor
	normed2    *Tensor
	gatePre    *Tensor
	upOut      *Tensor
	ffnMid     *Tensor
	q8Scratch  unsafe.Pointer
}

// CUDAFusedInference holds GPU-resident weights and scratch buffers
// for fused single-token decode inference on CUDA.
// All kernel dispatches run on the default stream without CPU sync
// until the final readback.
type CUDAFusedInference struct {
	eng   *CUDA
	built bool

	dim, kvDim, headDim        int
	nHeads, nKVHeads           int
	ffnDim, vocabSize, nLayers int
	maxSeq                     int

	norm1                            []*Tensor
	wq, wk, wv, wo                   []*Tensor
	wqS, wkS, wvS, woS              []*Tensor
	norm2                            []*Tensor
	wgate, wup, wdown                []*Tensor
	wgateS, wupS, wdownS            []*Tensor
	finalNorm                        *Tensor
	lmHead, lmHeadS                  *Tensor

	// FP16 weight flag: when true, weight tensors are FP16 (not Q4/Q8).
	// Matvec uses cuBLAS hgemm, needle uses KNeedleFP16.
	WeightsFP16 bool

	// FP16 activation scratch for hgemm matvec (allocated lazily)
	fp16Act *Tensor // [max(dim, ffnDim)] FP16

	kCache, vCache []*Tensor

	hidden    [2]*Tensor
	hiddenIdx int
	logits             *Tensor

	// Per-vNode scratch buffers. scratch[0] = Stage A (stream 0),
	// scratch[1] = Stage B (computeStream1). Independent sets prevent
	// data races when both stages run on separate CUDA streams.
	scratch [2]vnodeScratch

	cosTab, sinTab *Tensor
	halfHead       int

	// fp16 scratch for cuBLAS tiled matvec
	fp16Scratch unsafe.Pointer
	fp16ScratchSize int

	// L3 bridge: pinned host buffer for async GPU→CPU activation transfer.
	// Layout: [0:4] = mailbox counter (uint32), [64:] = hidden state (float32).
	hiddenL3       *L3Bridge
	mailboxExpected uint32

	// Separate CUDA stream for L3 D2H copies (enables true async overlap).
	// Compute runs on stream 0, L3 copy runs on copyStream.
	// WaitL3 syncs copyStream only, not the entire device.
	copyStream  C.cudaStream_t
	computeDone C.cudaEvent_t

	// vNode dual compute streams: Stage A runs on stream 0 (default),
	// Stage B runs on computeStream1. stageDone event synchronizes A→B.
	// copyStream remains dedicated to L3 D2H async copy.
	computeStream1 C.cudaStream_t
	stageDone      C.cudaEvent_t
	vNodeCount     int

	// Fused dispatch: pre-built pointer arrays for single CGo call
	fusedReady                                          bool
	norm1Ptrs, norm2Ptrs                                []unsafe.Pointer
	wqPtrs, wkPtrs, wvPtrs, woPtrs                     []unsafe.Pointer
	wgatePtrs, wupPtrs, wdownPtrs                       []unsafe.Pointer
	kCachePtrs, vCachePtrs                               []unsafe.Pointer
}

func NewCUDAFusedInference(eng *CUDA, dim, kvDim, headDim, nHeads, nKVHeads, ffnDim, vocabSize, nLayers, maxSeq int, ropeTheta float64) *CUDAFusedInference {
	if !KernelsLoaded() { return nil }

	halfHead := headDim / 2
	f := &CUDAFusedInference{
		eng: eng, dim: dim, kvDim: kvDim, headDim: headDim,
		nHeads: nHeads, nKVHeads: nKVHeads, ffnDim: ffnDim,
		vocabSize: vocabSize, nLayers: nLayers, maxSeq: maxSeq,
		halfHead: halfHead,
	}

	f.norm1 = make([]*Tensor, nLayers)
	f.wq = make([]*Tensor, nLayers); f.wqS = make([]*Tensor, nLayers)
	f.wk = make([]*Tensor, nLayers); f.wkS = make([]*Tensor, nLayers)
	f.wv = make([]*Tensor, nLayers); f.wvS = make([]*Tensor, nLayers)
	f.wo = make([]*Tensor, nLayers); f.woS = make([]*Tensor, nLayers)
	f.norm2 = make([]*Tensor, nLayers)
	f.wgate = make([]*Tensor, nLayers); f.wgateS = make([]*Tensor, nLayers)
	f.wup = make([]*Tensor, nLayers); f.wupS = make([]*Tensor, nLayers)
	f.wdown = make([]*Tensor, nLayers); f.wdownS = make([]*Tensor, nLayers)

	f.kCache = make([]*Tensor, nLayers)
	f.vCache = make([]*Tensor, nLayers)
	// KV caches allocated lazily by SetLayerRange — saves VRAM on partial-load nodes

	f.hidden[0] = eng.Zeros([]int{dim})
	f.hidden[1] = eng.Zeros([]int{dim})
	f.logits = eng.Zeros([]int{vocabSize})

	// Q8 scratch size for dp4a matvec
	q8MaxK := dim
	if ffnDim > q8MaxK { q8MaxK = ffnDim }
	q8ScratchSize := (q8MaxK / 32) * 40

	// Allocate scratch[0] (always used by Stage A / single-stream path)
	f.allocScratch(0, q8ScratchSize)

	cosData := make([]float32, maxSeq*halfHead)
	sinData := make([]float32, maxSeq*halfHead)
	for pos := 0; pos < maxSeq; pos++ {
		for j := 0; j < halfHead; j++ {
			freq := 1.0 / math.Pow(ropeTheta, float64(2*j)/float64(headDim))
			angle := float64(pos) * freq
			cosData[pos*halfHead+j] = float32(math.Cos(angle))
			sinData[pos*halfHead+j] = float32(math.Sin(angle))
		}
	}
	f.cosTab = eng.FromHost(cosData, []int{maxSeq * halfHead})
	f.sinTab = eng.FromHost(sinData, []int{maxSeq * halfHead})

	// Only allocate fp16 scratch for cuBLAS path if dp4a is NOT available
	if !HasQ4_0DP4A() {
		f.fp16ScratchSize = ffnDim * dim * 2
		f.fp16Scratch = C.tw_gpu_alloc(C.size_t(f.fp16ScratchSize))
		if f.fp16Scratch == nil {
			log.Printf("[Q4] WARNING: fp16 scratch alloc failed (%d MB), using custom kernel",
				f.fp16ScratchSize/1024/1024)
		}
	}

	// L3 pinned buffer: [0:4] mailbox counter + [64:] hidden state
	l3Size := 64 + dim*4
	f.hiddenL3 = eng.AllocL3Bridge(l3Size)
	if f.hiddenL3 == nil {
		log.Printf("[L3] WARNING: L3 bridge allocation FAILED (dim=%d, %d bytes) — overlap disabled", dim, l3Size)
	} else {
		log.Printf("[L3] bridge allocated: %d bytes pinned (64B mailbox + %dB hidden)", l3Size, dim*4)
		f.mailboxExpected = 1
	}

	// Separate copy stream for true async L3 overlap
	f.copyStream = C.tw_create_copy_stream()
	f.computeDone = C.tw_create_event()
	log.Printf("[L3] copy stream + event created for async overlap")

	// vNode compute stream for Stage B — blocking (synchronizes with NULL stream).
	// NonBlocking streams don't respect implicit NULL-stream barriers, which
	// can cause data races when Stage A runs on the default (NULL) stream.
	f.computeStream1 = C.tw_create_blocking_stream()
	f.stageDone = C.tw_create_event()
	f.vNodeCount = 1
	log.Printf("[VNODE] compute stream1 (blocking) + stageDone event created")

	f.built = true
	return f
}

func (f *CUDAFusedInference) NumWeights() int {
	return f.nLayers*12 + 2
}

func (f *CUDAFusedInference) ExportHidden() []float32 {
	if !f.built { return nil }
	KSync()
	return f.eng.ToHost(f.hidden[f.hiddenIdx])
}

func (f *CUDAFusedInference) SetWeight(idx int, data []float32) {
	nL := f.nLayers
	if idx < nL*12 {
		layer := idx / 12
		w := idx % 12
		switch w {
		case 0:
			f.norm1[layer] = f.eng.FromHost(data, []int{f.dim})
		case 1:
			f.wq[layer], f.wqS[layer] = f.quantizeQ8(data, f.dim, f.dim)
		case 2:
			f.wk[layer], f.wkS[layer] = f.quantizeQ8(data, f.kvDim, f.dim)
		case 3:
			f.wv[layer], f.wvS[layer] = f.quantizeQ8(data, f.kvDim, f.dim)
		case 4, 5, 6: // bias — skip
		case 7:
			f.wo[layer], f.woS[layer] = f.quantizeQ8(data, f.dim, f.dim)
		case 8:
			f.norm2[layer] = f.eng.FromHost(data, []int{f.dim})
		case 9:
			f.wgate[layer], f.wgateS[layer] = f.quantizeQ8(data, f.ffnDim, f.dim)
		case 10:
			f.wup[layer], f.wupS[layer] = f.quantizeQ8(data, f.ffnDim, f.dim)
		case 11:
			f.wdown[layer], f.wdownS[layer] = f.quantizeQ8(data, f.dim, f.ffnDim)
		}
	} else if idx == nL*12 {
		f.finalNorm = f.eng.FromHost(data, []int{f.dim})
	} else if idx == nL*12+1 {
		f.lmHead, f.lmHeadS = f.quantizeQ8(data, f.vocabSize, f.dim)
	}
}

// SetWeightFP16 uploads a weight as FP16 on GPU. No quantization.
// wqS[layer] is left nil — NeedlePoke uses this to detect FP16 vs Q8.
func (f *CUDAFusedInference) SetWeightFP16(idx int, data []float32) {
	nL := f.nLayers
	if idx < nL*12 {
		layer := idx / 12
		w := idx % 12
		switch w {
		case 0:
			f.norm1[layer] = f.eng.FromHost(data, []int{f.dim})
		case 1:
			f.wq[layer] = f.eng.FromHostFP16(data, []int{f.dim * f.dim})
			f.wqS[layer] = nil
		case 2:
			f.wk[layer] = f.eng.FromHostFP16(data, []int{f.kvDim * f.dim})
			f.wkS[layer] = nil
		case 3:
			f.wv[layer] = f.eng.FromHostFP16(data, []int{f.kvDim * f.dim})
			f.wvS[layer] = nil
		case 4, 5, 6: // bias — skip
		case 7:
			f.wo[layer] = f.eng.FromHostFP16(data, []int{f.dim * f.dim})
			f.woS[layer] = nil
		case 8:
			f.norm2[layer] = f.eng.FromHost(data, []int{f.dim})
		case 9:
			f.wgate[layer] = f.eng.FromHostFP16(data, []int{f.ffnDim * f.dim})
			f.wgateS[layer] = nil
		case 10:
			f.wup[layer] = f.eng.FromHostFP16(data, []int{f.ffnDim * f.dim})
			f.wupS[layer] = nil
		case 11:
			f.wdown[layer] = f.eng.FromHostFP16(data, []int{f.dim * f.ffnDim})
			f.wdownS[layer] = nil
		}
	} else if idx == nL*12 {
		f.finalNorm = f.eng.FromHost(data, []int{f.dim})
	} else if idx == nL*12+1 {
		f.lmHead = f.eng.FromHostFP16(data, []int{f.vocabSize * f.dim})
		f.lmHeadS = nil
	}
	f.WeightsFP16 = true
}

// SetWeightRawFP16 uploads raw FP16 bytes directly to GPU — no conversion.
// Norms (case 0, 8) and biases (case 4,5,6) are always FP32 and passed as normData.
// Weight matrices (case 1,2,3,7,9,10,11) and lm_head are raw FP16 bytes.
func (f *CUDAFusedInference) SetWeightRawFP16(idx int, fp16Bytes []byte, nElems int, normData []float32) {
	nL := f.nLayers
	if idx < nL*12 {
		layer := idx / 12
		w := idx % 12
		switch w {
		case 0:
			f.norm1[layer] = f.eng.FromHost(normData, []int{f.dim})
		case 1:
			f.wq[layer] = f.eng.FromHostRawFP16(fp16Bytes, nElems)
			f.wqS[layer] = nil
		case 2:
			f.wk[layer] = f.eng.FromHostRawFP16(fp16Bytes, nElems)
			f.wkS[layer] = nil
		case 3:
			f.wv[layer] = f.eng.FromHostRawFP16(fp16Bytes, nElems)
			f.wvS[layer] = nil
		case 4, 5, 6:
		case 7:
			f.wo[layer] = f.eng.FromHostRawFP16(fp16Bytes, nElems)
			f.woS[layer] = nil
		case 8:
			f.norm2[layer] = f.eng.FromHost(normData, []int{f.dim})
		case 9:
			f.wgate[layer] = f.eng.FromHostRawFP16(fp16Bytes, nElems)
			f.wgateS[layer] = nil
		case 10:
			f.wup[layer] = f.eng.FromHostRawFP16(fp16Bytes, nElems)
			f.wupS[layer] = nil
		case 11:
			f.wdown[layer] = f.eng.FromHostRawFP16(fp16Bytes, nElems)
			f.wdownS[layer] = nil
		}
	} else if idx == nL*12 {
		f.finalNorm = f.eng.FromHost(normData, []int{f.dim})
	} else if idx == nL*12+1 {
		f.lmHead = f.eng.FromHostRawFP16(fp16Bytes, nElems)
		f.lmHeadS = nil
	}
	f.WeightsFP16 = true
}

func (f *CUDAFusedInference) quantizeQ8(fp32 []float32, rows, cols int) (*Tensor, *Tensor) {
	scales := make([]float32, rows)
	q8 := make([]int8, rows*cols)
	for r := 0; r < rows; r++ {
		var amax float32
		for c := 0; c < cols; c++ {
			v := fp32[r*cols+c]
			if v < 0 { v = -v }
			if v > amax { amax = v }
		}
		s := amax / 127.0
		scales[r] = s
		if s > 0 {
			inv := 1.0 / s
			for c := 0; c < cols; c++ {
				v := int(fp32[r*cols+c] * inv)
				if v > 127 { v = 127 } else if v < -128 { v = -128 }
				q8[r*cols+c] = int8(v)
			}
		}
	}
	gpuPtr := C.tw_gpu_alloc(C.size_t(rows * cols))
	C.tw_gpu_upload(gpuPtr, unsafe.Pointer(&q8[0]), C.size_t(rows*cols))
	dataTensor := TensorFromDevicePtr(gpuPtr, rows*cols)
	dataTensor.eng = f.eng
	scalesTensor := f.eng.FromHost(scales, []int{rows})
	return dataTensor, scalesTensor
}

func (f *CUDAFusedInference) PartialStep(hiddenIn []float32, pos, layerStart, layerEnd int, hiddenOut, logitsOut []float32) int {
	if !f.built { return -1 }

	f.eng.UploadInto(f.hidden[f.hiddenIdx], hiddenIn)

	s := &f.scratch[0]
	hP := f.hidden[f.hiddenIdx].DevicePtr()
	nP := s.normed.DevicePtr()
	qP := s.Q.DevicePtr()
	kP := s.K.DevicePtr()
	vP := s.V.DevicePtr()
	aP := s.attnOut.DevicePtr()
	pP := s.proj.DevicePtr()
	n2P := s.normed2.DevicePtr()
	gP := s.gatePre.DevicePtr()
	uP := s.upOut.DevicePtr()
	fP := s.ffnMid.DevicePtr()

	cosOff := pos * f.halfHead
	sinOff := pos * f.halfHead
	seqLen := pos + 1

	for l := layerStart; l < layerEnd && l < f.nLayers; l++ {
		if f.norm1[l] == nil { continue }

		KRMSNormOut(hP, nP, f.norm1[l].DevicePtr(), 1, f.dim)

		KQ8Matvec(nP, f.wq[l].DevicePtr(), f.wqS[l].DevicePtr(), qP, f.dim, f.dim)
		KQ8Matvec(nP, f.wk[l].DevicePtr(), f.wkS[l].DevicePtr(), kP, f.kvDim, f.dim)
		KQ8Matvec(nP, f.wv[l].DevicePtr(), f.wvS[l].DevicePtr(), vP, f.kvDim, f.dim)

		cosSlice := unsafe.Add(f.cosTab.DevicePtr(), uintptr(cosOff*4))
		sinSlice := unsafe.Add(f.sinTab.DevicePtr(), uintptr(sinOff*4))
		KRoPE(qP, cosSlice, sinSlice, 1, f.dim, f.headDim, f.nHeads)
		KRoPE(kP, cosSlice, sinSlice, 1, f.kvDim, f.headDim, f.nKVHeads)

		KKVCacheWrite(f.kCache[l].DevicePtr(), kP, pos, f.kvDim)
		KKVCacheWrite(f.vCache[l].DevicePtr(), vP, pos, f.kvDim)

		KDecodeAttention(qP, f.kCache[l].DevicePtr(), f.vCache[l].DevicePtr(), aP,
			seqLen, f.dim, f.kvDim, f.nHeads, f.nKVHeads)

		KQ8Matvec(aP, f.wo[l].DevicePtr(), f.woS[l].DevicePtr(), pP, f.dim, f.dim)
		KAddInPlace(hP, pP, f.dim)

		KRMSNormOut(hP, n2P, f.norm2[l].DevicePtr(), 1, f.dim)

		KQ8Matvec(n2P, f.wgate[l].DevicePtr(), f.wgateS[l].DevicePtr(), gP, f.ffnDim, f.dim)
		KQ8Matvec(n2P, f.wup[l].DevicePtr(), f.wupS[l].DevicePtr(), uP, f.ffnDim, f.dim)
		KSiLUGateMul(gP, uP, fP, f.ffnDim)
		KQ8Matvec(fP, f.wdown[l].DevicePtr(), f.wdownS[l].DevicePtr(), pP, f.dim, f.ffnDim)
		KAddInPlace(hP, pP, f.dim)
	}

	if layerEnd >= f.nLayers && logitsOut != nil && f.finalNorm != nil {
		KRMSNormOut(hP, nP, f.finalNorm.DevicePtr(), 1, f.dim)
		KQ8Matvec(nP, f.lmHead.DevicePtr(), f.lmHeadS.DevicePtr(), f.logits.DevicePtr(), f.vocabSize, f.dim)
		KSync()
		copy(logitsOut, f.eng.ToHost(f.logits))
	}

	if hiddenOut != nil && layerEnd < f.nLayers {
		KSync()
		copy(hiddenOut, f.eng.ToHost(f.hidden[f.hiddenIdx]))
	}

	return 0
}

// PartialStepContinue runs layers without uploading hiddenIn — assumes
// f.hidden[f.hiddenIdx] already holds the correct GPU-resident state from a prior call.
func (f *CUDAFusedInference) PartialStepContinue(pos, layerStart, layerEnd int, hiddenOut, logitsOut []float32) int {
	if !f.built { return -1 }

	s := &f.scratch[0]
	hP := f.hidden[f.hiddenIdx].DevicePtr()
	nP := s.normed.DevicePtr()
	qP := s.Q.DevicePtr()
	kP := s.K.DevicePtr()
	vP := s.V.DevicePtr()
	aP := s.attnOut.DevicePtr()
	pP := s.proj.DevicePtr()
	n2P := s.normed2.DevicePtr()
	gP := s.gatePre.DevicePtr()
	uP := s.upOut.DevicePtr()
	fP := s.ffnMid.DevicePtr()

	cosOff := pos * f.halfHead
	sinOff := pos * f.halfHead
	seqLen := pos + 1

	for l := layerStart; l < layerEnd && l < f.nLayers; l++ {
		if f.norm1[l] == nil { continue }

		KRMSNormOut(hP, nP, f.norm1[l].DevicePtr(), 1, f.dim)

		KQ8Matvec(nP, f.wq[l].DevicePtr(), f.wqS[l].DevicePtr(), qP, f.dim, f.dim)
		KQ8Matvec(nP, f.wk[l].DevicePtr(), f.wkS[l].DevicePtr(), kP, f.kvDim, f.dim)
		KQ8Matvec(nP, f.wv[l].DevicePtr(), f.wvS[l].DevicePtr(), vP, f.kvDim, f.dim)

		cosSlice := unsafe.Add(f.cosTab.DevicePtr(), uintptr(cosOff*4))
		sinSlice := unsafe.Add(f.sinTab.DevicePtr(), uintptr(sinOff*4))
		KRoPE(qP, cosSlice, sinSlice, 1, f.dim, f.headDim, f.nHeads)
		KRoPE(kP, cosSlice, sinSlice, 1, f.kvDim, f.headDim, f.nKVHeads)

		KKVCacheWrite(f.kCache[l].DevicePtr(), kP, pos, f.kvDim)
		KKVCacheWrite(f.vCache[l].DevicePtr(), vP, pos, f.kvDim)

		KDecodeAttention(qP, f.kCache[l].DevicePtr(), f.vCache[l].DevicePtr(), aP,
			seqLen, f.dim, f.kvDim, f.nHeads, f.nKVHeads)

		KQ8Matvec(aP, f.wo[l].DevicePtr(), f.woS[l].DevicePtr(), pP, f.dim, f.dim)
		KAddInPlace(hP, pP, f.dim)

		KRMSNormOut(hP, n2P, f.norm2[l].DevicePtr(), 1, f.dim)

		KQ8Matvec(n2P, f.wgate[l].DevicePtr(), f.wgateS[l].DevicePtr(), gP, f.ffnDim, f.dim)
		KQ8Matvec(n2P, f.wup[l].DevicePtr(), f.wupS[l].DevicePtr(), uP, f.ffnDim, f.dim)
		KSiLUGateMul(gP, uP, fP, f.ffnDim)
		KQ8Matvec(fP, f.wdown[l].DevicePtr(), f.wdownS[l].DevicePtr(), pP, f.dim, f.ffnDim)
		KAddInPlace(hP, pP, f.dim)
	}

	if layerEnd >= f.nLayers && logitsOut != nil && f.finalNorm != nil {
		KRMSNormOut(hP, nP, f.finalNorm.DevicePtr(), 1, f.dim)
		KQ8Matvec(nP, f.lmHead.DevicePtr(), f.lmHeadS.DevicePtr(), f.logits.DevicePtr(), f.vocabSize, f.dim)
		KSync()
		copy(logitsOut, f.eng.ToHost(f.logits))
	}

	if hiddenOut != nil && layerEnd < f.nLayers {
		KSync()
		copy(hiddenOut, f.eng.ToHost(f.hidden[f.hiddenIdx]))
	}

	return 0
}

func (f *CUDAFusedInference) ResetKV() {
	for l := 0; l < f.nLayers; l++ {
		if f.kCache[l] != nil {
			zeros := make([]float32, f.maxSeq*f.kvDim)
			f.eng.UploadInto(f.kCache[l], zeros)
		}
	}
}

// SetLayerRange configures which layers this node owns and allocates KV caches
// only for those layers. Must be called before inference. Saves VRAM on nodes
// that only load a subset of layers (e.g., Sage with 30/80 layers).
func (f *CUDAFusedInference) SetLayerRange(layerStart, layerEnd int) {
	allocated := 0
	for l := layerStart; l < layerEnd && l < f.nLayers; l++ {
		if f.kCache[l] == nil {
			f.kCache[l] = f.eng.Zeros([]int{f.maxSeq * f.kvDim})
			f.vCache[l] = f.eng.Zeros([]int{f.maxSeq * f.kvDim})
			allocated++
		}
	}
	log.Printf("[VRAM] KV caches allocated for layers %d-%d (%d layers, %.1f MB)",
		layerStart, layerEnd-1, allocated,
		float64(allocated)*2*float64(f.maxSeq*f.kvDim*4)/1e6)
}

// SignalMailbox dispatches the mailbox signal kernel on stream 0.
// Must be called after the last compute kernel of a pipeline stage.
func (f *CUDAFusedInference) SignalMailbox() {
	if f.hiddenL3 == nil {
		return
	}
	KSignalMailbox(f.hiddenL3.DevicePtr(0))
}

// SignalMailboxOnStream1 dispatches the mailbox signal on computeStream1.
func (f *CUDAFusedInference) SignalMailboxOnStream1() {
	if f.hiddenL3 == nil {
		return
	}
	KSignalMailboxOnStream(f.hiddenL3.DevicePtr(0), f.ComputeStream1Ptr())
}

// WaitMailbox polls the L3 mailbox counter until it reaches the expected value.
// Returns when the GPU has signaled completion of the current pipeline stage.
func (f *CUDAFusedInference) WaitMailbox() {
	if f.hiddenL3 == nil {
		KSync()
		return
	}
	ptr := (*uint32)(f.hiddenL3.HostPtr(0))
	expected := f.mailboxExpected
	for {
		val := *(*uint32)(unsafe.Pointer(ptr))
		if val >= expected {
			break
		}
		// Yield to Go scheduler — ~1µs per check, ~11ms total wait
	}
	f.mailboxExpected++
}

// HasMailbox returns true if the mailbox signal kernel is loaded and L3 bridge available.
func (f *CUDAFusedInference) HasMailbox() bool {
	return f.hiddenL3 != nil && HasSignalMailbox()
}

// ReadHiddenBlocking does a synchronous D2H copy of hidden state to a host buffer.
// Safe to call after WaitMailbox() confirms Stage A is done and before Stage B dispatch.
func (f *CUDAFusedInference) ReadHiddenBlocking(dst []float32) {
	KSync()
	copy(dst, f.eng.ToHost(f.hidden[f.hiddenIdx]))
}

// HiddenIdx returns the current hidden buffer index (0 or 1).
func (f *CUDAFusedInference) HiddenIdx() int { return f.hiddenIdx }

// ComputeStream1Ptr returns the raw CUDA stream pointer for Stage B dispatch.
func (f *CUDAFusedInference) ComputeStream1Ptr() unsafe.Pointer {
	return unsafe.Pointer(f.computeStream1)
}

// allocScratch allocates a complete set of scratch buffers for vNode index i.
func (f *CUDAFusedInference) allocScratch(i, q8ScratchSize int) {
	s := &f.scratch[i]
	s.normed = f.eng.Zeros([]int{f.dim})
	s.Q = f.eng.Zeros([]int{f.dim})
	s.K = f.eng.Zeros([]int{f.kvDim})
	s.V = f.eng.Zeros([]int{f.kvDim})
	s.attnOut = f.eng.Zeros([]int{f.dim})
	s.proj = f.eng.Zeros([]int{f.dim})
	s.normed2 = f.eng.Zeros([]int{f.dim})
	s.gatePre = f.eng.Zeros([]int{f.ffnDim})
	s.upOut = f.eng.Zeros([]int{f.ffnDim})
	s.ffnMid = f.eng.Zeros([]int{f.ffnDim})
	s.q8Scratch = C.tw_gpu_alloc(C.size_t(q8ScratchSize))
	vramKB := (f.dim*4*7 + f.kvDim*4*2 + f.ffnDim*4*3 + q8ScratchSize) / 1024
	log.Printf("[VNODE] scratch[%d] allocated: %d KB VRAM", i, vramKB)
}

// SetVNodeCount configures how many vNodes this inference instance uses.
// 1 = single stream (default), 2 = dual compute streams (Stage A + Stage B).
// When set to 2, allocates independent scratch buffers for Stage B.
func (f *CUDAFusedInference) SetVNodeCount(n int) {
	if n < 1 { n = 1 }
	if n > 2 { n = 2 }
	f.vNodeCount = n
	if n >= 2 && f.scratch[1].Q == nil {
		q8MaxK := f.dim
		if f.ffnDim > q8MaxK { q8MaxK = f.ffnDim }
		q8ScratchSize := (q8MaxK / 32) * 40
		f.allocScratch(1, q8ScratchSize)
	}
	log.Printf("[VNODE] vNodeCount set to %d", n)
}

// VNodeCount returns the configured vNode count.
func (f *CUDAFusedInference) VNodeCount() int {
	return f.vNodeCount
}

// RecordStageDone records the stageDone event on the default compute stream.
// Stream1 can then wait on this event before starting Stage B.
func (f *CUDAFusedInference) RecordStageDone() {
	C.tw_event_record(f.stageDone, nil)
}

// WaitStageDone makes computeStream1 wait for the stageDone event.
// Call this before dispatching Stage B kernels on stream1.
func (f *CUDAFusedInference) WaitStageDone() {
	C.tw_stream_wait_event(f.computeStream1, f.stageDone)
}

// SyncStream1 blocks until all work on computeStream1 completes.
func (f *CUDAFusedInference) SyncStream1() {
	C.tw_stream_sync(f.computeStream1)
}

// QueueHiddenToL3OnStream1 queues the L3 copy from computeStream1 instead
// of the default stream. Used when Stage B (the last compute stage) runs on stream1.
func (f *CUDAFusedInference) QueueHiddenToL3OnStream1() {
	if f.hiddenL3 == nil {
		return
	}
	C.tw_event_record(f.computeDone, f.computeStream1)
	C.tw_stream_wait_event(f.copyStream, f.computeDone)
	C.tw_copy_to_l3_on_stream(
		f.hiddenL3.DevicePtr(64),
		f.hidden[f.hiddenIdx].DevicePtr(),
		C.size_t(f.dim*4),
		f.copyStream)
}

// QueueHiddenToL3 queues an async copy of the GPU-resident hidden state
// to the L3 pinned buffer via a separate CUDA stream. The compute stream
// continues unblocked. Call WaitL3() before reading GetL3Hidden().
func (f *CUDAFusedInference) QueueHiddenToL3() {
	if f.hiddenL3 == nil {
		return
	}
	// Record event on compute stream (stream 0) — marks compute complete
	C.tw_event_record(f.computeDone, nil)
	// Copy stream waits for compute to finish
	C.tw_stream_wait_event(f.copyStream, f.computeDone)
	// Async D2H copy on the copy stream (doesn't block compute)
	C.tw_copy_to_l3_on_stream(
		f.hiddenL3.DevicePtr(64),
		f.hidden[f.hiddenIdx].DevicePtr(),
		C.size_t(f.dim*4),
		f.copyStream)
}

// WaitL3 blocks until the L3 copy completes. Only synchronizes the copy
// stream — compute stream continues unblocked for next iteration.
var l3WaitLogged int
func (f *CUDAFusedInference) WaitL3() {
	if f.hiddenL3 == nil {
		KSync()
		return
	}
	t := time.Now()
	C.tw_stream_sync(f.copyStream)
	if l3WaitLogged < 5 {
		log.Printf("[L3] WaitL3 took %.2fms (copyStream sync)", float64(time.Since(t).Microseconds())/1000)
		l3WaitLogged++
	}
}

// GetL3Hidden returns the hidden state from the L3 pinned buffer as a
// CPU-accessible []float32 slice. Only valid after QueueHiddenToL3 + WaitL3.
func (f *CUDAFusedInference) GetL3Hidden() []float32 {
	if f.hiddenL3 == nil {
		return nil
	}
	return f.hiddenL3.Float32(64, f.dim)
}

// HasL3 reports whether the L3 bridge is available for async transfers.
func (f *CUDAFusedInference) HasL3() bool {
	return f.hiddenL3 != nil
}

// SwapHiddenBuffer alternates the GPU hidden buffer index. Call after
// WaitL3+send to ensure the next tick's UploadInto writes to a different
// buffer than the one the async L3 copy read from.
func (f *CUDAFusedInference) SwapHiddenBuffer() {
	f.hiddenIdx = 1 - f.hiddenIdx
}

func (f *CUDAFusedInference) DisableFusedDispatch() {
	f.fusedReady = false
	log.Printf("[FUSED-DISPATCH] disabled — using individual kernel calls")
}

func (f *CUDAFusedInference) BuildFusedDispatch() {
	if !HasFusedDispatch() {
		return
	}
	n := f.nLayers
	devPtr := func(t *Tensor) unsafe.Pointer {
		if t == nil {
			return nil
		}
		return t.DevicePtr()
	}
	f.norm1Ptrs = make([]unsafe.Pointer, n)
	f.norm2Ptrs = make([]unsafe.Pointer, n)
	f.wqPtrs = make([]unsafe.Pointer, n)
	f.wkPtrs = make([]unsafe.Pointer, n)
	f.wvPtrs = make([]unsafe.Pointer, n)
	f.woPtrs = make([]unsafe.Pointer, n)
	f.wgatePtrs = make([]unsafe.Pointer, n)
	f.wupPtrs = make([]unsafe.Pointer, n)
	f.wdownPtrs = make([]unsafe.Pointer, n)
	f.kCachePtrs = make([]unsafe.Pointer, n)
	f.vCachePtrs = make([]unsafe.Pointer, n)
	for l := 0; l < n; l++ {
		f.norm1Ptrs[l] = devPtr(f.norm1[l])
		f.norm2Ptrs[l] = devPtr(f.norm2[l])
		f.wqPtrs[l] = devPtr(f.wq[l])
		f.wkPtrs[l] = devPtr(f.wk[l])
		f.wvPtrs[l] = devPtr(f.wv[l])
		f.woPtrs[l] = devPtr(f.wo[l])
		f.wgatePtrs[l] = devPtr(f.wgate[l])
		f.wupPtrs[l] = devPtr(f.wup[l])
		f.wdownPtrs[l] = devPtr(f.wdown[l])
		f.kCachePtrs[l] = devPtr(f.kCache[l])
		f.vCachePtrs[l] = devPtr(f.vCache[l])
	}
	f.fusedReady = true
	log.Printf("[FUSED-DISPATCH] ready: %d layers, %d-dim, %d CGo calls → 1", n, f.dim, n*17)
}
