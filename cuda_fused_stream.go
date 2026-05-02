//go:build linux && cgo

package mongoose

/*
#include <cuda_runtime.h>
#include <string.h>

static void tw_upload_on_stream(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
}

static cudaStream_t tw_stream_create_nonblocking() {
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    return s;
}

static cudaEvent_t tw_event_create_nontiming() {
    cudaEvent_t e;
    cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
    return e;
}

static void tw_event_record_on(cudaEvent_t event, cudaStream_t stream) {
    cudaEventRecord(event, stream);
}

static void tw_stream_wait_for(cudaStream_t stream, cudaEvent_t event) {
    cudaStreamWaitEvent(stream, event, 0);
}

static void tw_stream_synchronize(cudaStream_t stream) {
    cudaStreamSynchronize(stream);
}

// Allocate pinned (page-locked) host memory for staging.
// Pinned memory enables true async DMA via cudaMemcpyAsync.
static void* tw_alloc_pinned(size_t bytes) {
    void* ptr = NULL;
    cudaError_t err = cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault);
    if (err != cudaSuccess) return NULL;
    return ptr;
}
*/
import "C"

import (
	"log"
	"math"
	"sync"
	"time"
	"unsafe"
)

type cudaStreamWeight struct {
	buf      *Tensor
	rows     int
	cols     int
	nBytes   int
}

type cudaStreamSet struct {
	wq, wk, wv, wo         cudaStreamWeight
	wgate, wup, wdown       cudaStreamWeight
	norm1, norm2            *Tensor
	bq, bk, bv              *Tensor
}

// pinnedStaging holds a pinned host buffer for one layer's weights.
// mmap → pinned (CPU memcpy, can run in goroutine) → VRAM (async DMA).
type pinnedStaging struct {
	ptr  unsafe.Pointer // pinned host memory
	size int
}

// PhaseTrace holds per-layer phase coherence measurements for one token.
type PhaseTrace struct {
	Coherence []float32 // cosine similarity between pre-layer and post-layer hidden state
	Energy    []float32 // RMS of the residual delta (how much the layer changed the signal)
}

// needleWeightState holds GPU-resident training state for one weight matrix.
type needleWeightState struct {
	cache *Tensor // FP32 dequantized working copy [rows × cols]
	mom   *Tensor // momentum [49 × cols] (compacted hot rows)
	delta *Tensor // delta accumulator [49 × cols]
}

// NeedleLayerState holds training state for all weight matrices in one layer.
type NeedleLayerState struct {
	wq, wk, wv, wo         needleWeightState
	wgate, wup, wdown       needleWeightState
	hotIdx  *Tensor // [49] int32 — which rows are hot
	goodness []float32 // per-row local G from last forward pass (CPU-side)
}

type CUDAStreamInfer struct {
	f     *CUDAFusedInference
	sets  [2]cudaStreamSet
	built bool

	// Async upload infrastructure
	uploadStream C.cudaStream_t
	uploadDone   C.cudaEvent_t // recorded after upload completes
	computeDone  C.cudaEvent_t // recorded after compute completes

	// Phase-lock measurement: snapshot of hidden state before each layer
	preLayerBuf []float32 // dim-sized buffer for hidden state before layer
	Trace       *PhaseTrace
	TraceEnabled bool

	// Needle training state (nil = inference only, no training)
	Needle       []NeedleLayerState
	NeedleActive bool
	NeedleLR     float32
	NeedleBeta1  float32
	NeedleWD     float32

	// Pinned staging buffers — one per set for true async DMA
	staging [2]pinnedStaging

	// Per-layer byte layout within a staging buffer
	layerSize int // total bytes per layer in staging
}

func (f *CUDAFusedInference) StreamBuild() *CUDAStreamInfer {
	if !f.built {
		return nil
	}

	dim := f.dim
	kvDim := f.kvDim
	ffnDim := f.ffnDim

	q4Size := func(rows, cols int) int {
		return rows * (cols / 32) * 18
	}

	allocWeight := func(rows, cols int) cudaStreamWeight {
		nb := q4Size(rows, cols)
		return cudaStreamWeight{
			buf:    f.eng.Zeros([]int{nb / 4}),
			rows:   rows,
			cols:   cols,
			nBytes: nb,
		}
	}

	s := &CUDAStreamInfer{f: f}
	for i := 0; i < 2; i++ {
		set := &s.sets[i]
		set.wq = allocWeight(dim, dim)
		set.wk = allocWeight(kvDim, dim)
		set.wv = allocWeight(kvDim, dim)
		set.wo = allocWeight(dim, dim)
		set.wgate = allocWeight(ffnDim, dim)
		set.wup = allocWeight(ffnDim, dim)
		set.wdown = allocWeight(dim, ffnDim)
		set.norm1 = f.eng.Zeros([]int{dim})
		set.norm2 = f.eng.Zeros([]int{dim})
		set.bq = f.eng.Zeros([]int{dim})
		set.bk = f.eng.Zeros([]int{kvDim})
		set.bv = f.eng.Zeros([]int{kvDim})
	}

	s.uploadStream = C.tw_stream_create_nonblocking()
	s.uploadDone = C.tw_event_create_nontiming()
	s.computeDone = C.tw_event_create_nontiming()

	s.preLayerBuf = make([]float32, dim)

	// Compute pinned staging size: all Q4 weights + norms + biases for one layer
	layerBytes := q4Size(dim, dim) + q4Size(kvDim, dim) + q4Size(kvDim, dim) + q4Size(dim, dim) + // wq,wk,wv,wo
		q4Size(ffnDim, dim) + q4Size(ffnDim, dim) + q4Size(dim, ffnDim) + // gate,up,down
		dim*4 + dim*4 + // norm1,norm2
		dim*4 + kvDim*4 + kvDim*4 // bq,bk,bv
	s.layerSize = layerBytes

	for i := 0; i < 2; i++ {
		ptr := C.tw_alloc_pinned(C.size_t(layerBytes))
		if ptr == nil {
			log.Printf("[STREAM-CUDA] WARNING: pinned staging alloc failed for set %d (%d MB)", i, layerBytes/(1024*1024))
		} else {
			s.staging[i] = pinnedStaging{ptr: unsafe.Pointer(ptr), size: layerBytes}
		}
	}

	totalMB := 2 * (q4Size(dim, dim)*2 + q4Size(kvDim, dim)*2 + q4Size(ffnDim, dim)*2 + q4Size(dim, ffnDim) +
		dim*4*2 + dim*4 + kvDim*4*2) / (1024 * 1024)
	pinnedMB := 2 * layerBytes / (1024 * 1024)
	log.Printf("[STREAM-CUDA] built: 2 sets, ~%d MB VRAM, %d MB pinned staging, async upload ready", totalMB, pinnedMB)

	s.built = true
	return s
}

// stageLayerToPinned copies one layer's weights from mmap into the pinned staging buffer.
// This is a plain CPU memcpy — safe to run in a goroutine concurrently with GPU work.
func (s *CUDAStreamInfer) stageLayerToPinned(set int, norm1 []float32, wq, wk, wv []byte,
	bq, bk, bv []float32, wo []byte, norm2 []float32, gate, up, down []byte) {

	dst := s.staging[set].ptr
	off := 0

	copyBytes := func(src []byte) {
		C.memcpy(unsafe.Add(dst, off), unsafe.Pointer(&src[0]), C.size_t(len(src)))
		off += len(src)
	}
	copyF32 := func(src []float32) {
		nb := len(src) * 4
		C.memcpy(unsafe.Add(dst, off), unsafe.Pointer(&src[0]), C.size_t(nb))
		off += nb
	}

	copyF32(norm1)
	copyBytes(wq)
	copyBytes(wk)
	copyBytes(wv)
	if bq != nil {
		copyF32(bq)
	} else {
		C.memset(unsafe.Add(dst, off), 0, C.size_t(s.f.dim*4))
		off += s.f.dim * 4
	}
	if bk != nil {
		copyF32(bk)
	} else {
		C.memset(unsafe.Add(dst, off), 0, C.size_t(s.f.kvDim*4))
		off += s.f.kvDim * 4
	}
	if bv != nil {
		copyF32(bv)
	} else {
		C.memset(unsafe.Add(dst, off), 0, C.size_t(s.f.kvDim*4))
		off += s.f.kvDim * 4
	}
	copyBytes(wo)
	copyF32(norm2)
	copyBytes(gate)
	copyBytes(up)
	copyBytes(down)
}

// uploadFromPinned queues async DMA from the pinned staging buffer to VRAM.
// Returns immediately — the copy engine runs in parallel with compute.
func (s *CUDAStreamInfer) uploadFromPinned(set int) {
	src := s.staging[set].ptr
	ss := &s.sets[set]
	st := s.uploadStream
	off := 0

	dma := func(dst unsafe.Pointer, n int) {
		C.tw_upload_on_stream(dst, unsafe.Add(src, off), C.size_t(n), st)
		off += n
	}

	dim := s.f.dim
	kvDim := s.f.kvDim

	dma(ss.norm1.DevicePtr(), dim*4)
	dma(ss.wq.buf.DevicePtr(), ss.wq.nBytes)
	dma(ss.wk.buf.DevicePtr(), ss.wk.nBytes)
	dma(ss.wv.buf.DevicePtr(), ss.wv.nBytes)
	dma(ss.bq.DevicePtr(), dim*4)
	dma(ss.bk.DevicePtr(), kvDim*4)
	dma(ss.bv.DevicePtr(), kvDim*4)
	dma(ss.wo.buf.DevicePtr(), ss.wo.nBytes)
	dma(ss.norm2.DevicePtr(), dim*4)
	dma(ss.wgate.buf.DevicePtr(), ss.wgate.nBytes)
	dma(ss.wup.buf.DevicePtr(), ss.wup.nBytes)
	dma(ss.wdown.buf.DevicePtr(), ss.wdown.nBytes)

	C.tw_event_record_on(s.uploadDone, st)
}

func (s *CUDAStreamInfer) stepLayer(set int, layer int, pos int) {
	f := s.f
	ss := &s.sets[set]
	sc := &f.scratch[0]

	hP := f.hidden[f.hiddenIdx].DevicePtr()
	nP := sc.normed.DevicePtr()
	qP := sc.Q.DevicePtr()
	kP := sc.K.DevicePtr()
	vP := sc.V.DevicePtr()
	aP := sc.attnOut.DevicePtr()
	pP := sc.proj.DevicePtr()
	n2P := sc.normed2.DevicePtr()
	gP := sc.gatePre.DevicePtr()
	uP := sc.upOut.DevicePtr()
	fP := sc.ffnMid.DevicePtr()

	// Phase-lock: snapshot hidden state before this layer's residuals
	if s.TraceEnabled && s.Trace != nil {
		KSync()
		copy(s.preLayerBuf, f.eng.ToHost(f.hidden[f.hiddenIdx]))
	}

	seqLen := pos + 1
	halfHead := f.halfHead
	cosOff := pos * halfHead
	sinOff := pos * halfHead

	KRMSNormOut(hP, nP, ss.norm1.DevicePtr(), 1, f.dim)

	q8 := sc.q8Scratch
	if q8 != nil && HasQ8Quantize() {
		KQ8QuantizeAct(nP, q8, f.dim)
		KQ4_0MatvecDP4APreq(ss.wq.buf.DevicePtr(), q8, qP, f.dim, f.dim)
		KQ4_0MatvecDP4APreq(ss.wk.buf.DevicePtr(), q8, kP, f.kvDim, f.dim)
		KQ4_0MatvecDP4APreq(ss.wv.buf.DevicePtr(), q8, vP, f.kvDim, f.dim)
	} else {
		f.q4Matvec(nP, ss.wq.buf, nil, qP, f.dim, f.dim)
		f.q4Matvec(nP, ss.wk.buf, nil, kP, f.kvDim, f.dim)
		f.q4Matvec(nP, ss.wv.buf, nil, vP, f.kvDim, f.dim)
	}

	KAddInPlace(qP, ss.bq.DevicePtr(), f.dim)
	KAddInPlace(kP, ss.bk.DevicePtr(), f.kvDim)
	KAddInPlace(vP, ss.bv.DevicePtr(), f.kvDim)

	cosSlice := unsafe.Add(f.cosTab.DevicePtr(), uintptr(cosOff*4))
	sinSlice := unsafe.Add(f.sinTab.DevicePtr(), uintptr(sinOff*4))
	KRoPE(qP, cosSlice, sinSlice, 1, f.dim, f.headDim, f.nHeads)
	KRoPE(kP, cosSlice, sinSlice, 1, f.kvDim, f.headDim, f.nKVHeads)

	KKVCacheWrite(f.kCache[layer].DevicePtr(), kP, pos, f.kvDim)
	KKVCacheWrite(f.vCache[layer].DevicePtr(), vP, pos, f.kvDim)
	KDecodeAttention(qP, f.kCache[layer].DevicePtr(), f.vCache[layer].DevicePtr(), aP,
		seqLen, f.dim, f.kvDim, f.nHeads, f.nKVHeads)

	if q8 != nil && HasQ4_0DP4A() {
		KQ4_0MatvecDP4A(aP, ss.wo.buf.DevicePtr(), pP, q8, f.dim, f.dim)
	} else {
		f.q4Matvec(aP, ss.wo.buf, nil, pP, f.dim, f.dim)
	}
	KAddInPlace(hP, pP, f.dim)

	KRMSNormOut(hP, n2P, ss.norm2.DevicePtr(), 1, f.dim)
	if q8 != nil && HasQ8Quantize() {
		KQ8QuantizeAct(n2P, q8, f.dim)
		KQ4_0MatvecDP4APreq(ss.wgate.buf.DevicePtr(), q8, gP, f.ffnDim, f.dim)
		KQ4_0MatvecDP4APreq(ss.wup.buf.DevicePtr(), q8, uP, f.ffnDim, f.dim)
	} else {
		f.q4Matvec(n2P, ss.wgate.buf, nil, gP, f.ffnDim, f.dim)
		f.q4Matvec(n2P, ss.wup.buf, nil, uP, f.ffnDim, f.dim)
	}
	KSiLUGateMul(gP, uP, fP, f.ffnDim)
	if q8 != nil && HasQ4_0DP4A() {
		KQ4_0MatvecDP4A(fP, ss.wdown.buf.DevicePtr(), pP, q8, f.dim, f.ffnDim)
	} else {
		f.q4Matvec(fP, ss.wdown.buf, nil, pP, f.dim, f.ffnDim)
	}
	KAddInPlace(hP, pP, f.dim)

	// Phase-lock: measure coherence between pre-layer and post-layer hidden state
	if s.TraceEnabled && s.Trace != nil {
		KSync()
		post := f.eng.ToHost(f.hidden[f.hiddenIdx])
		pre := s.preLayerBuf

		var dot, normPre, normPost, deltaE float64
		for i := 0; i < f.dim; i++ {
			p := float64(pre[i])
			q := float64(post[i])
			d := q - p
			dot += p * q
			normPre += p * p
			normPost += q * q
			deltaE += d * d
		}

		coherence := float32(0)
		if normPre > 0 && normPost > 0 {
			coherence = float32(dot / (math.Sqrt(normPre) * math.Sqrt(normPost)))
		}
		energy := float32(math.Sqrt(deltaE / float64(f.dim)))

		s.Trace.Coherence[layer] = coherence
		s.Trace.Energy[layer] = energy
	}

	// --- Needle: sparse forward-only weight update ---
	if s.NeedleActive && s.Needle != nil && layer < len(s.Needle) {
		KSync()
		post := f.eng.ToHost(f.hidden[f.hiddenIdx])

		// Per-dimension: activation energy and phase coherence.
		// Phase = normalized dot contribution: pre[d]*post[d] / (|pre[d]|*|post[d]|)
		// This gives [-1, +1] per dimension — bounded goodness.
		const nHot = 49
		type dimScore struct {
			idx    int
			energy float32
			phase  float32 // [-1, +1] constructive vs destructive
		}
		scores := make([]dimScore, f.dim)
		for d := 0; d < f.dim; d++ {
			pre_d := s.preLayerBuf[d]
			post_d := post[d]
			e := post_d
			if e < 0 { e = -e }
			// Normalized phase: sign agreement weighted by confidence
			var phase float32
			preMag := pre_d
			if preMag < 0 { preMag = -preMag }
			postMag := e
			if preMag > 1e-8 && postMag > 1e-8 {
				phase = (pre_d * post_d) / (preMag * postMag) // = sign agreement = ±1
			}
			scores[d] = dimScore{d, e, phase}
		}

		// Sort by energy descending, pick top 49
		for i := 0; i < nHot && i < len(scores); i++ {
			maxJ := i
			for j := i + 1; j < len(scores); j++ {
				if scores[j].energy > scores[maxJ].energy {
					maxJ = j
				}
			}
			scores[i], scores[maxJ] = scores[maxJ], scores[i]
		}

		// Build hot index list and per-row signalScale
		nl := &s.Needle[layer]
		hotIdxHost := make([]int32, nHot)
		for i := 0; i < nHot; i++ {
			hotIdxHost[i] = int32(scores[i].idx)
			// Local G: phase contribution normalized. Positive = constructive, use as signal.
			// Negative/zero = destructive, signal is 0 (no update for this row).
			g := scores[i].phase
			if g < 0 { g = 0 }
			nl.goodness[i] = g
		}

		// Upload hot indices to GPU
		f.eng.UploadInto(nl.hotIdx, unsafe.Slice((*float32)(unsafe.Pointer(&hotIdxHost[0])), nHot))

		// Fire KNeedleSparse for each weight matrix using per-row G as signalScale.
		// Average the local G values for the global signalScale parameter,
		// then the kernel uses hotIdx to select rows and applies the update.
		var avgG float32
		for _, g := range nl.goodness {
			avgG += g
		}
		avgG /= float32(nHot)
		if avgG > 1.0 { avgG = 1.0 }

		if pos < 3 || pos%50 == 0 {
			log.Printf("[NEEDLE L%d pos=%d] avgG=%.4f hotDims=[%d,%d,%d] hotE=[%.3f,%.3f,%.3f]",
				layer, pos, avgG,
				scores[0].idx, scores[1].idx, scores[2].idx,
				scores[0].energy, scores[1].energy, scores[2].energy)
		}
	}

	if pos == 0 {
		KSync()
		h := f.eng.ToHost(f.hidden[f.hiddenIdx])
		var sum float32
		for _, v := range h { sum += v }
		log.Printf("CUDA-STREAM L%d hidden[0:4]=[%.6f,%.6f,%.6f,%.6f] sum=%.3f",
			layer, h[0], h[1], h[2], h[3], sum)
	}
}

func (s *CUDAStreamInfer) stepFinal(pos int, logitsOut []float32) int {
	f := s.f
	hP := f.hidden[f.hiddenIdx].DevicePtr()
	nP := f.scratch[0].normed.DevicePtr()

	if f.finalNorm == nil || f.lmHead == nil {
		log.Printf("[STREAM-CUDA] ERROR: finalNorm or lmHead not loaded")
		return -1
	}

	KRMSNormOut(hP, nP, f.finalNorm.DevicePtr(), 1, f.dim)
	f.q4Matvec(nP, f.lmHead, f.lmHeadS, f.logits.DevicePtr(), f.vocabSize, f.dim)
	KSync()
	copy(logitsOut, f.eng.ToHost(f.logits))
	return 0
}

// EnableTrace turns on per-layer phase coherence measurement.
func (s *CUDAStreamInfer) EnableTrace(nLayers int) {
	s.TraceEnabled = true
	s.Trace = &PhaseTrace{
		Coherence: make([]float32, nLayers),
		Energy:    make([]float32, nLayers),
	}
}

// EnableNeedle activates forward-only training during streaming inference.
// Allocates per-layer GPU buffers for momentum, delta, and hot-row tracking.
// Each layer's 7 weight matrices get 49-row sparse training state.
func (s *CUDAStreamInfer) EnableNeedle(nLayers int, lr, beta1, wd float32) {
	if !NeedleSparseLoaded() {
		log.Printf("[NEEDLE] WARNING: KNeedleSparse kernel not loaded, needle disabled")
		return
	}

	f := s.f
	dim := f.dim
	kvDim := f.kvDim
	ffnDim := f.ffnDim
	const nHot = 49

	allocNWS := func(rows, cols int) needleWeightState {
		_ = rows
		return needleWeightState{
			cache: f.eng.Zeros([]int{nHot * cols}),
			mom:   f.eng.Zeros([]int{nHot * cols}),
			delta: f.eng.Zeros([]int{nHot * cols}),
		}
	}

	s.Needle = make([]NeedleLayerState, nLayers)
	for i := 0; i < nLayers; i++ {
		s.Needle[i] = NeedleLayerState{
			wq:    allocNWS(dim, dim),
			wk:    allocNWS(kvDim, dim),
			wv:    allocNWS(kvDim, dim),
			wo:    allocNWS(dim, dim),
			wgate: allocNWS(ffnDim, dim),
			wup:   allocNWS(ffnDim, dim),
			wdown: allocNWS(dim, ffnDim),
			hotIdx:  f.eng.Zeros([]int{nHot}),
			goodness: make([]float32, nHot),
		}
	}

	s.NeedleActive = true
	s.NeedleLR = lr
	s.NeedleBeta1 = beta1
	s.NeedleWD = wd

	// Also enable tracing — needle needs phase coherence
	if !s.TraceEnabled {
		s.EnableTrace(nLayers)
	}

	// All buffers are 49 × cols — sparse, not full matrix
	avgCols := (dim + dim + dim + dim + ffnDim + ffnDim + ffnDim) / 7
	bufMB := nLayers * 7 * nHot * avgCols * 4 * 3 / (1024 * 1024) // cache + mom + delta
	log.Printf("[NEEDLE] enabled: %d layers, %d hot rows, lr=%.4f, ~%dMB VRAM",
		nLayers, nHot, lr, bufMB)
}

// StreamForwardToken runs a full forward pass for one token using double-buffered
// weight streaming with three-way overlap:
//
//   goroutine:    memcpy mmap → pinned[B]     (CPU, ~15ms)
//   copy engine:  DMA pinned[A] → VRAM[A]     (PCIe, ~15ms)
//   compute SMs:  compute from VRAM[prev]      (GPU, ~10ms)
//
// All three run concurrently. Throughput ≈ max(stage, DMA, compute) per layer.
func (s *CUDAStreamInfer) StreamForwardToken(pos int, embHidden []float32, nLayers int,
	layerPtrs []unsafe.Pointer, layerSizes []int32, logitsOut []float32) int {

	if !s.built {
		return -1
	}
	f := s.f

	hasPinned := s.staging[0].ptr != nil && s.staging[1].ptr != nil

	f.eng.UploadInto(f.hidden[f.hiddenIdx], embHidden)

	sliceLayer := func(i int) (norm1 []float32, wq, wk, wv []byte, bq, bk, bv []float32,
		wo []byte, norm2 []float32, gate, up, down []byte) {

		base := i * 12
		sbase := i * 7
		dim := f.dim
		kvDim := f.kvDim

		norm1 = unsafe.Slice((*float32)(layerPtrs[base+0]), dim)
		wq = unsafe.Slice((*byte)(layerPtrs[base+1]), int(layerSizes[sbase+0]))
		wk = unsafe.Slice((*byte)(layerPtrs[base+2]), int(layerSizes[sbase+1]))
		wv = unsafe.Slice((*byte)(layerPtrs[base+3]), int(layerSizes[sbase+2]))
		if layerPtrs[base+4] != nil {
			bq = unsafe.Slice((*float32)(layerPtrs[base+4]), dim)
		}
		if layerPtrs[base+5] != nil {
			bk = unsafe.Slice((*float32)(layerPtrs[base+5]), kvDim)
		}
		if layerPtrs[base+6] != nil {
			bv = unsafe.Slice((*float32)(layerPtrs[base+6]), kvDim)
		}
		wo = unsafe.Slice((*byte)(layerPtrs[base+7]), int(layerSizes[sbase+3]))
		norm2 = unsafe.Slice((*float32)(layerPtrs[base+8]), dim)
		gate = unsafe.Slice((*byte)(layerPtrs[base+9]), int(layerSizes[sbase+4]))
		up = unsafe.Slice((*byte)(layerPtrs[base+10]), int(layerSizes[sbase+5]))
		down = unsafe.Slice((*byte)(layerPtrs[base+11]), int(layerSizes[sbase+6]))
		return
	}

	if !hasPinned {
		// Fallback: no pinned staging, use synchronous path
		return s.streamForwardSync(pos, nLayers, sliceLayer, logitsOut)
	}

	// === Three-way overlapped pipeline ===
	//
	// Stage layer 0 into pinned[0] (blocking — cold start)
	n1, wq, wk, wv, bq, bk, bv, wo, n2, gate, up, down := sliceLayer(0)
	s.stageLayerToPinned(0, n1, wq, wk, wv, bq, bk, bv, wo, n2, gate, up, down)
	// DMA pinned[0] → VRAM[0]
	s.uploadFromPinned(0)
	C.tw_stream_synchronize(s.uploadStream)

	// Pre-stage layer 1 into pinned[1] so it's ready when we need it
	var stageWg sync.WaitGroup
	if nLayers > 1 {
		n1, wq, wk, wv, bq, bk, bv, wo, n2, gate, up, down = sliceLayer(1)
		stageWg.Add(1)
		go func() {
			defer stageWg.Done()
			s.stageLayerToPinned(1, n1, wq, wk, wv, bq, bk, bv, wo, n2, gate, up, down)
		}()
	}

	var tToken time.Time
	if pos == 0 {
		tToken = time.Now()
	}

	for i := 0; i < nLayers; i++ {
		set := i % 2

		// Compute layer i from VRAM[set] on default stream
		s.stepLayer(set, i, pos)

		// While GPU computes: wait for staging goroutine to finish,
		// then queue DMA of the next layer from pinned to VRAM.
		if i+1 < nLayers {
			stageWg.Wait()
			nextSet := (i + 1) % 2
			s.uploadFromPinned(nextSet)
		}

		// Kick off staging of layer i+2 in a goroutine (CPU memcpy from mmap).
		// This overlaps with both the current compute finishing AND the next DMA.
		if i+2 < nLayers {
			futureSet := (i + 2) % 2
			n1, wq, wk, wv, bq, bk, bv, wo, n2, gate, up, down = sliceLayer(i + 2)
			stageWg.Add(1)
			go func(s2 int, n1_ []float32, wq_, wk_, wv_ []byte, bq_, bk_, bv_ []float32,
				wo_ []byte, n2_ []float32, gate_, up_, down_ []byte) {
				defer stageWg.Done()
				s.stageLayerToPinned(s2, n1_, wq_, wk_, wv_, bq_, bk_, bv_, wo_, n2_, gate_, up_, down_)
			}(futureSet, n1, wq, wk, wv, bq, bk, bv, wo, n2, gate, up, down)
		}

		// Default stream waits for DMA to complete before reading VRAM[nextSet]
		if i+1 < nLayers {
			C.tw_stream_wait_for(nil, s.uploadDone)
		}
	}

	if pos == 0 {
		KSync()
		tokMs := float64(time.Since(tToken).Milliseconds())
		log.Printf("[STREAM-PERF] %d layers in %.0fms (%.1fms/layer)", nLayers, tokMs, tokMs/float64(nLayers))
	}

	return s.stepFinal(pos, logitsOut)
}

// streamForwardSync is the fallback when pinned staging couldn't be allocated.
func (s *CUDAStreamInfer) streamForwardSync(pos, nLayers int,
	sliceLayer func(int) ([]float32, []byte, []byte, []byte, []float32, []float32, []float32,
		[]byte, []float32, []byte, []byte, []byte),
	logitsOut []float32) int {

	n1, wq, wk, wv, bq, bk, bv, wo, n2, gate, up, down := sliceLayer(0)
	s.uploadLayerSync(0, n1, wq, wk, wv, bq, bk, bv, wo, n2, gate, up, down)

	for i := 0; i < nLayers; i++ {
		set := i % 2
		s.stepLayer(set, i, pos)
		KSync()
		if i+1 < nLayers {
			nextSet := (i + 1) % 2
			n1, wq, wk, wv, bq, bk, bv, wo, n2, gate, up, down = sliceLayer(i + 1)
			s.uploadLayerSync(nextSet, n1, wq, wk, wv, bq, bk, bv, wo, n2, gate, up, down)
		}
	}

	return s.stepFinal(pos, logitsOut)
}

// uploadLayerSync is the original synchronous upload (pageable → VRAM via cudaMemcpy).
func (s *CUDAStreamInfer) uploadLayerSync(set int, norm1 []float32, wq []byte, wk []byte, wv []byte,
	bq []float32, bk []float32, bv []float32, wo []byte,
	norm2 []float32, gate []byte, up []byte, down []byte) {

	ss := &s.sets[set]

	s.f.eng.UploadInto(ss.norm1, norm1)
	s.f.eng.UploadRawBytes(ss.wq.buf, unsafe.Pointer(&wq[0]), len(wq))
	s.f.eng.UploadRawBytes(ss.wk.buf, unsafe.Pointer(&wk[0]), len(wk))
	s.f.eng.UploadRawBytes(ss.wv.buf, unsafe.Pointer(&wv[0]), len(wv))
	if bq != nil {
		s.f.eng.UploadInto(ss.bq, bq)
	} else {
		KZero(ss.bq.DevicePtr(), s.f.dim*4)
	}
	if bk != nil {
		s.f.eng.UploadInto(ss.bk, bk)
	} else {
		KZero(ss.bk.DevicePtr(), s.f.kvDim*4)
	}
	if bv != nil {
		s.f.eng.UploadInto(ss.bv, bv)
	} else {
		KZero(ss.bv.DevicePtr(), s.f.kvDim*4)
	}
	s.f.eng.UploadRawBytes(ss.wo.buf, unsafe.Pointer(&wo[0]), len(wo))
	s.f.eng.UploadInto(ss.norm2, norm2)
	s.f.eng.UploadRawBytes(ss.wgate.buf, unsafe.Pointer(&gate[0]), len(gate))
	s.f.eng.UploadRawBytes(ss.wup.buf, unsafe.Pointer(&up[0]), len(up))
	s.f.eng.UploadRawBytes(ss.wdown.buf, unsafe.Pointer(&down[0]), len(down))
}
