//go:build linux && cgo

package mongoose

/*
#include <dlfcn.h>
#include <stdio.h>

void tw_gpu_download(void* dst, const void* src, size_t bytes);

typedef int (*fn_q4_cublas_matvec)(const float*, const void*, void*, float*, int, int, void*);
static fn_q4_cublas_matvec k_q4_cublas = NULL;
extern void* tw_cublas_handle;

static void tw_load_q4_cublas() {
    if (!k_q4_cublas) {
        void* lib = dlopen("./libmongoose_kernels.so", RTLD_NOW | RTLD_NOLOAD);
        if (!lib) lib = dlopen("./libmongoose_kernels.so", RTLD_NOW);
        if (lib) {
            k_q4_cublas = (fn_q4_cublas_matvec)dlsym(lib, "tw_q4_cublas_matvec");
        }
    }
}

static int _cublas_logged = 0;
static int tw_call_q4_cublas(const float* act, const void* w, void* scratch, float* out, int N, int K) {
    tw_load_q4_cublas();
    if (!_cublas_logged) {
        _cublas_logged = 1;
        fprintf(stderr, "[Q4] cuBLAS path: %s (handle=%p)\n",
            k_q4_cublas ? "ACTIVE" : "FALLBACK (dlsym failed)",
            (void*)tw_cublas_handle);
    }
    if (!k_q4_cublas) return -1;
    return k_q4_cublas(act, w, scratch, out, N, K, (void*)tw_cublas_handle);
}
*/
import "C"

import (
	"log"
	"time"
	"unsafe"
)

func (f *CUDAFusedInference) SetWeightQ4_0(idx int, q4Data []byte) {
	nL := f.nLayers
	if idx < nL*12 {
		layer := idx / 12
		w := idx % 12
		switch w {
		case 0:
		case 1:
			f.wq[layer] = f.eng.UploadQ4_0(q4Data, f.dim, f.dim)
			f.wqS[layer] = nil
		case 2:
			f.wk[layer] = f.eng.UploadQ4_0(q4Data, f.kvDim, f.dim)
			f.wkS[layer] = nil
		case 3:
			f.wv[layer] = f.eng.UploadQ4_0(q4Data, f.kvDim, f.dim)
			f.wvS[layer] = nil
		case 4, 5, 6:
		case 7:
			f.wo[layer] = f.eng.UploadQ4_0(q4Data, f.dim, f.dim)
			f.woS[layer] = nil
		case 8:
		case 9:
			f.wgate[layer] = f.eng.UploadQ4_0(q4Data, f.ffnDim, f.dim)
			f.wgateS[layer] = nil
		case 10:
			f.wup[layer] = f.eng.UploadQ4_0(q4Data, f.ffnDim, f.dim)
			f.wupS[layer] = nil
		case 11:
			f.wdown[layer] = f.eng.UploadQ4_0(q4Data, f.dim, f.ffnDim)
			f.wdownS[layer] = nil
		}
	} else if idx == nL*12 {
		// finalNorm — not Q4
	} else if idx == nL*12+1 {
		f.lmHead = f.eng.UploadQ4_0(q4Data, f.vocabSize, f.dim)
		f.lmHeadS = nil
		if f.lmHead == nil || f.lmHead.DevicePtr() == nil {
			log.Printf("[VRAM] ERROR: lm_head upload FAILED (nil tensor or nil DevicePtr) — likely OOM. q4Data=%d bytes, vocab=%d, dim=%d",
				len(q4Data), f.vocabSize, f.dim)
		} else {
			log.Printf("[VRAM] lm_head uploaded: %d bytes, DevicePtr=%v", len(q4Data), f.lmHead.DevicePtr() != nil)
		}
		f.BuildFusedDispatch()
	}
}

func (f *CUDAFusedInference) PartialStepQ4(hiddenIn []float32, pos, layerStart, layerEnd int, hiddenOut, logitsOut []float32) int {
	if !f.built {
		return -1
	}

	f.eng.UploadInto(f.hidden[f.hiddenIdx], hiddenIn)

	s := &f.scratch[0]
	hP := f.hidden[f.hiddenIdx].DevicePtr()
	nP := s.normed.DevicePtr()

	if f.fusedReady {
		profile := 0
		if pos == 1 { profile = 1 }
		KPartialStepQ4Fused(
			unsafe.Pointer(&f.norm1Ptrs[0]), unsafe.Pointer(&f.wqPtrs[0]),
			unsafe.Pointer(&f.wkPtrs[0]), unsafe.Pointer(&f.wvPtrs[0]), unsafe.Pointer(&f.woPtrs[0]),
			unsafe.Pointer(&f.norm2Ptrs[0]), unsafe.Pointer(&f.wgatePtrs[0]),
			unsafe.Pointer(&f.wupPtrs[0]), unsafe.Pointer(&f.wdownPtrs[0]),
			unsafe.Pointer(&f.kCachePtrs[0]), unsafe.Pointer(&f.vCachePtrs[0]),
			hP, nP, s.Q.DevicePtr(), s.K.DevicePtr(), s.V.DevicePtr(),
			s.attnOut.DevicePtr(), s.proj.DevicePtr(), s.normed2.DevicePtr(),
			s.gatePre.DevicePtr(), s.upOut.DevicePtr(), s.ffnMid.DevicePtr(),
			s.q8Scratch,
			f.cosTab.DevicePtr(), f.sinTab.DevicePtr(),
			f.dim, f.kvDim, f.headDim, f.ffnDim, f.nHeads, f.nKVHeads, f.halfHead,
			pos, layerStart, layerEnd, profile)

		if layerEnd >= f.nLayers && logitsOut != nil && f.finalNorm != nil {
			KRMSNormOut(hP, nP, f.finalNorm.DevicePtr(), 1, f.dim)
			f.q4Matvec(nP, f.lmHead, f.lmHeadS, f.logits.DevicePtr(), f.vocabSize, f.dim)
			KSync()
			copy(logitsOut, f.eng.ToHost(f.logits))
		}
		if hiddenOut != nil && layerEnd < f.nLayers {
			KSync()
			copy(hiddenOut, f.eng.ToHost(f.hidden[f.hiddenIdx]))
		}
		return 0
	}

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

	profile := pos == 1
	var tNorm, tMatvec, tRope, tKV, tAttn, tSilu, tAdd time.Duration
	var tOp time.Time

	for l := layerStart; l < layerEnd && l < f.nLayers; l++ {
		if f.norm1[l] == nil {
			continue
		}

		if profile { KSync(); tOp = time.Now() }
		KRMSNormOut(hP, nP, f.norm1[l].DevicePtr(), 1, f.dim)
		if profile { KSync(); tNorm += time.Since(tOp) }

		if profile { tOp = time.Now() }
		useQ8Cache := s.q8Scratch != nil && HasQ8Quantize() && f.wqS[l] == nil
		if useQ8Cache {
			if !q8CacheLogged {
				log.Printf("[Q4] using Q8 activation caching for QKV+gate/up")
				q8CacheLogged = true
			}
			f.q8QuantizeAct(nP, f.dim)
			f.q4MatvecPreq(f.wq[l].DevicePtr(), qP, f.dim, f.dim)
			f.q4MatvecPreq(f.wk[l].DevicePtr(), kP, f.kvDim, f.dim)
			f.q4MatvecPreq(f.wv[l].DevicePtr(), vP, f.kvDim, f.dim)
		} else {
			f.q4Matvec(nP, f.wq[l], f.wqS[l], qP, f.dim, f.dim)
			f.q4Matvec(nP, f.wk[l], f.wkS[l], kP, f.kvDim, f.dim)
			f.q4Matvec(nP, f.wv[l], f.wvS[l], vP, f.kvDim, f.dim)
		}
		if profile { KSync(); tMatvec += time.Since(tOp) }

		cosSlice := unsafe.Add(f.cosTab.DevicePtr(), uintptr(cosOff*4))
		sinSlice := unsafe.Add(f.sinTab.DevicePtr(), uintptr(sinOff*4))
		if profile { tOp = time.Now() }
		KRoPE(qP, cosSlice, sinSlice, 1, f.dim, f.headDim, f.nHeads)
		KRoPE(kP, cosSlice, sinSlice, 1, f.kvDim, f.headDim, f.nKVHeads)
		if profile { KSync(); tRope += time.Since(tOp) }

		if profile { tOp = time.Now() }
		KKVCacheWrite(f.kCache[l].DevicePtr(), kP, pos, f.kvDim)
		KKVCacheWrite(f.vCache[l].DevicePtr(), vP, pos, f.kvDim)
		if profile { KSync(); tKV += time.Since(tOp) }

		if profile { tOp = time.Now() }
		KDecodeAttention(qP, f.kCache[l].DevicePtr(), f.vCache[l].DevicePtr(), aP,
			seqLen, f.dim, f.kvDim, f.nHeads, f.nKVHeads)
		if profile { KSync(); tAttn += time.Since(tOp) }

		if profile { tOp = time.Now() }
		f.q4Matvec(aP, f.wo[l], f.woS[l], pP, f.dim, f.dim)
		if profile { KSync(); tMatvec += time.Since(tOp) }

		if profile { tOp = time.Now() }
		KAddInPlace(hP, pP, f.dim)
		if profile { KSync(); tAdd += time.Since(tOp) }

		if profile { tOp = time.Now() }
		KRMSNormOut(hP, n2P, f.norm2[l].DevicePtr(), 1, f.dim)
		if profile { KSync(); tNorm += time.Since(tOp) }

		if profile { tOp = time.Now() }
		if useQ8Cache {
			f.q8QuantizeAct(n2P, f.dim)
			f.q4MatvecPreq(f.wgate[l].DevicePtr(), gP, f.ffnDim, f.dim)
			f.q4MatvecPreq(f.wup[l].DevicePtr(), uP, f.ffnDim, f.dim)
		} else {
			f.q4Matvec(n2P, f.wgate[l], f.wgateS[l], gP, f.ffnDim, f.dim)
			f.q4Matvec(n2P, f.wup[l], f.wupS[l], uP, f.ffnDim, f.dim)
		}
		if profile { KSync(); tMatvec += time.Since(tOp) }

		if profile { tOp = time.Now() }
		KSiLUGateMul(gP, uP, fP, f.ffnDim)
		if profile { KSync(); tSilu += time.Since(tOp) }

		if profile { tOp = time.Now() }
		f.q4Matvec(fP, f.wdown[l], f.wdownS[l], pP, f.dim, f.ffnDim)
		if profile { KSync(); tMatvec += time.Since(tOp) }

		if profile { tOp = time.Now() }
		KAddInPlace(hP, pP, f.dim)
		if profile { KSync(); tAdd += time.Since(tOp) }
	}

	if profile {
		log.Printf("[PROFILE] %d layers: matvec=%.1fms norm=%.1fms rope=%.1fms kv=%.1fms attn=%.1fms silu=%.1fms add=%.1fms total=%.1fms",
			layerEnd-layerStart,
			float64(tMatvec.Microseconds())/1000,
			float64(tNorm.Microseconds())/1000,
			float64(tRope.Microseconds())/1000,
			float64(tKV.Microseconds())/1000,
			float64(tAttn.Microseconds())/1000,
			float64(tSilu.Microseconds())/1000,
			float64(tAdd.Microseconds())/1000,
			float64((tMatvec+tNorm+tRope+tKV+tAttn+tSilu+tAdd).Microseconds())/1000)
	}

	if layerEnd >= f.nLayers && logitsOut != nil && f.finalNorm != nil {
		KRMSNormOut(hP, nP, f.finalNorm.DevicePtr(), 1, f.dim)
		f.q4Matvec(nP, f.lmHead, f.lmHeadS, f.logits.DevicePtr(), f.vocabSize, f.dim)
		KSync()
		copy(logitsOut, f.eng.ToHost(f.logits))
	}

	if hiddenOut != nil && layerEnd < f.nLayers {
		KSync()
		copy(hiddenOut, f.eng.ToHost(f.hidden[f.hiddenIdx]))
	}

	if pos == 0 {
		KSync()
		h := f.eng.ToHost(f.hidden[f.hiddenIdx])
		var sum float32
		for _, v := range h { sum += v }
		log.Printf("CUDA-RESIDENT layers=%d->%d hidden[0:4]=[%.6f,%.6f,%.6f,%.6f] sum=%.3f",
			layerStart, layerEnd, h[0], h[1], h[2], h[3], sum)
	}

	return 0
}

// PartialStepQ4Continue runs layers without uploading hiddenIn — assumes
// f.hidden[f.hiddenIdx] already holds the correct GPU-resident state from a prior
// PartialStepQ4 or PartialStepQ4Continue call. This avoids a CPU→GPU
// round-trip between pipeline stages on the same device.
func (f *CUDAFusedInference) PartialStepQ4Continue(pos, layerStart, layerEnd int, hiddenOut, logitsOut []float32) int {
	if !f.built {
		return -1
	}

	s := &f.scratch[0]
	hP := f.hidden[f.hiddenIdx].DevicePtr()
	nP := s.normed.DevicePtr()

	if f.fusedReady {
		KPartialStepQ4Fused(
			unsafe.Pointer(&f.norm1Ptrs[0]), unsafe.Pointer(&f.wqPtrs[0]),
			unsafe.Pointer(&f.wkPtrs[0]), unsafe.Pointer(&f.wvPtrs[0]), unsafe.Pointer(&f.woPtrs[0]),
			unsafe.Pointer(&f.norm2Ptrs[0]), unsafe.Pointer(&f.wgatePtrs[0]),
			unsafe.Pointer(&f.wupPtrs[0]), unsafe.Pointer(&f.wdownPtrs[0]),
			unsafe.Pointer(&f.kCachePtrs[0]), unsafe.Pointer(&f.vCachePtrs[0]),
			hP, nP, s.Q.DevicePtr(), s.K.DevicePtr(), s.V.DevicePtr(),
			s.attnOut.DevicePtr(), s.proj.DevicePtr(), s.normed2.DevicePtr(),
			s.gatePre.DevicePtr(), s.upOut.DevicePtr(), s.ffnMid.DevicePtr(),
			s.q8Scratch,
			f.cosTab.DevicePtr(), f.sinTab.DevicePtr(),
			f.dim, f.kvDim, f.headDim, f.ffnDim, f.nHeads, f.nKVHeads, f.halfHead,
			pos, layerStart, layerEnd, 0)

		if layerEnd >= f.nLayers && logitsOut != nil && f.finalNorm != nil {
			KRMSNormOut(hP, nP, f.finalNorm.DevicePtr(), 1, f.dim)
			f.q4Matvec(nP, f.lmHead, f.lmHeadS, f.logits.DevicePtr(), f.vocabSize, f.dim)
			KSync()
			copy(logitsOut, f.eng.ToHost(f.logits))
		}
		if hiddenOut != nil && layerEnd < f.nLayers {
			KSync()
			copy(hiddenOut, f.eng.ToHost(f.hidden[f.hiddenIdx]))
		}
		return 0
	}

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
		if f.norm1[l] == nil {
			continue
		}

		KRMSNormOut(hP, nP, f.norm1[l].DevicePtr(), 1, f.dim)

		useQ8Cache := s.q8Scratch != nil && HasQ8Quantize() && f.wqS[l] == nil
		if useQ8Cache {
			f.q8QuantizeAct(nP, f.dim)
			f.q4MatvecPreq(f.wq[l].DevicePtr(), qP, f.dim, f.dim)
			f.q4MatvecPreq(f.wk[l].DevicePtr(), kP, f.kvDim, f.dim)
			f.q4MatvecPreq(f.wv[l].DevicePtr(), vP, f.kvDim, f.dim)
		} else {
			f.q4Matvec(nP, f.wq[l], f.wqS[l], qP, f.dim, f.dim)
			f.q4Matvec(nP, f.wk[l], f.wkS[l], kP, f.kvDim, f.dim)
			f.q4Matvec(nP, f.wv[l], f.wvS[l], vP, f.kvDim, f.dim)
		}

		cosSlice := unsafe.Add(f.cosTab.DevicePtr(), uintptr(cosOff*4))
		sinSlice := unsafe.Add(f.sinTab.DevicePtr(), uintptr(sinOff*4))
		KRoPE(qP, cosSlice, sinSlice, 1, f.dim, f.headDim, f.nHeads)
		KRoPE(kP, cosSlice, sinSlice, 1, f.kvDim, f.headDim, f.nKVHeads)

		KKVCacheWrite(f.kCache[l].DevicePtr(), kP, pos, f.kvDim)
		KKVCacheWrite(f.vCache[l].DevicePtr(), vP, pos, f.kvDim)

		KDecodeAttention(qP, f.kCache[l].DevicePtr(), f.vCache[l].DevicePtr(), aP,
			seqLen, f.dim, f.kvDim, f.nHeads, f.nKVHeads)

		f.q4Matvec(aP, f.wo[l], f.woS[l], pP, f.dim, f.dim)
		KAddInPlace(hP, pP, f.dim)

		KRMSNormOut(hP, n2P, f.norm2[l].DevicePtr(), 1, f.dim)

		if useQ8Cache {
			f.q8QuantizeAct(n2P, f.dim)
			f.q4MatvecPreq(f.wgate[l].DevicePtr(), gP, f.ffnDim, f.dim)
			f.q4MatvecPreq(f.wup[l].DevicePtr(), uP, f.ffnDim, f.dim)
		} else {
			f.q4Matvec(n2P, f.wgate[l], f.wgateS[l], gP, f.ffnDim, f.dim)
			f.q4Matvec(n2P, f.wup[l], f.wupS[l], uP, f.ffnDim, f.dim)
		}

		KSiLUGateMul(gP, uP, fP, f.ffnDim)
		f.q4Matvec(fP, f.wdown[l], f.wdownS[l], pP, f.dim, f.ffnDim)
		KAddInPlace(hP, pP, f.dim)
	}

	if layerEnd >= f.nLayers && logitsOut != nil && f.finalNorm != nil {
		KRMSNormOut(hP, nP, f.finalNorm.DevicePtr(), 1, f.dim)
		f.q4Matvec(nP, f.lmHead, f.lmHeadS, f.logits.DevicePtr(), f.vocabSize, f.dim)
		KSync()
		copy(logitsOut, f.eng.ToHost(f.logits))
	}

	if hiddenOut != nil && layerEnd < f.nLayers {
		KSync()
		copy(hiddenOut, f.eng.ToHost(f.hidden[f.hiddenIdx]))
	}

	return 0
}

// PartialStepQ4ContinueOnStream1 runs layers on computeStream1 (Stage B).
// Waits for stageDone event (recorded after Stage A on stream 0), then
// dispatches all layer kernels on stream1. No CPU→GPU upload — hidden state
// is already GPU-resident from Stage A.
func (f *CUDAFusedInference) PartialStepQ4ContinueOnStream1(pos, layerStart, layerEnd int, hiddenOut, logitsOut []float32) int {
	if !f.built {
		return -1
	}

	s := &f.scratch[1]
	if s.Q == nil {
		log.Printf("[VNODE] ERROR: scratch[1] not allocated! Falling back to scratch[0]")
		s = &f.scratch[0]
	}
	hP := f.hidden[f.hiddenIdx].DevicePtr()
	nP := s.normed.DevicePtr()

	if f.fusedReady {
		nilWeights := 0
		for l := layerStart; l < layerEnd && l < len(f.norm1Ptrs); l++ {
			if f.norm1Ptrs[l] == nil { nilWeights++ }
		}
		if nilWeights > 0 {
			log.Printf("[VNODE] WARNING: %d/%d nil weight pointers in Stage B range [%d,%d)",
				nilWeights, layerEnd-layerStart, layerStart, layerEnd)
		}
		KPartialStepQ4FusedOnStream(
			unsafe.Pointer(&f.norm1Ptrs[0]), unsafe.Pointer(&f.wqPtrs[0]),
			unsafe.Pointer(&f.wkPtrs[0]), unsafe.Pointer(&f.wvPtrs[0]), unsafe.Pointer(&f.woPtrs[0]),
			unsafe.Pointer(&f.norm2Ptrs[0]), unsafe.Pointer(&f.wgatePtrs[0]),
			unsafe.Pointer(&f.wupPtrs[0]), unsafe.Pointer(&f.wdownPtrs[0]),
			unsafe.Pointer(&f.kCachePtrs[0]), unsafe.Pointer(&f.vCachePtrs[0]),
			hP, nP, s.Q.DevicePtr(), s.K.DevicePtr(), s.V.DevicePtr(),
			s.attnOut.DevicePtr(), s.proj.DevicePtr(), s.normed2.DevicePtr(),
			s.gatePre.DevicePtr(), s.upOut.DevicePtr(), s.ffnMid.DevicePtr(),
			s.q8Scratch,
			f.cosTab.DevicePtr(), f.sinTab.DevicePtr(),
			f.dim, f.kvDim, f.headDim, f.ffnDim, f.nHeads, f.nKVHeads, f.halfHead,
			pos, layerStart, layerEnd, 0,
			f.ComputeStream1Ptr())

		if layerEnd >= f.nLayers && logitsOut != nil && f.finalNorm != nil {
			KRMSNormOut(hP, nP, f.finalNorm.DevicePtr(), 1, f.dim)
			f.q4Matvec(nP, f.lmHead, f.lmHeadS, f.logits.DevicePtr(), f.vocabSize, f.dim)
			f.SyncStream1()
			KSync()
			copy(logitsOut, f.eng.ToHost(f.logits))
		}
		if hiddenOut != nil && layerEnd < f.nLayers {
			f.SyncStream1()
			copy(hiddenOut, f.eng.ToHost(f.hidden[f.hiddenIdx]))
		}
		return 0
	}

	// Fallback: no fused dispatch → use default stream (sequential)
	return f.PartialStepQ4Continue(pos, layerStart, layerEnd, hiddenOut, logitsOut)
}

// StepLayerResident runs a single transformer layer using individual kernel
// dispatch against GPU-resident Q4 weights (f.wq[layer] etc.). Snapshots
// hidden state before and after the layer for phase measurement.
// For layer 0, uploads hiddenIn to GPU first. For the last layer, produces
// logits in logitsOut (if non-nil).
func (f *CUDAFusedInference) StepLayerResident(layer, pos int, hiddenIn, preOut, postOut, logitsOut []float32) {
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

	verbose := pos == 0 && layer < 3

	if layer == 0 {
		f.eng.UploadInto(f.hidden[f.hiddenIdx], hiddenIn)
	}

	if verbose {
		KSync()
		h := f.eng.ToHost(f.hidden[f.hiddenIdx])
		var sum float32
		for _, v := range h { if v > 0 { sum += v } else { sum -= v } }
		log.Printf("[SLR L%d pos%d] entry hiddenIdx=%d hP=%v sum=%.6f h[0:4]=[%.6f,%.6f,%.6f,%.6f]",
			layer, pos, f.hiddenIdx, hP != nil, sum, h[0], h[1], h[2], h[3])
	}

	// Snapshot pre-layer hidden state
	if preOut != nil {
		KSync()
		copy(preOut, f.eng.ToHost(f.hidden[f.hiddenIdx]))
	}

	// Check weights exist
	if f.norm1[layer] == nil {
		log.Printf("[SLR L%d] ERROR: norm1 is nil", layer)
		return
	}
	if f.wq[layer] == nil {
		log.Printf("[SLR L%d] ERROR: wq is nil", layer)
		return
	}

	seqLen := pos + 1
	cosOff := pos * f.halfHead
	cosSlice := unsafe.Add(f.cosTab.DevicePtr(), uintptr(cosOff*4))
	sinSlice := unsafe.Add(f.sinTab.DevicePtr(), uintptr(cosOff*4))

	KRMSNormOut(hP, nP, f.norm1[layer].DevicePtr(), 1, f.dim)

	if verbose {
		KSync()
		n := f.eng.ToHost(s.normed)
		var sum float32
		for _, v := range n { if v > 0 { sum += v } else { sum -= v } }
		log.Printf("[SLR L%d pos%d] after RMSNorm normed sum=%.6f n[0:4]=[%.6f,%.6f,%.6f,%.6f]",
			layer, pos, sum, n[0], n[1], n[2], n[3])
	}

	q8 := s.q8Scratch
	if q8 != nil && HasQ8Quantize() && f.wqS[layer] == nil {
		KQ8QuantizeAct(nP, q8, f.dim)
		KQ4_0MatvecDP4APreq(f.wq[layer].DevicePtr(), q8, qP, f.dim, f.dim)
		KQ4_0MatvecDP4APreq(f.wk[layer].DevicePtr(), q8, kP, f.kvDim, f.dim)
		KQ4_0MatvecDP4APreq(f.wv[layer].DevicePtr(), q8, vP, f.kvDim, f.dim)
	} else {
		f.q4Matvec(nP, f.wq[layer], f.wqS[layer], qP, f.dim, f.dim)
		f.q4Matvec(nP, f.wk[layer], f.wkS[layer], kP, f.kvDim, f.dim)
		f.q4Matvec(nP, f.wv[layer], f.wvS[layer], vP, f.kvDim, f.dim)
	}

	if verbose {
		KSync()
		q := f.eng.ToHost(s.Q)
		var qsum float32
		for _, v := range q { if v > 0 { qsum += v } else { qsum -= v } }
		log.Printf("[SLR L%d pos%d] after QKV matvec Q sum=%.6f Q[0:4]=[%.6f,%.6f,%.6f,%.6f]",
			layer, pos, qsum, q[0], q[1], q[2], q[3])
	}

	KRoPE(qP, cosSlice, sinSlice, 1, f.dim, f.headDim, f.nHeads)
	KRoPE(kP, cosSlice, sinSlice, 1, f.kvDim, f.headDim, f.nKVHeads)

	KKVCacheWrite(f.kCache[layer].DevicePtr(), kP, pos, f.kvDim)
	KKVCacheWrite(f.vCache[layer].DevicePtr(), vP, pos, f.kvDim)
	KDecodeAttention(qP, f.kCache[layer].DevicePtr(), f.vCache[layer].DevicePtr(), aP,
		seqLen, f.dim, f.kvDim, f.nHeads, f.nKVHeads)

	if verbose {
		KSync()
		a := f.eng.ToHost(s.attnOut)
		var asum float32
		for _, v := range a { if v > 0 { asum += v } else { asum -= v } }
		log.Printf("[SLR L%d pos%d] after attn sum=%.6f a[0:4]=[%.6f,%.6f,%.6f,%.6f]",
			layer, pos, asum, a[0], a[1], a[2], a[3])
	}

	if q8 != nil && HasQ4_0DP4A() && f.woS[layer] == nil {
		KQ4_0MatvecDP4A(aP, f.wo[layer].DevicePtr(), pP, q8, f.dim, f.dim)
	} else {
		f.q4Matvec(aP, f.wo[layer], f.woS[layer], pP, f.dim, f.dim)
	}
	KAddInPlace(hP, pP, f.dim)

	if verbose {
		KSync()
		h := f.eng.ToHost(f.hidden[f.hiddenIdx])
		var sum float32
		for _, v := range h { if v > 0 { sum += v } else { sum -= v } }
		log.Printf("[SLR L%d pos%d] after attn residual hidden sum=%.6f h[0:4]=[%.6f,%.6f,%.6f,%.6f]",
			layer, pos, sum, h[0], h[1], h[2], h[3])
	}

	KRMSNormOut(hP, n2P, f.norm2[layer].DevicePtr(), 1, f.dim)
	if q8 != nil && HasQ8Quantize() && f.wgateS[layer] == nil {
		KQ8QuantizeAct(n2P, q8, f.dim)
		KQ4_0MatvecDP4APreq(f.wgate[layer].DevicePtr(), q8, gP, f.ffnDim, f.dim)
		KQ4_0MatvecDP4APreq(f.wup[layer].DevicePtr(), q8, uP, f.ffnDim, f.dim)
	} else {
		f.q4Matvec(n2P, f.wgate[layer], f.wgateS[layer], gP, f.ffnDim, f.dim)
		f.q4Matvec(n2P, f.wup[layer], f.wupS[layer], uP, f.ffnDim, f.dim)
	}
	KSiLUGateMul(gP, uP, fP, f.ffnDim)
	if q8 != nil && HasQ4_0DP4A() && f.wdownS[layer] == nil {
		KQ4_0MatvecDP4A(fP, f.wdown[layer].DevicePtr(), pP, q8, f.dim, f.ffnDim)
	} else {
		f.q4Matvec(fP, f.wdown[layer], f.wdownS[layer], pP, f.dim, f.ffnDim)
	}
	KAddInPlace(hP, pP, f.dim)

	if verbose {
		KSync()
		h := f.eng.ToHost(f.hidden[f.hiddenIdx])
		var sum float32
		for _, v := range h { if v > 0 { sum += v } else { sum -= v } }
		log.Printf("[SLR L%d pos%d] after FFN residual hidden sum=%.6f h[0:4]=[%.6f,%.6f,%.6f,%.6f]",
			layer, pos, sum, h[0], h[1], h[2], h[3])
	}

	// Snapshot post-layer hidden state
	if postOut != nil {
		KSync()
		copy(postOut, f.eng.ToHost(f.hidden[f.hiddenIdx]))
	}

	// Final norm + lm_head for last layer
	if logitsOut != nil {
		KRMSNormOut(hP, nP, f.finalNorm.DevicePtr(), 1, f.dim)
		f.q4Matvec(nP, f.lmHead, f.lmHeadS, f.logits.DevicePtr(), f.vocabSize, f.dim)
		KSync()
		copy(logitsOut, f.eng.ToHost(f.logits))
	}
}

var q4MatvecLogged bool
var q4MatvecCompared bool
var q8CacheLogged bool

func (f *CUDAFusedInference) q8QuantizeAct(actPtr unsafe.Pointer, K int) {
	q8 := f.scratch[0].q8Scratch
	if q8 != nil && HasQ8Quantize() {
		KQ8QuantizeAct(actPtr, q8, K)
	}
}

func (f *CUDAFusedInference) q4MatvecPreq(weightPtr, outPtr unsafe.Pointer, N, K int) {
	KQ4_0MatvecDP4APreq(weightPtr, f.scratch[0].q8Scratch, outPtr, N, K)
}

func (f *CUDAFusedInference) q4Matvec(actPtr unsafe.Pointer, weight, scales *Tensor, outPtr unsafe.Pointer, N, K int) {
	q8 := f.scratch[0].q8Scratch
	if scales == nil {
		if q8 != nil && HasQ4_0DP4A() {
			if !q4MatvecLogged {
				log.Printf("[Q4] using dp4a kernel (Q4×Q8 integer dot product)")
				q4MatvecLogged = true
			}
			KQ4_0MatvecDP4A(actPtr, weight.DevicePtr(), outPtr, q8, N, K)

			if !q4MatvecCompared {
				KSync()
				dp4aOut := make([]float32, N)
				C.tw_gpu_download(unsafe.Pointer(&dp4aOut[0]), outPtr, C.size_t(N*4))
				hasNonZero := false
				for _, v := range dp4aOut[:min(N, 100)] {
					if v != 0 { hasNonZero = true; break }
				}
				if hasNonZero {
					q4MatvecCompared = true
					refBuf := f.eng.Zeros([]int{N})
					KQ4_0Matvec(actPtr, weight.DevicePtr(), refBuf.DevicePtr(), N, K)
					KSync()
					scalarOut := f.eng.ToHost(refBuf)
					maxDiff := float32(0)
					maxRel := float32(0)
					for i := 0; i < N; i++ {
						d := dp4aOut[i] - scalarOut[i]
						if d < 0 { d = -d }
						if d > maxDiff { maxDiff = d }
						denom := scalarOut[i]
						if denom < 0 { denom = -denom }
						if denom > 0.001 {
							rel := d / denom
							if rel > maxRel { maxRel = rel }
						}
					}
					log.Printf("[DP4A vs SCALAR] N=%d K=%d maxDiff=%.6f maxRel=%.4f dp4a[0:4]=%v scalar[0:4]=%v",
						N, K, maxDiff, maxRel, dp4aOut[:4], scalarOut[:4])
				}
			}
		} else {
			if !q4MatvecLogged {
				log.Printf("[Q4] using scalar kernel (dp4a not available)")
				q4MatvecLogged = true
			}
			KQ4_0Matvec(actPtr, weight.DevicePtr(), outPtr, N, K)
		}
	} else {
		KQ8Matvec(actPtr, weight.DevicePtr(), scales.DevicePtr(), outPtr, N, K)
	}
}
