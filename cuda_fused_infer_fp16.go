//go:build linux && cgo

package mongoose

import (
	"log"
	"unsafe"
)

// fp16Matvec computes out[N] = act_fp32[K] @ weight_fp16[N*K] using cuBLAS hgemm.
// Converts activation FP32→FP16 on GPU, then calls hgemm with M=1.
func (f *CUDAFusedInference) fp16Matvec(actPtr unsafe.Pointer, weightFP16 *Tensor, outPtr unsafe.Pointer, N, K int) {
	if f.fp16Act == nil {
		maxK := f.dim
		if f.ffnDim > maxK {
			maxK = f.ffnDim
		}
		f.fp16Act = f.eng.Zeros([]int{(maxK + 1) / 2})
		f.fp16Act.Size = maxK
		log.Printf("[FP16] allocated activation scratch: %d FP16 elements", maxK)
	}
	KFP32ToFP16(actPtr, f.fp16Act.DevicePtr(), K)
	f.eng.HGemmTransBRaw(f.fp16Act.DevicePtr(), weightFP16.DevicePtr(), outPtr, 1, K, N)
}

// StepLayerResidentFP16 runs a single transformer layer using individual kernel
// dispatch against GPU-resident FP16 weights. All matvecs use cuBLAS hgemm.
// Activations remain FP32 — only weight storage is FP16.
// Same structure as StepLayerResident (Q4) but dedicated FP16 dispatch.
func (f *CUDAFusedInference) StepLayerResidentFP16(layer, pos int, hiddenIn, preOut, postOut, logitsOut []float32) {
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
		for _, v := range h {
			if v > 0 {
				sum += v
			} else {
				sum -= v
			}
		}
		log.Printf("[SLR-FP16 L%d pos%d] entry hiddenIdx=%d hP=%v sum=%.6f h[0:4]=[%.6f,%.6f,%.6f,%.6f]",
			layer, pos, f.hiddenIdx, hP != nil, sum, h[0], h[1], h[2], h[3])
	}

	if preOut != nil {
		KSync()
		copy(preOut, f.eng.ToHost(f.hidden[f.hiddenIdx]))
	}

	if f.norm1[layer] == nil {
		log.Printf("[SLR-FP16 L%d] ERROR: norm1 is nil", layer)
		return
	}
	if f.wq[layer] == nil {
		log.Printf("[SLR-FP16 L%d] ERROR: wq is nil", layer)
		return
	}

	seqLen := pos + 1
	cosOff := pos * f.halfHead
	cosSlice := unsafe.Add(f.cosTab.DevicePtr(), uintptr(cosOff*4))
	sinSlice := unsafe.Add(f.sinTab.DevicePtr(), uintptr(cosOff*4))

	// Attention block
	KRMSNormOut(hP, nP, f.norm1[layer].DevicePtr(), 1, f.dim)

	if verbose {
		KSync()
		n := f.eng.ToHost(s.normed)
		var sum float32
		for _, v := range n {
			if v > 0 {
				sum += v
			} else {
				sum -= v
			}
		}
		log.Printf("[SLR-FP16 L%d pos%d] after RMSNorm normed sum=%.6f n[0:4]=[%.6f,%.6f,%.6f,%.6f]",
			layer, pos, sum, n[0], n[1], n[2], n[3])
	}

	f.fp16Matvec(nP, f.wq[layer], qP, f.dim, f.dim)
	f.fp16Matvec(nP, f.wk[layer], kP, f.kvDim, f.dim)
	f.fp16Matvec(nP, f.wv[layer], vP, f.kvDim, f.dim)

	if verbose {
		KSync()
		q := f.eng.ToHost(s.Q)
		var qsum float32
		for _, v := range q {
			if v > 0 {
				qsum += v
			} else {
				qsum -= v
			}
		}
		log.Printf("[SLR-FP16 L%d pos%d] after QKV matvec Q sum=%.6f Q[0:4]=[%.6f,%.6f,%.6f,%.6f]",
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
		for _, v := range a {
			if v > 0 {
				asum += v
			} else {
				asum -= v
			}
		}
		log.Printf("[SLR-FP16 L%d pos%d] after attn sum=%.6f a[0:4]=[%.6f,%.6f,%.6f,%.6f]",
			layer, pos, asum, a[0], a[1], a[2], a[3])
	}

	f.fp16Matvec(aP, f.wo[layer], pP, f.dim, f.dim)
	KAddInPlace(hP, pP, f.dim)

	if verbose {
		KSync()
		h := f.eng.ToHost(f.hidden[f.hiddenIdx])
		var sum float32
		for _, v := range h {
			if v > 0 {
				sum += v
			} else {
				sum -= v
			}
		}
		log.Printf("[SLR-FP16 L%d pos%d] after attn residual hidden sum=%.6f h[0:4]=[%.6f,%.6f,%.6f,%.6f]",
			layer, pos, sum, h[0], h[1], h[2], h[3])
	}

	// FFN block
	KRMSNormOut(hP, n2P, f.norm2[layer].DevicePtr(), 1, f.dim)

	f.fp16Matvec(n2P, f.wgate[layer], gP, f.ffnDim, f.dim)
	f.fp16Matvec(n2P, f.wup[layer], uP, f.ffnDim, f.dim)
	KSiLUGateMul(gP, uP, fP, f.ffnDim)
	f.fp16Matvec(fP, f.wdown[layer], pP, f.dim, f.ffnDim)
	KAddInPlace(hP, pP, f.dim)

	if verbose {
		KSync()
		h := f.eng.ToHost(f.hidden[f.hiddenIdx])
		var sum float32
		for _, v := range h {
			if v > 0 {
				sum += v
			} else {
				sum -= v
			}
		}
		log.Printf("[SLR-FP16 L%d pos%d] after FFN residual hidden sum=%.6f h[0:4]=[%.6f,%.6f,%.6f,%.6f]",
			layer, pos, sum, h[0], h[1], h[2], h[3])
	}

	if postOut != nil {
		KSync()
		copy(postOut, f.eng.ToHost(f.hidden[f.hiddenIdx]))
	}

	if logitsOut != nil {
		KRMSNormOut(hP, nP, f.finalNorm.DevicePtr(), 1, f.dim)
		f.fp16Matvec(nP, f.lmHead, f.logits.DevicePtr(), f.vocabSize, f.dim)
		KSync()
		copy(logitsOut, f.eng.ToHost(f.logits))
	}
}

// PartialStepFP16 runs multiple transformer layers using FP16 weight matvecs.
// Equivalent of PartialStepQ4 but with dedicated FP16 dispatch — no Q4 logic.
func (f *CUDAFusedInference) PartialStepFP16(hiddenIn []float32, pos, layerStart, layerEnd int, hiddenOut, logitsOut []float32) int {
	if !f.built {
		return -1
	}

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
	seqLen := pos + 1

	for l := layerStart; l < layerEnd && l < f.nLayers; l++ {
		if f.norm1[l] == nil {
			continue
		}

		KRMSNormOut(hP, nP, f.norm1[l].DevicePtr(), 1, f.dim)

		f.fp16Matvec(nP, f.wq[l], qP, f.dim, f.dim)
		f.fp16Matvec(nP, f.wk[l], kP, f.kvDim, f.dim)
		f.fp16Matvec(nP, f.wv[l], vP, f.kvDim, f.dim)

		cosSlice := unsafe.Add(f.cosTab.DevicePtr(), uintptr(cosOff*4))
		sinSlice := unsafe.Add(f.sinTab.DevicePtr(), uintptr(cosOff*4))
		KRoPE(qP, cosSlice, sinSlice, 1, f.dim, f.headDim, f.nHeads)
		KRoPE(kP, cosSlice, sinSlice, 1, f.kvDim, f.headDim, f.nKVHeads)

		KKVCacheWrite(f.kCache[l].DevicePtr(), kP, pos, f.kvDim)
		KKVCacheWrite(f.vCache[l].DevicePtr(), vP, pos, f.kvDim)

		KDecodeAttention(qP, f.kCache[l].DevicePtr(), f.vCache[l].DevicePtr(), aP,
			seqLen, f.dim, f.kvDim, f.nHeads, f.nKVHeads)

		f.fp16Matvec(aP, f.wo[l], pP, f.dim, f.dim)
		KAddInPlace(hP, pP, f.dim)

		KRMSNormOut(hP, n2P, f.norm2[l].DevicePtr(), 1, f.dim)

		f.fp16Matvec(n2P, f.wgate[l], gP, f.ffnDim, f.dim)
		f.fp16Matvec(n2P, f.wup[l], uP, f.ffnDim, f.dim)
		KSiLUGateMul(gP, uP, fP, f.ffnDim)
		f.fp16Matvec(fP, f.wdown[l], pP, f.dim, f.ffnDim)
		KAddInPlace(hP, pP, f.dim)
	}

	if layerEnd >= f.nLayers && logitsOut != nil && f.finalNorm != nil {
		KRMSNormOut(hP, nP, f.finalNorm.DevicePtr(), 1, f.dim)
		f.fp16Matvec(nP, f.lmHead, f.logits.DevicePtr(), f.vocabSize, f.dim)
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
		for _, v := range h {
			sum += v
		}
		log.Printf("CUDA-FP16 layers=%d->%d hidden[0:4]=[%.6f,%.6f,%.6f,%.6f] sum=%.3f",
			layerStart, layerEnd, h[0], h[1], h[2], h[3], sum)
	}

	return 0
}
