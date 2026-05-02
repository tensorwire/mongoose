//go:build linux && cgo

package mongoose

import (
	"log"
	"math"
	"sort"
	"unsafe"
)

// NeedleTrainState holds per-layer training state for forward-only needle training
// on GPU-resident Q4 weights. The conductor (Go-side) selects hot rows and
// computes signalScale. The kernel (KNeedleSparse) does the actual weight update.
type NeedleTrainState struct {
	ci       *CUDAFusedInference
	nLayers  int
	dim      int
	nHot     int
	lr       float32
	beta1    float32
	wd       float32

	// Per-layer state
	layers []needleLayerGPU

	// Hidden state snapshots for phase measurement (CPU-side, dim-sized)
	preHidden  []float32
	postHidden []float32

	// Per-layer coherence history for anti-Hebbian
	coherence     []float32
	prevCoherence []float32
	stallCount    []int

	// Hidden history ring buffer for context-gated goodness
	historyRing [3][]float32
	histIdx     int

	// L3 bridge for mailbox-based sync
	l3 *L3Bridge
}

type needleLayerGPU struct {
	// Momentum + delta buffers on GPU for each weight matrix's hot rows
	// Each is [nHot × cols] — compacted sparse representation
	momWq, momWk, momWv, momWo       *Tensor
	momGate, momUp, momDown          *Tensor
	deltaWq, deltaWk, deltaWv, deltaWo *Tensor
	deltaGate, deltaUp, deltaDown    *Tensor
	hotIdx *Tensor // [nHot] int32 on GPU
}

// NewNeedleTrainState creates training state for forward-only needle training.
func NewNeedleTrainState(ci *CUDAFusedInference, nLayers, dim, nHot int, lr, beta1, wd float32) *NeedleTrainState {
	kvDim := ci.kvDim
	ffnDim := ci.ffnDim

	alloc := func(n int) *Tensor {
		return ci.eng.Zeros([]int{n})
	}

	layers := make([]needleLayerGPU, nLayers)
	for i := range layers {
		layers[i] = needleLayerGPU{
			momWq: alloc(nHot * dim), momWk: alloc(nHot * kvDim),
			momWv: alloc(nHot * kvDim), momWo: alloc(nHot * dim),
			momGate: alloc(nHot * ffnDim), momUp: alloc(nHot * ffnDim),
			momDown: alloc(nHot * dim),
			deltaWq: alloc(nHot * dim), deltaWk: alloc(nHot * kvDim),
			deltaWv: alloc(nHot * kvDim), deltaWo: alloc(nHot * dim),
			deltaGate: alloc(nHot * ffnDim), deltaUp: alloc(nHot * ffnDim),
			deltaDown: alloc(nHot * dim),
			hotIdx: alloc(nHot),
		}
	}

	s := &NeedleTrainState{
		ci:            ci,
		nLayers:       nLayers,
		dim:           dim,
		nHot:          nHot,
		lr:            lr,
		beta1:         beta1,
		wd:            wd,
		layers:        layers,
		preHidden:     make([]float32, dim),
		postHidden:    make([]float32, dim),
		coherence:     make([]float32, nLayers),
		prevCoherence: make([]float32, nLayers),
		stallCount:    make([]int, nLayers),
		l3:            ci.hiddenL3,
	}

	vramMB := nLayers * 7 * nHot * dim * 4 * 3 / (1024 * 1024) // rough estimate
	log.Printf("[NEEDLE-GPU] state: %d layers, %d hot rows, lr=%.5f, ~%dMB VRAM", nLayers, nHot, lr, vramMB)
	return s
}

type hotDim struct {
	idx    int
	energy float32
	phase  float32
}

// ForwardOneLayer runs a single layer via direct kernel dispatch and measures
// phase coherence. Uses StepLayerResident — same kernel sequence as the proven
// streaming stepLayer path, but against GPU-resident weights.
func (s *NeedleTrainState) ForwardOneLayer(layer, pos int, hiddenIn, logitsOut []float32) (coherence float32) {
	dim := s.dim

	var lo []float32
	if layer == s.nLayers-1 && logitsOut != nil {
		lo = logitsOut
	}
	if s.ci.WeightsFP16 {
		s.ci.StepLayerResidentFP16(layer, pos, hiddenIn, s.preHidden, s.postHidden, lo)
	} else {
		s.ci.StepLayerResident(layer, pos, hiddenIn, s.preHidden, s.postHidden, lo)
	}

	var dot, nPre, nPost float64
	for i := 0; i < dim; i++ {
		p := float64(s.preHidden[i])
		q := float64(s.postHidden[i])
		dot += p * q
		nPre += p * p
		nPost += q * q
	}
	if nPre > 0 && nPost > 0 {
		coherence = float32(dot / (math.Sqrt(nPre) * math.Sqrt(nPost)))
	}

	if pos == 0 && layer < 3 {
		log.Printf("[NEEDLE-DIAG L%d] pre[0:4]=[%.4f,%.4f,%.4f,%.4f] nPre=%.4f post[0:4]=[%.4f,%.4f,%.4f,%.4f] nPost=%.4f coh=%.6f",
			layer, s.preHidden[0], s.preHidden[1], s.preHidden[2], s.preHidden[3], nPre,
			s.postHidden[0], s.postHidden[1], s.postHidden[2], s.postHidden[3], nPost, coherence)
	}

	return coherence
}

// NeedlePoke fires the needle kernel on hot rows of each weight matrix.
// Dispatches KNeedleQ4 for Q4 weights (scales == nil) or KNeedleSparse for Q8.
func (s *NeedleTrainState) NeedlePoke(layer int, avgG float32) {
	ci := s.ci
	dim := s.dim
	hasQ4 := NeedleQ4Loaded()
	hasQ8 := NeedleQ8Loaded()
	hasFP16 := NeedleFP16Loaded()
	isFP16 := ci.WeightsFP16
	if !hasQ4 && !hasQ8 && !hasFP16 {
		return
	}

	scores := make([]hotDim, dim)
	for d := 0; d < dim; d++ {
		preVal := s.preHidden[d]
		postVal := s.postHidden[d]
		e := postVal
		if e < 0 { e = -e }

		var phase float32
		preMag := preVal; if preMag < 0 { preMag = -preMag }
		if preMag > 1e-8 && e > 1e-8 {
			phase = (preVal * postVal) / (preMag * e)
		}
		scores[d] = hotDim{d, e, phase}
	}

	sort.Slice(scores, func(i, j int) bool {
		return scores[i].energy > scores[j].energy
	})

	hotIdxHost := make([]int32, s.nHot)
	var goodG float32
	nGood := 0
	for i := 0; i < s.nHot && i < len(scores); i++ {
		hotIdxHost[i] = int32(scores[i].idx)
		if scores[i].phase > 0 {
			goodG += scores[i].phase
			nGood++
		}
	}
	if nGood > 0 {
		goodG /= float32(nGood)
	}
	if goodG > 1 { goodG = 1 }

	const stallThresh = 5
	delta := s.coherence[layer] - s.prevCoherence[layer]
	if delta < 0 { delta = -delta }
	if delta < 0.001 {
		s.stallCount[layer]++
	} else {
		s.stallCount[layer] = 0
	}
	s.prevCoherence[layer] = s.coherence[layer]

	if s.stallCount[layer] >= stallThresh {
		goodG = -goodG
	}

	if goodG < 0.05 && goodG > -0.05 {
		return
	}

	signalScale := avgG * goodG

	ci.eng.UploadInto(s.layers[layer].hotIdx,
		unsafe.Slice((*float32)(unsafe.Pointer(&hotIdxHost[0])), s.nHot))

	nl := &s.layers[layer]
	hotIdxP := nl.hotIdx.DevicePtr()

	fire := func(wt, scales *Tensor, mom, delta *Tensor, cols int) {
		if wt == nil { return }
		if isFP16 && hasFP16 {
			KNeedleFP16(wt.DevicePtr(), mom.DevicePtr(), delta.DevicePtr(), hotIdxP,
				signalScale, s.lr, s.beta1, s.wd, s.nHot, cols)
		} else if scales == nil && hasQ4 {
			KNeedleQ4(wt.DevicePtr(), mom.DevicePtr(), delta.DevicePtr(), hotIdxP,
				signalScale, s.lr, s.beta1, s.wd, s.nHot, cols)
		} else if scales != nil && hasQ8 {
			KNeedleQ8(wt.DevicePtr(), scales.DevicePtr(), mom.DevicePtr(), delta.DevicePtr(), hotIdxP,
				signalScale, s.lr, s.beta1, s.wd, s.nHot, cols)
		}
	}

	fire(ci.wq[layer], ci.wqS[layer], nl.momWq, nl.deltaWq, dim)
	fire(ci.wk[layer], ci.wkS[layer], nl.momWk, nl.deltaWk, ci.kvDim)
	fire(ci.wv[layer], ci.wvS[layer], nl.momWv, nl.deltaWv, ci.kvDim)
	fire(ci.wo[layer], ci.woS[layer], nl.momWo, nl.deltaWo, dim)
	fire(ci.wgate[layer], ci.wgateS[layer], nl.momGate, nl.deltaGate, ci.ffnDim)
	fire(ci.wup[layer], ci.wupS[layer], nl.momUp, nl.deltaUp, ci.ffnDim)
	fire(ci.wdown[layer], ci.wdownS[layer], nl.momDown, nl.deltaDown, dim)
}

// UpdateCoherence stores the measured coherence for a layer.
func (s *NeedleTrainState) UpdateCoherence(layer int, c float32) {
	s.coherence[layer] = c
}

// SnapshotHistory saves current post-hidden to the ring buffer for context gating.
func (s *NeedleTrainState) SnapshotHistory() {
	snap := make([]float32, s.dim)
	copy(snap, s.postHidden)
	s.historyRing[s.histIdx%3] = snap
	s.histIdx++
}

// AvgCoherence returns the mean coherence across all layers.
func (s *NeedleTrainState) AvgCoherence() float32 {
	var sum float32
	for _, c := range s.coherence {
		sum += c
	}
	return sum / float32(s.nLayers)
}

// StallCount returns how many layers are stalled.
func (s *NeedleTrainState) StallCount() int {
	n := 0
	for _, sc := range s.stallCount {
		if sc >= 5 { n++ }
	}
	return n
}
