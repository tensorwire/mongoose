//go:build linux && cgo

package mongoose

import (
	"log"
	"math"
)

// NeedleInlineState holds per-layer state for inline per-weight needle training.
// KNeedleInline fires BEFORE each matmul, updating the FP32 weight cache
// in-place. Each weight element is individually gated by the row mask.
// Inference IS training — the poke happens during the forward pass.
type NeedleInlineState struct {
	ci      *CUDAFusedInference
	nLayers int
	dim     int
	lr      float32
	beta1   float32
	wd      float32

	layers []inlineLayerState

	// Phase measurement
	preHidden  []float32
	postHidden []float32
	coherence  []float32
	prevCoh    []float32
	stallCount []int
}

type inlineLayerState struct {
	// Per weight matrix: mask + compacted momentum/delta
	// mask: [rows] float32 — 0=frozen, >0 = compact_row_index+1
	// mom/delta: [nActive * cols] FP16 (compacted)
	masks  [7]*Tensor // wq,wk,wv,wo,gate,up,down
	moms   [7]*Tensor
	deltas [7]*Tensor
	nActive int // how many rows are currently active per matrix
}

// NewNeedleInlineState creates per-weight inline needle training state.
// nActive controls how many rows per matrix are trainable. The rest stay frozen.
func NewNeedleInlineState(ci *CUDAFusedInference, nLayers, nActive int, lr, beta1, wd float32) *NeedleInlineState {
	dim := ci.dim
	kvDim := ci.kvDim
	ffnDim := ci.ffnDim

	// Matrix dimensions: [rows, cols] for each weight type
	// wq: [dim, dim], wk: [kvDim, dim], wv: [kvDim, dim], wo: [dim, dim]
	// gate: [ffnDim, dim], up: [ffnDim, dim], down: [dim, ffnDim]
	type matSpec struct {
		rows, cols int
	}
	specs := [7]matSpec{
		{dim, dim}, {kvDim, dim}, {kvDim, dim}, {dim, dim},
		{ffnDim, dim}, {ffnDim, dim}, {dim, ffnDim},
	}

	layers := make([]inlineLayerState, nLayers)
	for l := range layers {
		layers[l].nActive = nActive
		for m := 0; m < 7; m++ {
			rows := specs[m].rows
			cols := specs[m].cols
			layers[l].masks[m] = ci.eng.Zeros([]int{rows})
			// FP16 compacted: nActive * cols * 2 bytes = nActive * cols / 2 float32s
			fp16Size := (nActive * cols * 2 + 3) / 4 // ceil to float32 count
			layers[l].moms[m] = ci.eng.Zeros([]int{fp16Size})
			layers[l].deltas[m] = ci.eng.Zeros([]int{fp16Size})
		}
	}

	vramMB := 0
	for m := 0; m < 7; m++ {
		vramMB += nLayers * (specs[m].rows*4 + nActive*specs[m].cols*2*2) // mask + mom + delta
	}
	vramMB /= 1024 * 1024

	s := &NeedleInlineState{
		ci:         ci,
		nLayers:    nLayers,
		dim:        dim,
		lr:         lr,
		beta1:      beta1,
		wd:         wd,
		layers:     layers,
		preHidden:  make([]float32, dim),
		postHidden: make([]float32, dim),
		coherence:  make([]float32, nLayers),
		prevCoh:    make([]float32, nLayers),
		stallCount: make([]int, nLayers),
	}

	log.Printf("[NEEDLE-INLINE] state: %d layers, %d active rows/matrix, lr=%.5f, ~%dMB VRAM",
		nLayers, nActive, lr, vramMB)
	return s
}

// SetMask updates the row mask for a weight matrix based on per-dim goodness.
// Rows with positive phase coherence get activated; others stay frozen.
// The mask value encodes the compacted row index (1-based).
func (s *NeedleInlineState) SetMask(layer, matIdx int, goodness []float32, nRows int) {
	ls := &s.layers[layer]
	mask := make([]float32, nRows)
	compactIdx := 1 // 1-based
	for r := 0; r < nRows && compactIdx <= ls.nActive; r++ {
		if r < len(goodness) && goodness[r] > 0 {
			mask[r] = float32(compactIdx)
			compactIdx++
		}
	}
	s.ci.eng.UploadInto(ls.masks[matIdx], mask)
}

// ForwardOneLayer runs a single layer via direct kernel dispatch and measures
// phase coherence. Uses StepLayerResident — same kernel sequence as the proven
// streaming stepLayer path, but against GPU-resident weights.
func (s *NeedleInlineState) ForwardOneLayer(layer, pos int, hiddenIn, logitsOut []float32) float32 {
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
	coherence := float32(0)
	if nPre > 0 && nPost > 0 {
		coherence = float32(dot / (math.Sqrt(nPre) * math.Sqrt(nPost)))
	}
	s.coherence[layer] = coherence
	return coherence
}

// PokeInline fires KNeedleInline on each weight matrix for this layer.
// The mask determines which rows are active. signalScale gates the update.
func (s *NeedleInlineState) PokeInline(layer int, signalScale float32) {
	if !NeedleInlineLoaded() {
		return
	}

	ci := s.ci
	ls := &s.layers[layer]
	dim := ci.dim

	// Per-dim goodness for mask selection
	goodness := make([]float32, dim)
	for d := 0; d < dim; d++ {
		preVal := s.preHidden[d]
		postVal := s.postHidden[d]
		if preVal != 0 && postVal != 0 {
			preMag := preVal; if preMag < 0 { preMag = -preMag }
			postMag := postVal; if postMag < 0 { postMag = -postMag }
			goodness[d] = (preVal * postVal) / (preMag * postMag) // ±1
		}
	}

	// Anti-Hebbian stall check
	const stallThresh = 5
	delta := s.coherence[layer] - s.prevCoh[layer]
	if delta < 0 { delta = -delta }
	if delta < 0.001 {
		s.stallCount[layer]++
	} else {
		s.stallCount[layer] = 0
	}
	s.prevCoh[layer] = s.coherence[layer]

	if s.stallCount[layer] >= stallThresh {
		signalScale = -signalScale
	}

	if signalScale < 0.05 && signalScale > -0.05 {
		return
	}

	// Set masks and fire inline needle on each weight matrix
	// matIdx: 0=wq, 1=wk, 2=wv, 3=wo, 4=gate, 5=up, 6=down
	type matRef struct {
		wt   *Tensor
		rows int
		cols int
	}
	mats := [7]matRef{
		{ci.wq[layer], dim, dim},
		{ci.wk[layer], ci.kvDim, dim},
		{ci.wv[layer], ci.kvDim, dim},
		{ci.wo[layer], dim, dim},
		{ci.wgate[layer], ci.ffnDim, dim},
		{ci.wup[layer], ci.ffnDim, dim},
		{ci.wdown[layer], dim, ci.ffnDim},
	}

	for m := 0; m < 7; m++ {
		mt := mats[m]
		if mt.wt == nil { continue }

		// Set mask based on goodness (use output dim for attn, input dim for FFN)
		maskDim := mt.rows
		g := goodness
		if maskDim > len(g) {
			g = make([]float32, maskDim) // zero-fill for ffnDim > dim
		}
		s.SetMask(layer, m, g, maskDim)

		KNeedleInline(
			mt.wt.DevicePtr(),
			nil, // scales
			nil, // cache (uses weight data directly)
			ls.moms[m].DevicePtr(),
			ls.deltas[m].DevicePtr(),
			ls.masks[m].DevicePtr(),
			signalScale, s.lr, s.beta1, s.wd,
			mt.rows, mt.cols,
		)
	}
}

func (s *NeedleInlineState) AvgCoherence() float32 {
	var sum float32
	for _, c := range s.coherence { sum += c }
	return sum / float32(s.nLayers)
}

func (s *NeedleInlineState) StallCount() int {
	n := 0
	for _, sc := range s.stallCount {
		if sc >= 5 { n++ }
	}
	return n
}
