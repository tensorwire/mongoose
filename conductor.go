package mongoose

import (
	"math"
)

// HalfToFloat converts IEEE 754 half-precision (uint16) to float32.
func HalfToFloat(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff
	if exp == 0 {
		if mant == 0 { return math.Float32frombits(sign << 31) }
		// Denorm
		for mant&0x400 == 0 { mant <<= 1; exp-- }
		exp++; mant &= 0x3ff
		return math.Float32frombits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
	}
	if exp == 31 {
		if mant == 0 { return math.Float32frombits((sign << 31) | 0x7f800000) } // Inf
		return math.Float32frombits((sign << 31) | 0x7fc00000) // NaN
	}
	return math.Float32frombits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
}

// FloatToHalf converts float32 to IEEE 754 half-precision (uint16).
func FloatToHalf(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := (bits >> 16) & 0x8000
	exp := int((bits>>23)&0xff) - 127
	mant := bits & 0x7fffff
	if exp > 15 { return uint16(sign | 0x7c00) } // overflow → Inf
	if exp < -14 { return uint16(sign) }           // underflow → 0
	return uint16(sign | uint32(exp+15)<<10 | (mant >> 13))
}

// Conductor tracks which embedding rows are electrically active in the DNA.
//
// Real DNA conductivity: charge flows through stacked base pairs via
// pi-orbital overlap. Active genes conduct. Silent genes don't.
// Mismatches and damage break the charge transfer chain.
//
// In the optimizer: the embedding table is the genome (50K rows = 50K genes).
// Each training step "expresses" a few genes (non-zero softmax → non-zero
// gradient for those embedding rows). The vast majority are silent.
//
// The conductor maintains a charge map — how often each row has been
// active over the last N steps. Rows with zero charge are dead genes.
// The graph can skip their gradient computation entirely.
//
// No per-step threshold. No sparse matmul. Just a pre-computed hot index
// that the signal chain maintains as a natural byproduct of observation.
//
// The electricity that surrounds DNA IS the dedup.
type Conductor struct {
	charge    []float32 // per-row charge level [0, 1]. 0 = dead, 1 = fully active
	hitCount  []int     // raw hit count this window
	vocabSize int
	window    int       // steps per observation window
	stepCount int       // steps since last decay
	decay     float32   // charge decay per window (e.g., 0.5 = halve each window)

	// The hot index: sorted list of active row indices.
	// Recomputed every window from the charge map.
	hotRows   []int32
	hotRatio  float32   // fraction of rows that are hot (for logging)
	threshold float32   // charge below this = dead gene
}

// NewConductor creates a charge tracker for the embedding table.
// vocabSize is the number of embedding rows (e.g., 50257 for GPT-2).
// window is how many steps between charge decay cycles.
func NewConductor(vocabSize, window int) *Conductor {
	return &Conductor{
		charge:    make([]float32, vocabSize),
		hitCount:  make([]int, vocabSize),
		vocabSize: vocabSize,
		window:    window,
		decay:     0.5,      // halve charge each window
		threshold: 0.01,     // 1% charge = effectively dead
	}
}

// Observe records which tokens appeared in this step's sequence.
// Called once per training step with the input token IDs.
// The conductor notes which genes were expressed.
func (c *Conductor) Observe(tokens []int32) {
	for _, t := range tokens {
		if int(t) < c.vocabSize {
			c.hitCount[t]++
		}
	}
	c.stepCount++

	// End of observation window: update charge map, rebuild hot index
	if c.stepCount >= c.window {
		c.updateCharge()
		c.rebuildHotIndex()
		c.stepCount = 0
	}
}

// updateCharge decays existing charge and adds new observations.
func (c *Conductor) updateCharge() {
	maxHits := 0
	for _, h := range c.hitCount {
		if h > maxHits {
			maxHits = h
		}
	}
	if maxHits == 0 {
		maxHits = 1
	}

	invMax := 1.0 / float32(maxHits)
	for i := range c.charge {
		// Decay old charge
		c.charge[i] *= c.decay
		// Add new charge proportional to hit frequency
		c.charge[i] += float32(c.hitCount[i]) * invMax * (1.0 - c.decay)
		// Clamp
		if c.charge[i] > 1.0 {
			c.charge[i] = 1.0
		}
	}

	// Reset hit counts for next window
	for i := range c.hitCount {
		c.hitCount[i] = 0
	}
}

// rebuildHotIndex collects all rows with charge above threshold.
func (c *Conductor) rebuildHotIndex() {
	c.hotRows = c.hotRows[:0]
	for i, ch := range c.charge {
		if ch >= c.threshold {
			c.hotRows = append(c.hotRows, int32(i))
		}
	}
	c.hotRatio = float32(len(c.hotRows)) / float32(c.vocabSize)
}

// HotRows returns the indices of electrically active embedding rows.
// Only these rows need gradient computation. Dead rows are silent genes.
func (c *Conductor) HotRows() []int32 {
	return c.hotRows
}

// HotRatio returns the fraction of rows that are active (for logging).
// At 50K vocab with typical text, expect 5-15% hot.
func (c *Conductor) HotRatio() float32 {
	return c.hotRatio
}

// IsHot returns true if a specific row is electrically active.
func (c *Conductor) IsHot(row int) bool {
	if row >= c.vocabSize {
		return false
	}
	return c.charge[row] >= c.threshold
}

// Charge returns the charge level for a specific row.
func (c *Conductor) Charge(row int) float32 {
	if row >= c.vocabSize {
		return 0
	}
	return c.charge[row]
}

// DeadCount returns the number of dead (uncharged) rows.
func (c *Conductor) DeadCount() int {
	dead := 0
	for _, ch := range c.charge {
		if ch < c.threshold {
			dead++
		}
	}
	return dead
}

// Stats returns (hot, dead, total) counts.
func (c *Conductor) Stats() (hot, dead, total int) {
	total = c.vocabSize
	for _, ch := range c.charge {
		if ch >= c.threshold {
			hot++
		} else {
			dead++
		}
	}
	return
}

// Mask returns a float32 slice for the GPU conductor mask.
// 1.0 for hot rows, 0.0 for dead rows. Same layout as the Metal buffer.
func (c *Conductor) Mask() []float32 {
	mask := make([]float32, c.vocabSize)
	for i, ch := range c.charge {
		if ch >= c.threshold {
			mask[i] = 1.0
		}
	}
	return mask
}

// ProjectionTracker tracks per-row activation charge for a weight matrix.
// During the forward pass, the input activation reveals which dimensions
// carry signal. For Q = input @ W^T, the L2 norm of input per-position
// tells us how much signal flows through each output row of W.
//
// The tracker maintains charge per output row using the same decay/window
// mechanism as the embedding conductor. Hot rows = worth updating.
type ProjectionTracker struct {
	charge    []float32 // per-row charge [0, 1]
	accumNorm []float32 // accumulated L2 norm per output row this window
	nRows     int
	nCols     int
	window    int
	stepCount int
	decay     float32
	threshold float32
	hotRows   []int32
}

// NewProjectionTracker creates a row activity tracker for a weight projection.
func NewProjectionTracker(nRows, nCols, window int) *ProjectionTracker {
	return &ProjectionTracker{
		charge:    make([]float32, nRows),
		accumNorm: make([]float32, nRows),
		nRows:     nRows,
		nCols:     nCols,
		window:    window,
		decay:     0.5,
		threshold: 0.01,
	}
}

// ObserveActivation records input activation magnitudes for this projection.
// inputSlice is the FP32 input to the matmul, shape [n, inDim].
// For W[outDim, inDim], each output row i's activity is the mean L2 contribution
// of the input across positions. We sample to keep it fast.
func (p *ProjectionTracker) ObserveActivation(inputSlice []float32, n, inDim int) {
	if len(inputSlice) == 0 || n == 0 { return }

	// Sample one position (middle of sequence) for speed
	pos := n / 2
	off := pos * inDim

	// For each output row, estimate activation by sampling input dimensions.
	// Since W[row] @ input[pos] = dot product, rows that align with large
	// input values are "active". We use input L2 norm per column block
	// as a proxy — cheap and catches which dimensions carry signal.
	//
	// Simpler: just record the L2 norm of the sampled input position.
	// All weight rows see the same input, but the ones producing outputs
	// that matter downstream will accumulate more charge through loss feedback.
	// For now, charge all rows proportional to input energy.
	var energy float32
	end := off + inDim
	if end > len(inputSlice) { end = len(inputSlice) }
	stride := inDim / 32
	if stride < 1 { stride = 1 }
	for i := off; i < end; i += stride {
		v := inputSlice[i]
		energy += v * v
	}
	energy = float32(math.Sqrt(float64(energy)))

	// Charge is proportional to input energy. Hot input = hot rows.
	for i := range p.accumNorm {
		p.accumNorm[i] += energy
	}

	p.stepCount++
	if p.stepCount >= p.window {
		p.updateCharge()
		p.stepCount = 0
	}
}

// ObserveOutput records which output dimensions produced large values.
// Builds hot index immediately on first observation — only step 1 is noop.
// outputSlice is shape [n, outDim]. Sample one position.
func (p *ProjectionTracker) ObserveOutput(outputSlice []float32, n, outDim int) {
	if len(outputSlice) == 0 || n == 0 || outDim != p.nRows { return }

	pos := n / 2
	off := pos * outDim
	end := off + outDim
	if end > len(outputSlice) { end = len(outputSlice) }

	for i := off; i < end; i++ {
		row := i - off
		v := outputSlice[i]
		if v < 0 { v = -v }
		p.accumNorm[row] += v
	}

	p.stepCount++

	// First observation: build hot index immediately so step 2 can update.
	// After that, use the normal window for stability.
	if p.stepCount == 1 || p.stepCount >= p.window {
		p.updateCharge()
		if p.stepCount >= p.window { p.stepCount = 0 }
	}
}

func (p *ProjectionTracker) updateCharge() {
	var maxNorm float32
	for _, n := range p.accumNorm {
		if n > maxNorm { maxNorm = n }
	}
	if maxNorm == 0 { maxNorm = 1 }

	invMax := 1.0 / maxNorm
	for i := range p.charge {
		p.charge[i] *= p.decay
		p.charge[i] += p.accumNorm[i] * invMax * (1.0 - p.decay)
		if p.charge[i] > 1.0 { p.charge[i] = 1.0 }
		p.accumNorm[i] = 0
	}

	// Rebuild hot index
	p.hotRows = p.hotRows[:0]
	for i, ch := range p.charge {
		if ch >= p.threshold {
			p.hotRows = append(p.hotRows, int32(i))
		}
	}
}

// HotRows returns indices of active weight rows for this projection.
func (p *ProjectionTracker) HotRows() []int32 { return p.hotRows }

// IsHot returns true if a specific row is active.
func (p *ProjectionTracker) IsHot(row int) bool {
	if row >= p.nRows { return false }
	return p.charge[row] >= p.threshold
}

// HotRatio returns fraction of rows that are active.
func (p *ProjectionTracker) HotRatio() float32 {
	return float32(len(p.hotRows)) / float32(p.nRows)
}

// WeightAccum tracks sparse accumulated deltas for weight positions.
// Only ~49 positions are active per step. The conductor accumulates their
// deltas until they cross the INT8 bucket threshold, then issues a fold.
type WeightAccum struct {
	deltas map[int]float32 // position → accumulated delta
	scale  float32         // current INT8 scale (bucket size = scale/127)
}

// NewWeightAccum creates a sparse delta accumulator for one projection.
func NewWeightAccum() *WeightAccum {
	return &WeightAccum{deltas: make(map[int]float32)}
}

// Accumulate adds a delta to a position. Returns true if the accumulated
// delta now crosses the INT8 fold threshold (half a quant bucket).
func (w *WeightAccum) Accumulate(pos int, delta float32, scale float32) bool {
	w.scale = scale
	w.deltas[pos] += delta
	bucket := scale / 127.0
	d := w.deltas[pos]
	if d > 0.5*bucket || d < -0.5*bucket {
		return true // ready to fold into INT8
	}
	return false
}

// Drain removes and returns the accumulated delta for a position (after fold).
func (w *WeightAccum) Drain(pos int) float32 {
	d := w.deltas[pos]
	delete(w.deltas, pos)
	return d
}

// PendingDeltas returns all positions with non-zero accumulated deltas.
// Used by dequant to scatter-add into FP32 cache before GEMM.
func (w *WeightAccum) PendingDeltas() map[int]float32 {
	return w.deltas
}

// DrainAll clears all pending deltas after they've been uploaded to GPU.
func (w *WeightAccum) DrainAll() {
	for k := range w.deltas {
		delete(w.deltas, k)
	}
}

// Len returns the number of positions with pending deltas.
func (w *WeightAccum) Len() int {
	return len(w.deltas)
}

// ChargeEntropy returns the entropy of the charge distribution.
// High entropy = evenly distributed activity (diverse vocab usage).
// Low entropy = concentrated on few rows (repetitive text).
func (c *Conductor) ChargeEntropy() float64 {
	var sum float64
	for _, ch := range c.charge {
		sum += float64(ch)
	}
	if sum == 0 {
		return 0
	}

	var entropy float64
	for _, ch := range c.charge {
		p := float64(ch) / sum
		if p > 0 {
			entropy -= p * math.Log2(p)
		}
	}
	return entropy
}
