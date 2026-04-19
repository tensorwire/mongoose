package mongoose

import "math"

// CosineDecay implements cosine annealing of the learning rate.
//
// The cosine schedule smoothly decreases the learning rate from peakLR to minLR
// following a half-cosine curve. This is the universal standard in LLM training:
//
//   - Karpathy (nanoGPT, llm.c):  cosine to 10% of peak (minLR = peakLR/10)
//   - Pythia (EleutherAI):        cosine to 10% of peak
//   - SmolLM2 (HuggingFace):      cosine to 10% of peak
//   - GPT-3 (OpenAI):             cosine to 10% of peak
//   - Llama (Meta):               cosine to 10% of peak
//   - Chinchilla (DeepMind):      cosine to 10% of peak
//
// The mathematical form:
//
//	lr(t) = minLR + 0.5 * (peakLR - minLR) * (1 + cos(pi * t / T))
//
// where t is the step (after warmup) and T is the total decay steps.
//
// At t=0:  lr = peakLR  (cos(0) = 1)
// At t=T:  lr = minLR   (cos(pi) = -1)
//
// Usage:
//
//	decay := mongoose.NewCosineDecay(peakLR, minLR, totalSteps, warmupSteps)
//	for step := 1; step <= totalSteps; step++ {
//	    lr := decay.LR(step)
//	    // use lr for this step's optimizer update
//	}
type CosineDecay struct {
	peakLR      float32
	minLR       float32
	totalSteps  int
	warmupSteps int
	decaySteps  int // totalSteps - warmupSteps
}

// NewCosineDecay creates a cosine decay schedule with linear warmup.
//
// This is the complete LR schedule — warmup + decay in one object.
// peakLR is the maximum learning rate (reached at end of warmup).
// minLR is the floor (typically peakLR / 10).
// totalSteps is the full training duration.
// warmupSteps is the linear ramp-up period at the start.
func NewCosineDecay(peakLR, minLR float32, totalSteps, warmupSteps int) *CosineDecay {
	if warmupSteps < 0 {
		warmupSteps = 0
	}
	if warmupSteps > totalSteps {
		warmupSteps = totalSteps
	}
	decaySteps := totalSteps - warmupSteps
	if decaySteps < 1 {
		decaySteps = 1
	}
	return &CosineDecay{
		peakLR:      peakLR,
		minLR:       minLR,
		totalSteps:  totalSteps,
		warmupSteps: warmupSteps,
		decaySteps:  decaySteps,
	}
}

// LR returns the learning rate at the given step.
//
// Three phases:
//  1. Warmup (step < warmupSteps): linear ramp from 0 to peakLR
//  2. Cosine decay (warmupSteps <= step < totalSteps): smooth cosine from peakLR to minLR
//  3. After training (step >= totalSteps): returns minLR
func (c *CosineDecay) LR(step int) float32 {
	if step < 1 {
		step = 1
	}

	// Phase 1: linear warmup
	if step < c.warmupSteps {
		return c.peakLR * float32(step) / float32(c.warmupSteps)
	}

	// Phase 3: past end of schedule
	if step >= c.totalSteps {
		return c.minLR
	}

	// Phase 2: cosine decay
	// Progress through the decay phase: 0.0 at warmup end, 1.0 at totalSteps
	progress := float64(step-c.warmupSteps) / float64(c.decaySteps)
	cosine := 0.5 * (1.0 + math.Cos(math.Pi*progress))
	return c.minLR + float32(cosine)*float32(c.peakLR-c.minLR)
}

// PeakLR returns the maximum learning rate.
func (c *CosineDecay) PeakLR() float32 { return c.peakLR }

// MinLR returns the minimum learning rate (floor).
func (c *CosineDecay) MinLR() float32 { return c.minLR }

// TotalSteps returns the full training duration.
func (c *CosineDecay) TotalSteps() int { return c.totalSteps }

// WarmupSteps returns the warmup duration.
func (c *CosineDecay) WarmupSteps() int { return c.warmupSteps }

// DecaySteps returns the duration of the cosine decay phase.
func (c *CosineDecay) DecaySteps() int { return c.decaySteps }

// NewCosineDecayDefaults creates a cosine decay schedule with standard defaults:
//   - minLR = peakLR / 10  (universal standard)
//   - warmupSteps = 1% of totalSteps, clamped to [100, 2000]
func NewCosineDecayDefaults(peakLR float32, totalSteps int) *CosineDecay {
	minLR := peakLR / 10.0
	warmup := RecommendWarmupSteps(totalSteps)
	return NewCosineDecay(peakLR, minLR, totalSteps, warmup)
}

// LRScheduleTable prints the LR at key points for verification.
// Returns a slice of (step, lr) pairs at 0%, 1%, 10%, 25%, 50%, 75%, 90%, 100% of training.
func (c *CosineDecay) LRScheduleTable() [][2]float64 {
	percentages := []float64{0.0, 0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0}
	table := make([][2]float64, len(percentages))
	for i, pct := range percentages {
		step := int(pct * float64(c.totalSteps))
		if step < 1 {
			step = 1
		}
		table[i] = [2]float64{float64(step), float64(c.LR(step))}
	}
	return table
}
