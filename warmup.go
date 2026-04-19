package mongoose

// LRWarmup implements linear learning rate warmup.
// During warmup, LR ramps linearly from 0 to peakLR over warmupSteps.
//
// Every successful training recipe uses warmup:
//   - nanoGPT / llm.c:     2000 steps (Karpathy)
//   - Pythia 70M:           1430 steps (1% of training)
//   - SmolLM2 135M:         2000 steps
//   - TinyStories:          ~500 steps
//
// Without warmup, Adam's momentum/velocity estimates are garbage for the
// first few hundred steps, causing destructive early updates that the
// model never recovers from.
//
// Usage:
//
//	warmup := mongoose.NewLRWarmup(peakLR, warmupSteps)
//	for step := 1; step <= totalSteps; step++ {
//	    lr := warmup.LR(step) // use with cosine decay: lr = warmup.LR(step) * decay.Factor(step)
//	    ...
//	}
type LRWarmup struct {
	peakLR      float32
	warmupSteps int
}

// NewLRWarmup creates a linear warmup schedule.
// peakLR is the target learning rate after warmup completes.
// warmupSteps is the number of steps to ramp up over (0 = no warmup).
func NewLRWarmup(peakLR float32, warmupSteps int) *LRWarmup {
	if warmupSteps < 0 {
		warmupSteps = 0
	}
	return &LRWarmup{
		peakLR:      peakLR,
		warmupSteps: warmupSteps,
	}
}

// LR returns the learning rate at the given step.
// During warmup (step <= warmupSteps): linear ramp from 0 to peakLR.
// After warmup: returns peakLR (combine with CosineDecay for full schedule).
func (w *LRWarmup) LR(step int) float32 {
	if w.warmupSteps <= 0 || step >= w.warmupSteps {
		return w.peakLR
	}
	return w.peakLR * float32(step) / float32(w.warmupSteps)
}

// Done returns true if warmup is complete at the given step.
func (w *LRWarmup) Done(step int) bool {
	return step >= w.warmupSteps
}

// WarmupSteps returns the configured warmup duration.
func (w *LRWarmup) WarmupSteps() int { return w.warmupSteps }

// PeakLR returns the target learning rate.
func (w *LRWarmup) PeakLR() float32 { return w.peakLR }

// RecommendWarmupSteps returns a warmup duration based on total training steps.
// Uses 1% of total steps, clamped to [100, 2000].
// This matches Pythia (1%) and nanoGPT/SmolLM (2000 cap).
func RecommendWarmupSteps(totalSteps int) int {
	w := totalSteps / 100 // 1%
	if w < 100 {
		w = 100
	}
	if w > 2000 {
		w = 2000
	}
	return w
}
