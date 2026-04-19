package mongoose

import "math"

// GradAccumulator accumulates gradients over multiple micro-batches before
// applying the optimizer step. This allows training with effective batch sizes
// much larger than what fits in a single forward/backward pass.
//
// Usage (in training loop):
//
//	acc := mongoose.NewGradAccumulator(model.Params(), accumSteps)
//	for step := 1; step <= totalSteps; step++ {
//	    loss := trainStepFn(...)  // gradients written to param.G
//	    acc.Accumulate()          // adds G into internal buffer, zeros G
//	    if acc.Ready() {
//	        acc.Average()         // divides accumulated grads by accumSteps, writes to G
//	        clipGradsFn(...)
//	        adamStepFn(...)
//	        acc.Reset()           // zeros the accumulation buffer
//	    }
//	}
type GradAccumulator struct {
	params    []GradParam
	buffers   [][]float32 // accumulated gradient sums
	accumN    int         // how many micro-batches to accumulate
	count     int         // current micro-batch count
	totalLoss float64     // accumulated loss for averaging
}

// GradParam is the interface a parameter must satisfy for gradient accumulation.
// Compatible with tParam in train.go — G is the gradient slice, N is the element count.
type GradParam interface {
	Grad() []float32
	Size() int
}

// SimpleGradParam wraps a raw gradient slice for use with GradAccumulator.
type SimpleGradParam struct {
	G []float32
	N int
}

func (p *SimpleGradParam) Grad() []float32 { return p.G }
func (p *SimpleGradParam) Size() int       { return p.N }

// NewGradAccumulator creates an accumulator for the given params.
// accumSteps is the number of micro-batches per optimizer step.
// If accumSteps <= 1, accumulation is effectively a no-op (always ready).
func NewGradAccumulator(params []GradParam, accumSteps int) *GradAccumulator {
	if accumSteps < 1 {
		accumSteps = 1
	}
	buffers := make([][]float32, len(params))
	for i, p := range params {
		buffers[i] = make([]float32, p.Size())
	}
	return &GradAccumulator{
		params:  params,
		buffers: buffers,
		accumN:  accumSteps,
	}
}

// Accumulate adds the current gradients (param.G) into the accumulation buffer
// and zeros param.G for the next micro-batch. Also accumulates loss.
func (a *GradAccumulator) Accumulate(loss float32) {
	for i, p := range a.params {
		g := p.Grad()
		buf := a.buffers[i]
		for j := range buf {
			buf[j] += g[j]
		}
		// Zero the gradient for next micro-batch
		for j := range g {
			g[j] = 0
		}
	}
	a.totalLoss += float64(loss)
	a.count++
}

// Ready returns true when accumSteps micro-batches have been accumulated
// and it's time to run the optimizer.
func (a *GradAccumulator) Ready() bool {
	return a.count >= a.accumN
}

// Average divides the accumulated gradients by accumSteps and writes
// the result back into each param's G slice. Call this before clip + adam.
func (a *GradAccumulator) Average() {
	scale := 1.0 / float64(a.accumN)
	for i, p := range a.params {
		g := p.Grad()
		buf := a.buffers[i]
		for j := range g {
			g[j] = buf[j] * float32(scale)
		}
	}
}

// AverageLoss returns the mean loss over the accumulated micro-batches.
func (a *GradAccumulator) AverageLoss() float32 {
	if a.count == 0 {
		return 0
	}
	return float32(a.totalLoss / float64(a.count))
}

// Reset zeros the accumulation buffers and resets the counter.
// Call after the optimizer step.
func (a *GradAccumulator) Reset() {
	for _, buf := range a.buffers {
		for j := range buf {
			buf[j] = 0
		}
	}
	a.count = 0
	a.totalLoss = 0
}

// Count returns the number of micro-batches accumulated so far.
func (a *GradAccumulator) Count() int { return a.count }

// AccumSteps returns the configured number of accumulation steps.
func (a *GradAccumulator) AccumSteps() int { return a.accumN }

// EffectiveBatchTokens returns the total tokens per optimizer step
// given a per-micro-batch sequence length and batch size of 1.
func (a *GradAccumulator) EffectiveBatchTokens(seqLen int) int {
	return a.accumN * seqLen
}

// RecommendAccumSteps calculates the accumulation steps needed to reach
// a target effective batch size (in tokens) given a per-step sequence length.
//
// Standard targets from the literature:
//   - TinyStories 33M:  ~32K-64K tokens/batch
//   - Pythia 70M:       ~512K tokens/batch (2M for 160M+)
//   - GPT-2 124M:       ~524K tokens/batch
//   - SmolLM 135M:      ~1M tokens/batch
//
// Example: RecommendAccumSteps(65536, 512) = 128 micro-batches
func RecommendAccumSteps(targetBatchTokens, seqLen int) int {
	steps := targetBatchTokens / seqLen
	if steps < 1 {
		steps = 1
	}
	// Round up to nearest power of 2 for clean division
	p := 1
	for p < steps {
		p *= 2
	}
	// But don't overshoot by more than 50%
	if p > steps && p/2 >= steps*2/3 {
		p /= 2
	}
	if p < 1 {
		p = 1
	}
	return p
}

// TokensPerBatch returns recommended batch sizes for different model scales.
// Returns (batchTokens, seqLen) pairs from the literature.
func TokensPerBatch(paramCount int) (batchTokens, seqLen int) {
	switch {
	case paramCount < 20_000_000: // < 20M
		return 32768, 512 // 64 sequences of 512
	case paramCount < 80_000_000: // 20M-80M
		return 65536, 512 // 128 sequences of 512
	case paramCount < 200_000_000: // 80M-200M
		return 524288, 1024 // 512 sequences of 1024
	default: // 200M+
		return 1048576, 2048 // 512 sequences of 2048
	}
}

// EstimateTrainingTokens returns the recommended total training tokens
// based on model size. Uses Chinchilla scaling (20× params) as floor,
// with practical minimums from successful small model training.
func EstimateTrainingTokens(paramCount int) int {
	// Chinchilla optimal: 20 tokens per parameter
	chinchilla := paramCount * 20

	// Practical floors from the literature — small models benefit from overtraining
	switch {
	case paramCount < 50_000_000:
		// TinyStories: 500M tokens works for 33M params
		if chinchilla < 500_000_000 {
			return 500_000_000
		}
	case paramCount < 150_000_000:
		// Pythia/SmolLM: 10B+ tokens for 70-135M models
		if chinchilla < 2_000_000_000 {
			return 2_000_000_000
		}
	}

	return int(math.Max(float64(chinchilla), 500_000_000))
}
