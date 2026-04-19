package mongoose

import (
	"math"
	"testing"
)

// --- CosineDecay tests ---

func TestCosineDecay_BoundaryValues(t *testing.T) {
	// Standard config: peakLR=6e-4, minLR=6e-5, 10000 steps, 500 warmup
	d := NewCosineDecay(6e-4, 6e-5, 10000, 500)

	// At step 1 (start of warmup): should be near zero
	lr1 := d.LR(1)
	if lr1 > 2e-6 {
		t.Errorf("step 1: want near-zero warmup LR, got %e", lr1)
	}

	// At end of warmup: should equal peakLR
	lrWarmup := d.LR(500)
	if math.Abs(float64(lrWarmup-6e-4)) > 1e-7 {
		t.Errorf("step 500 (warmup end): want 6e-4, got %e", lrWarmup)
	}

	// At midpoint of decay: should be midpoint of peak and min
	// cosine at 50% = 0, so lr = min + 0.5*(peak-min) = 3.3e-4
	midStep := 500 + (10000-500)/2 // 5250
	lrMid := d.LR(midStep)
	expectedMid := float32(6e-5 + 0.5*(6e-4-6e-5)) // 3.3e-4
	if math.Abs(float64(lrMid-expectedMid)) > 1e-6 {
		t.Errorf("step %d (midpoint): want %e, got %e", midStep, expectedMid, lrMid)
	}

	// At end of training: should equal minLR
	lrEnd := d.LR(10000)
	if math.Abs(float64(lrEnd-6e-5)) > 1e-7 {
		t.Errorf("step 10000 (end): want 6e-5, got %e", lrEnd)
	}

	// Past end: should still be minLR
	lrPast := d.LR(20000)
	if math.Abs(float64(lrPast-6e-5)) > 1e-7 {
		t.Errorf("step 20000 (past end): want 6e-5, got %e", lrPast)
	}
}

func TestCosineDecay_Monotonic(t *testing.T) {
	// After warmup, LR must be monotonically non-increasing
	d := NewCosineDecay(1e-3, 1e-4, 50000, 1000)

	prevLR := d.LR(1000) // start of decay
	for step := 1001; step <= 50000; step += 100 {
		lr := d.LR(step)
		if lr > prevLR+1e-10 { // small epsilon for float imprecision
			t.Errorf("non-monotonic at step %d: prev=%e, curr=%e", step, prevLR, lr)
		}
		prevLR = lr
	}
}

func TestCosineDecay_WarmupLinear(t *testing.T) {
	d := NewCosineDecay(1e-3, 1e-4, 10000, 1000)

	// Warmup should be strictly linear
	for step := 1; step < 1000; step += 50 {
		lr := d.LR(step)
		expected := float32(step) / 1000.0 * 1e-3
		if math.Abs(float64(lr-expected)) > 1e-8 {
			t.Errorf("warmup step %d: want %e (linear), got %e", step, expected, lr)
		}
	}
}

func TestCosineDecay_NoWarmup(t *testing.T) {
	d := NewCosineDecay(1e-3, 1e-4, 10000, 0)

	// Step 1 should be at peakLR (no warmup)
	lr1 := d.LR(1)
	if math.Abs(float64(lr1-1e-3)) > 1e-6 {
		t.Errorf("step 1 (no warmup): want 1e-3, got %e", lr1)
	}
}

func TestCosineDecayDefaults(t *testing.T) {
	d := NewCosineDecayDefaults(6e-4, 100000)

	// minLR should be peakLR / 10
	if math.Abs(float64(d.MinLR()-6e-5)) > 1e-8 {
		t.Errorf("minLR: want 6e-5, got %e", d.MinLR())
	}

	// warmup should be 1% of 100000 = 1000
	if d.WarmupSteps() != 1000 {
		t.Errorf("warmup: want 1000, got %d", d.WarmupSteps())
	}
}

func TestLRScheduleTable(t *testing.T) {
	d := NewCosineDecay(1e-3, 1e-4, 10000, 500)
	table := d.LRScheduleTable()

	if len(table) != 8 {
		t.Errorf("table length: want 8, got %d", len(table))
	}

	// First entry should be step 1
	if table[0][0] != 1 {
		t.Errorf("first step: want 1, got %.0f", table[0][0])
	}

	// Last entry should be step 10000 at minLR
	last := table[len(table)-1]
	if last[0] != 10000 {
		t.Errorf("last step: want 10000, got %.0f", last[0])
	}
	if math.Abs(last[1]-1e-4) > 1e-7 {
		t.Errorf("last LR: want 1e-4, got %e", last[1])
	}
}

// --- LRWarmup standalone tests ---

func TestLRWarmup_Linear(t *testing.T) {
	w := NewLRWarmup(1e-3, 500)

	// Step 250 = 50% warmup → 5e-4
	lr := w.LR(250)
	if math.Abs(float64(lr-5e-4)) > 1e-8 {
		t.Errorf("step 250: want 5e-4, got %e", lr)
	}

	// Step 500 = warmup complete → peakLR
	lr = w.LR(500)
	if math.Abs(float64(lr-1e-3)) > 1e-8 {
		t.Errorf("step 500: want 1e-3, got %e", lr)
	}

	// Step 1000 = past warmup → still peakLR
	lr = w.LR(1000)
	if math.Abs(float64(lr-1e-3)) > 1e-8 {
		t.Errorf("step 1000: want 1e-3, got %e", lr)
	}
}

func TestLRWarmup_ZeroSteps(t *testing.T) {
	w := NewLRWarmup(1e-3, 0)

	// No warmup → always peakLR
	if w.LR(1) != 1e-3 {
		t.Errorf("step 1 (no warmup): want 1e-3, got %e", w.LR(1))
	}
}

func TestRecommendWarmupSteps(t *testing.T) {
	cases := []struct {
		total    int
		expected int
	}{
		{1000, 100},     // 1% = 10, clamped to 100
		{10000, 100},    // 1% = 100
		{50000, 500},    // 1% = 500
		{200000, 2000},  // 1% = 2000
		{1000000, 2000}, // 1% = 10000, clamped to 2000
	}
	for _, tc := range cases {
		got := RecommendWarmupSteps(tc.total)
		if got != tc.expected {
			t.Errorf("RecommendWarmupSteps(%d): want %d, got %d", tc.total, tc.expected, got)
		}
	}
}

// --- GradAccumulator tests ---

func TestGradAccumulator_Averaging(t *testing.T) {
	// 3 params of size 4, accumulate 4 micro-batches
	params := []GradParam{
		&SimpleGradParam{G: make([]float32, 4), N: 4},
		&SimpleGradParam{G: make([]float32, 4), N: 4},
		&SimpleGradParam{G: make([]float32, 4), N: 4},
	}
	acc := NewGradAccumulator(params, 4)

	// Simulate 4 micro-batches, each with gradient = 1.0
	for mb := 0; mb < 4; mb++ {
		for _, p := range params {
			g := p.Grad()
			for i := range g {
				g[i] = 1.0
			}
		}
		acc.Accumulate(2.0) // loss = 2.0 each
	}

	if !acc.Ready() {
		t.Error("accumulator should be ready after 4 micro-batches")
	}

	// Average loss should be 2.0
	avgLoss := acc.AverageLoss()
	if math.Abs(float64(avgLoss-2.0)) > 1e-6 {
		t.Errorf("average loss: want 2.0, got %f", avgLoss)
	}

	// Average: sum=4.0, count=4, avg=1.0
	acc.Average()
	for pi, p := range params {
		g := p.Grad()
		for i, v := range g {
			if math.Abs(float64(v-1.0)) > 1e-6 {
				t.Errorf("param[%d].G[%d]: want 1.0, got %f", pi, i, v)
			}
		}
	}

	// Reset should zero everything
	acc.Reset()
	if acc.Count() != 0 {
		t.Errorf("count after reset: want 0, got %d", acc.Count())
	}
}

func TestGradAccumulator_VaryingGradients(t *testing.T) {
	// 1 param of size 2, accumulate 2 micro-batches with different gradients
	p := &SimpleGradParam{G: make([]float32, 2), N: 2}
	acc := NewGradAccumulator([]GradParam{p}, 2)

	// Micro-batch 1: gradient = [2.0, 4.0]
	p.G[0] = 2.0
	p.G[1] = 4.0
	acc.Accumulate(1.0)

	// Micro-batch 2: gradient = [6.0, 8.0]
	p.G[0] = 6.0
	p.G[1] = 8.0
	acc.Accumulate(3.0)

	acc.Average()

	// Average of [2,4] and [6,8] = [4.0, 6.0]
	if math.Abs(float64(p.G[0]-4.0)) > 1e-6 {
		t.Errorf("G[0]: want 4.0, got %f", p.G[0])
	}
	if math.Abs(float64(p.G[1]-6.0)) > 1e-6 {
		t.Errorf("G[1]: want 6.0, got %f", p.G[1])
	}

	// Average loss: (1.0 + 3.0) / 2 = 2.0
	if math.Abs(float64(acc.AverageLoss()-2.0)) > 1e-6 {
		t.Errorf("avg loss: want 2.0, got %f", acc.AverageLoss())
	}
}

func TestGradAccumulator_SingleStep(t *testing.T) {
	// accumSteps=1 means every micro-batch triggers an optimizer step
	p := &SimpleGradParam{G: make([]float32, 2), N: 2}
	acc := NewGradAccumulator([]GradParam{p}, 1)

	p.G[0] = 3.0
	p.G[1] = 7.0
	acc.Accumulate(1.5)

	if !acc.Ready() {
		t.Error("should be ready after 1 step with accumSteps=1")
	}

	acc.Average()
	// With accumSteps=1, average = the gradient itself
	if math.Abs(float64(p.G[0]-3.0)) > 1e-6 {
		t.Errorf("G[0]: want 3.0, got %f", p.G[0])
	}
}

func TestRecommendAccumSteps(t *testing.T) {
	// 64K tokens target, seqLen=512 → 128 steps
	got := RecommendAccumSteps(65536, 512)
	if got != 128 {
		t.Errorf("RecommendAccumSteps(65536, 512): want 128, got %d", got)
	}

	// 32K tokens, seqLen=512 → 64 steps
	got = RecommendAccumSteps(32768, 512)
	if got != 64 {
		t.Errorf("RecommendAccumSteps(32768, 512): want 64, got %d", got)
	}
}

func TestTokensPerBatch(t *testing.T) {
	// 47M param model → 65536 tokens, seqLen 512
	batchTokens, seqLen := TokensPerBatch(47_000_000)
	if batchTokens != 65536 {
		t.Errorf("47M model batch tokens: want 65536, got %d", batchTokens)
	}
	if seqLen != 512 {
		t.Errorf("47M model seq len: want 512, got %d", seqLen)
	}
}
