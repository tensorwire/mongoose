//go:build darwin && cgo

package mongoose

import (
	"math"
	"testing"
)

// Targets cross the cgo boundary as the float32 bit-pattern of an int32 —
// see ai/train_finetune_metal.go, which writes math.Float32frombits(uint32(tgt)).
func targetBits(v int32) float32 { return math.Float32frombits(uint32(v)) }

// Prompt/completion loss masking is the whole of B2. If it silently failed the
// trainer would still put ~98% of its gradient on reproducing a 2,700-token
// system prompt, while every source-level test passed and the loss curve looked
// entirely reasonable.
//
// A negative target means "this position trains nothing": zero loss, zero
// gradient, and no read of logits[target] — which would index out of bounds.
func TestSoftmaxCEGradMasksNegativeTargets(t *testing.T) {
	m := NewMetal()
	if m == nil {
		t.Skip("Metal unavailable")
	}

	const seqLen, vocab = 4, 8

	logitsHost := make([]float32, seqLen*vocab)
	for i := range logitsHost {
		logitsHost[i] = float32(i%vocab) * 0.25
	}
	logits := m.FromHost(logitsHost, []int{seqLen, vocab})
	defer m.Release(logits)

	// Positions 0 and 2 are masked (prompt); 1 and 3 are supervised.
	targets := m.FromHost([]float32{
		targetBits(-1), targetBits(3), targetBits(-1), targetBits(5),
	}, []int{seqLen})
	defer m.Release(targets)

	losses := m.Zeros([]int{seqLen})
	defer m.Release(losses)
	grad := m.Zeros([]int{seqLen, vocab})
	defer m.Release(grad)

	m.SoftmaxCEGrad(logits, targets, losses, grad, seqLen, vocab, 1.0/float32(seqLen))

	gotLoss := m.ToHost(losses)
	gotGrad := m.ToHost(grad)

	for _, pos := range []int{0, 2} {
		if gotLoss[pos] != 0 {
			t.Errorf("masked position %d has loss %v, want 0", pos, gotLoss[pos])
		}
		for j := 0; j < vocab; j++ {
			if g := gotGrad[pos*vocab+j]; g != 0 {
				t.Errorf("masked position %d, vocab %d has gradient %v, want 0 — "+
					"the prompt is still being trained", pos, j, g)
			}
		}
	}

	// Supervised positions must be unaffected by their masked neighbours.
	for _, pos := range []int{1, 3} {
		if gotLoss[pos] <= 0 {
			t.Errorf("supervised position %d has loss %v, want > 0", pos, gotLoss[pos])
		}
		var nonzero bool
		for j := 0; j < vocab; j++ {
			if gotGrad[pos*vocab+j] != 0 {
				nonzero = true
			}
		}
		if !nonzero {
			t.Errorf("supervised position %d produced no gradient", pos)
		}
	}
}

// The gradient at a supervised position must still be exactly softmax(logits)
// with 1 subtracted at the target, scaled by invN. Masking must not perturb the
// arithmetic of the positions it leaves alone.
func TestSoftmaxCEGradUnmaskedMathIsUnchanged(t *testing.T) {
	m := NewMetal()
	if m == nil {
		t.Skip("Metal unavailable")
	}

	const seqLen, vocab = 1, 6
	const target = 2
	const invN = 1.0

	logitsHost := []float32{0.1, 0.5, 2.0, -0.3, 1.1, 0.0}
	logits := m.FromHost(logitsHost, []int{seqLen, vocab})
	defer m.Release(logits)
	targets := m.FromHost([]float32{targetBits(target)}, []int{seqLen})
	defer m.Release(targets)
	losses := m.Zeros([]int{seqLen})
	defer m.Release(losses)
	grad := m.Zeros([]int{seqLen, vocab})
	defer m.Release(grad)

	m.SoftmaxCEGrad(logits, targets, losses, grad, seqLen, vocab, invN)

	// CPU reference.
	mx := logitsHost[0]
	for _, v := range logitsHost {
		if v > mx {
			mx = v
		}
	}
	var se float64
	exp := make([]float64, vocab)
	for i, v := range logitsHost {
		exp[i] = math.Exp(float64(v - mx))
		se += exp[i]
	}

	gotGrad := m.ToHost(grad)
	for i := range logitsHost {
		want := exp[i] / se * invN
		if i == target {
			want -= invN
		}
		if d := math.Abs(float64(gotGrad[i]) - want); d > 1e-4 {
			t.Errorf("vocab %d: gradient %v, want %v", i, gotGrad[i], want)
		}
	}

	wantLoss := -math.Log(exp[target] / se)
	if d := math.Abs(float64(m.ToHost(losses)[0]) - wantLoss); d > 1e-4 {
		t.Errorf("loss %v, want %v", m.ToHost(losses)[0], wantLoss)
	}
}

// An all-masked batch must produce nothing at all rather than NaN. This is the
// degenerate case an SFT corpus hits if the mask boundary is computed wrongly,
// and NaN would poison the optimizer state for every subsequent step.
func TestSoftmaxCEGradAllMaskedProducesNoNaN(t *testing.T) {
	m := NewMetal()
	if m == nil {
		t.Skip("Metal unavailable")
	}

	const seqLen, vocab = 3, 5
	logits := m.FromHost(make([]float32, seqLen*vocab), []int{seqLen, vocab})
	defer m.Release(logits)
	targets := m.FromHost([]float32{
		targetBits(-1), targetBits(-1), targetBits(-1),
	}, []int{seqLen})
	defer m.Release(targets)
	losses := m.Zeros([]int{seqLen})
	defer m.Release(losses)
	grad := m.Zeros([]int{seqLen, vocab})
	defer m.Release(grad)

	m.SoftmaxCEGrad(logits, targets, losses, grad, seqLen, vocab, 1.0/float32(seqLen))

	for i, v := range m.ToHost(losses) {
		if math.IsNaN(float64(v)) || v != 0 {
			t.Errorf("loss[%d] = %v, want 0", i, v)
		}
	}
	for i, v := range m.ToHost(grad) {
		if math.IsNaN(float64(v)) || v != 0 {
			t.Errorf("grad[%d] = %v, want 0", i, v)
		}
	}
}
