//go:build darwin && cgo

package mongoose

import (
	"math"
	"testing"
)

// ScaleInPlace carries all four of Granite's architecture multipliers into the
// training forward pass. If it silently did nothing, the trainer would look
// fixed — the source-level tests would pass, the loss curve would look
// plausible — while still optimizing the wrong function.
//
// That is not hypothetical: FusedGemmF32BT on the LoRA critical path returned
// all zeros for months without any test noticing, because nothing checked its
// arithmetic.
func TestScaleInPlaceActuallyScales(t *testing.T) {
	m := NewMetal()
	if m == nil {
		t.Skip("Metal unavailable")
	}

	const n = 1024
	host := make([]float32, n)
	for i := range host {
		host[i] = float32(i%17) - 8 // spans negative, zero, positive
	}
	tn := m.FromHost(host, []int{n})
	defer m.Release(tn)

	const s = 0.22 // Granite-4.1-3b's residual_multiplier
	m.ScaleInPlace(tn, s, n)

	got := m.ToHost(tn)
	for i := range host {
		want := host[i] * s
		if d := math.Abs(float64(got[i] - want)); d > 1e-5 {
			t.Fatalf("element %d: got %v, want %v (input %v x %v)",
				i, got[i], want, host[i], s)
		}
	}
}

// The reciprocal path: logits_scaling DIVIDES, and the trainer applies it as
// 1/scaling. Granite ships 10.0, so a multiply instead of a divide is a 100x
// error per logit — which does not crash, it just trains against a
// differently-shaped distribution than inference samples from.
func TestScaleInPlaceReciprocalMatchesDivision(t *testing.T) {
	m := NewMetal()
	if m == nil {
		t.Skip("Metal unavailable")
	}

	const n = 256
	const logitsScaling = 10.0

	host := make([]float32, n)
	for i := range host {
		host[i] = float32(i) * 0.37
	}
	tn := m.FromHost(host, []int{n})
	defer m.Release(tn)

	m.ScaleInPlace(tn, 1.0/logitsScaling, n)

	got := m.ToHost(tn)
	for i := range host {
		want := host[i] / logitsScaling
		if d := math.Abs(float64(got[i] - want)); d > 1e-5 {
			t.Fatalf("element %d: got %v, want %v", i, got[i], want)
		}
	}
}

// A scale of exactly 1.0 must be a no-op. The trainer guards on != 0 rather
// than != 1, so an identity multiplier still reaches the kernel and must not
// perturb the values.
func TestScaleInPlaceIdentityIsExact(t *testing.T) {
	m := NewMetal()
	if m == nil {
		t.Skip("Metal unavailable")
	}

	const n = 128
	host := make([]float32, n)
	for i := range host {
		host[i] = float32(i) * 1.7
	}
	tn := m.FromHost(host, []int{n})
	defer m.Release(tn)

	m.ScaleInPlace(tn, 1.0, n)

	got := m.ToHost(tn)
	for i := range host {
		if got[i] != host[i] {
			t.Fatalf("element %d changed under an identity scale: %v -> %v",
				i, host[i], got[i])
		}
	}
}
