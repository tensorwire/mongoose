//go:build darwin && cgo

package mongoose

import (
	"math"
	"testing"
)

// RoPE has two incompatible conventions in the wild:
//
//	split-half (NeoX / Llama-HF / Granite):  pairs (x[i], x[i+headDim/2])
//	interleaved (GPT-J):                     pairs (x[2i], x[2i+1])
//
// mongoose has three RoPE implementations — the MPSGraph training graph, the
// MPSGraph single-token inference graph, and the inference Metal kernel. Each
// was internally consistent, so nothing failed; the training graph was on the
// interleaved convention while the two inference paths were split-half. The
// only symptom was a slightly worse loss curve and subtly wrong served
// checkpoints.
//
// These tests pin all three to split-half and to each other.

const ropeTheta = 10000.0

// ropeSplitHalfRef is the reference implementation, transcribed from
// transformers' GraniteForCausalLM.rotate_half.
func ropeSplitHalfRef(x []float32, nHeads, headDim, pos int, theta float64) []float32 {
	out := make([]float32, len(x))
	copy(out, x)
	half := headDim / 2
	for h := 0; h < nHeads; h++ {
		base := h * headDim
		for j := 0; j < half; j++ {
			freq := 1.0 / math.Pow(theta, float64(2*j)/float64(headDim))
			angle := float64(pos) * freq
			c, s := math.Cos(angle), math.Sin(angle)
			x0 := float64(x[base+j])
			x1 := float64(x[base+j+half])
			out[base+j] = float32(x0*c - x1*s)
			out[base+j+half] = float32(x0*s + x1*c)
		}
	}
	return out
}

func ropeTestInput(n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		// Deterministic, non-degenerate, and spread across magnitudes so a
		// mispairing cannot coincidentally agree.
		x[i] = float32(math.Sin(float64(i)*0.7)) * float32(1+i%5)
	}
	return x
}

func maxAbsDiff(a, b []float32) float64 {
	m := 0.0
	for i := range a {
		d := math.Abs(float64(a[i]) - float64(b[i]))
		if d > m {
			m = d
		}
	}
	return m
}

// The inference kernel is the reference: it already serves correct checkpoints,
// so the other paths must match IT, not the other way round.
func TestRoPEInferenceKernelIsSplitHalf(t *testing.T) {
	const nHeads, headDim, pos = 4, 16, 7
	x := ropeTestInput(nHeads * headDim)
	want := ropeSplitHalfRef(x, nHeads, headDim, pos, ropeTheta)

	got := make([]float32, len(x))
	copy(got, x)
	if err := RoPEProbeInferenceKernel(got, nHeads, headDim, pos, ropeTheta); err != nil {
		t.Skipf("Metal unavailable: %v", err)
	}
	if d := maxAbsDiff(got, want); d > 1e-4 {
		t.Errorf("inference kernel is not split-half: max abs diff %.3e", d)
	}
}

func TestRoPEInferenceGraphIsSplitHalf(t *testing.T) {
	const nHeads, headDim, pos = 4, 16, 7
	x := ropeTestInput(nHeads * headDim)
	want := ropeSplitHalfRef(x, nHeads, headDim, pos, ropeTheta)

	got := make([]float32, len(x))
	copy(got, x)
	if err := RoPEProbeInferenceGraph(got, nHeads, headDim, pos, ropeTheta); err != nil {
		t.Skipf("Metal unavailable: %v", err)
	}
	if d := maxAbsDiff(got, want); d > 1e-4 {
		t.Errorf("inference graph is not split-half: max abs diff %.3e", d)
	}
}

// The regression that mattered: the training graph used interleaved pairs while
// everything else used split-half, so a model trained here and served through
// the inference kernel had a different rotation applied.
func TestRoPETrainingGraphIsSplitHalf(t *testing.T) {
	const nHeads, headDim, n = 4, 16, 3
	x := ropeTestInput(n * nHeads * headDim)

	// buildRoPE rotates row p by position p.
	want := make([]float32, len(x))
	rowLen := nHeads * headDim
	for p := 0; p < n; p++ {
		row := ropeSplitHalfRef(x[p*rowLen:(p+1)*rowLen], nHeads, headDim, p, ropeTheta)
		copy(want[p*rowLen:], row)
	}

	got := make([]float32, len(x))
	copy(got, x)
	if err := RoPEProbeTrainingGraph(got, n, nHeads, headDim, ropeTheta); err != nil {
		t.Skipf("Metal unavailable: %v", err)
	}
	if d := maxAbsDiff(got, want); d > 1e-4 {
		t.Errorf("training graph is not split-half: max abs diff %.3e\n"+
			"this is the training/inference mismatch — a model trained with this "+
			"rotation is served with a different one", d)
	}
}

// Even with both matching a reference, assert them against each other directly:
// this is the invariant that actually matters, and it fails loudly if someone
// "fixes" the reference to match a broken path.
func TestRoPETrainingAndInferenceAgree(t *testing.T) {
	const nHeads, headDim, pos = 4, 16, 2
	rowLen := nHeads * headDim
	x := ropeTestInput((pos + 1) * rowLen)

	train := make([]float32, len(x))
	copy(train, x)
	if err := RoPEProbeTrainingGraph(train, pos+1, nHeads, headDim, ropeTheta); err != nil {
		t.Skipf("Metal unavailable: %v", err)
	}

	// Compare the last row, which the training graph rotates by `pos`.
	infer := make([]float32, rowLen)
	copy(infer, x[pos*rowLen:])
	if err := RoPEProbeInferenceKernel(infer, nHeads, headDim, pos, ropeTheta); err != nil {
		t.Skipf("Metal unavailable: %v", err)
	}

	if d := maxAbsDiff(train[pos*rowLen:], infer); d > 1e-4 {
		t.Errorf("training and inference RoPE disagree: max abs diff %.3e", d)
	}
}

// A guard on the reference itself: split-half and interleaved must actually
// differ for these inputs, or the tests above would pass under either
// convention and prove nothing.
func TestRoPEConventionsAreDistinguishable(t *testing.T) {
	const nHeads, headDim, pos = 4, 16, 7
	x := ropeTestInput(nHeads * headDim)
	split := ropeSplitHalfRef(x, nHeads, headDim, pos, ropeTheta)

	// Interleaved reference, for contrast only.
	inter := make([]float32, len(x))
	copy(inter, x)
	half := headDim / 2
	for h := 0; h < nHeads; h++ {
		base := h * headDim
		for j := 0; j < half; j++ {
			freq := 1.0 / math.Pow(ropeTheta, float64(2*j)/float64(headDim))
			angle := float64(pos) * freq
			c, s := math.Cos(angle), math.Sin(angle)
			x0 := float64(x[base+2*j])
			x1 := float64(x[base+2*j+1])
			inter[base+2*j] = float32(x0*c - x1*s)
			inter[base+2*j+1] = float32(x0*s + x1*c)
		}
	}
	if d := maxAbsDiff(split, inter); d < 1e-3 {
		t.Fatalf("the two conventions are indistinguishable for this input "+
			"(diff %.3e); the convention tests would be vacuous", d)
	}
}
