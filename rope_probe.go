
//go:build darwin && cgo

package mongoose

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Foundation -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph

int mtl_graph_rope_probe(float* x, int n, int nHeads, int headDim, float theta);
int mtl_graph_rope1_probe(float* x, int nHeads, int headDim, int pos, float theta);
int mtl_infer_rope_probe(float* x, int nHeads, int headDim, int pos, float theta);
*/
import "C"
import _ "unsafe"

import (
	"fmt"
	"unsafe"
)

// RoPE probes — direct, single-purpose entry points into each of mongoose's RoPE
// implementations.
//
// These exist for correctness testing, not for the hot path. Each RoPE site is
// otherwise only reachable through a full graph build or a full model build,
// where a wrong rotation convention surfaces as nothing more than a slightly
// worse scalar loss. That opacity is exactly how the interleaved-vs-split-half
// mismatch between the training and inference paths survived: every path agreed
// with itself, nothing compared them to each other or to a reference.
//
// All three probes operate in place and use the split-half (NeoX/HF) convention
// that Granite and Llama-HF require: pairs (x[i], x[i + headDim/2]).
//
// cgo is not permitted in _test.go files, so the bridge lives here and
// rope_convention_test.go calls these.

// RoPEProbeTrainingGraph applies the MPSGraph TRAINING RoPE (buildRoPE in
// metal_graph_darwin.m) to x in place.
//
// x is [n, nHeads*headDim] row-major; row index is the position. headDim must be
// even and >= 2.
func RoPEProbeTrainingGraph(x []float32, n, nHeads, headDim int, theta float32) error {
	if err := ropeProbeCheck(x, n, nHeads, headDim); err != nil {
		return err
	}
	rc := (C.mtl_graph_rope_probe)((*C.float)(unsafe.Pointer(&x[0])),
		C.int(n), C.int(nHeads), C.int(headDim), C.float(theta))
	if rc != 0 {
		return fmt.Errorf("rope probe (training graph): failed (%d)", int(rc))
	}
	return nil
}

// RoPEProbeInferenceGraph applies the MPSGraph single-token INFERENCE RoPE
// (buildRoPE1 in metal_graph_darwin.m) to x in place at the given position.
//
// x is [nHeads*headDim].
func RoPEProbeInferenceGraph(x []float32, nHeads, headDim, pos int, theta float32) error {
	if err := ropeProbeCheck(x, 1, nHeads, headDim); err != nil {
		return err
	}
	rc := (C.mtl_graph_rope1_probe)((*C.float)(unsafe.Pointer(&x[0])),
		C.int(nHeads), C.int(headDim), C.int(pos), C.float(theta))
	if rc != 0 {
		return fmt.Errorf("rope probe (inference graph): failed (%d)", int(rc))
	}
	return nil
}

// RoPEProbeInferenceKernel applies the INFERENCE Metal kernel (rope_rotate_half)
// to x in place at the given position.
//
// x is [nHeads*headDim]. This is the kernel that actually serves tokens, so it
// is the one training must agree with — changing it instead of training would
// invalidate every checkpoint already served correctly.
func RoPEProbeInferenceKernel(x []float32, nHeads, headDim, pos int, theta float32) error {
	if err := ropeProbeCheck(x, 1, nHeads, headDim); err != nil {
		return err
	}
	rc := (C.mtl_infer_rope_probe)((*C.float)(unsafe.Pointer(&x[0])),
		C.int(nHeads), C.int(headDim), C.int(pos), C.float(theta))
	if rc != 0 {
		return fmt.Errorf("rope probe (inference kernel): failed (%d)", int(rc))
	}
	return nil
}

func ropeProbeCheck(x []float32, n, nHeads, headDim int) error {
	if headDim < 2 || headDim%2 != 0 {
		return fmt.Errorf("rope probe: headDim must be even and >= 2, got %d", headDim)
	}
	if n <= 0 || nHeads <= 0 {
		return fmt.Errorf("rope probe: n=%d nHeads=%d must be positive", n, nHeads)
	}
	if want := n * nHeads * headDim; len(x) != want {
		return fmt.Errorf("rope probe: x has %d elements, want n*nHeads*headDim = %d",
			len(x), want)
	}
	return nil
}
