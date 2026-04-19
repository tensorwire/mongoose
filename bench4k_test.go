//go:build linux && cgo

package mongoose

import (
	"fmt"
	"testing"
)

func runBench(cuda *CUDA, label string, dims []int, setup func(dim int) func(), iters int) {
	fmt.Printf("\n=== %s ===\n", label)
	fmt.Printf("  %10s  %10s  %10s  %10s\n", "Size", "TFLOPS", "GPU us/op", "Wall us/op")
	fmt.Printf("  %10s  %10s  %10s  %10s\n", "----------", "----------", "----------", "----------")

	for _, dim := range dims {
		fn := setup(dim)

		// Warmup
		for i := 0; i < 10; i++ { fn() }
		cuda.Sync()

		n := iters
		if dim >= 4096 { n = iters / 2 }

		// GPU-side timing (same as torch.cuda.Event)
		timer := NewCUDATimer()
		timer.Start()
		for i := 0; i < n; i++ { fn() }
		gpuMs := timer.StopMs()
		timer.Destroy()

		gpuUsPerOp := float64(gpuMs) * 1000.0 / float64(n)
		gpuTflops := float64(2*dim*dim*dim*n) / (float64(gpuMs)/1000.0) / 1e12

		fmt.Printf("  %4dx%-5d  %10.2f  %10.1f\n", dim, dim, gpuTflops, gpuUsPerOp)
	}
}

func TestBench4096(t *testing.T) {
	cuda := NewCUDA()
	if cuda == nil { t.Skip("no CUDA") }

	dims := []int{512, 1024, 2048, 4096}

	// FP32 TF32 with alloc per iter
	runBench(cuda, "FP32 TF32 (alloc per iter)", dims, func(dim int) func() {
		a := make([]float32, dim*dim)
		b := make([]float32, dim*dim)
		for i := range a { a[i] = 0.001 * float32(i%1000); b[i] = 0.001 * float32(i%997) }
		tA := cuda.FromHost(a, []int{dim, dim})
		tB := cuda.FromHost(b, []int{dim, dim})
		return func() {
			c := cuda.MatMulT(tA, tB, dim, dim, dim)
			cuda.Release(c)
		}
	}, 200)

	// FP16 GemmEx with alloc per iter
	runBench(cuda, "FP16 cublasGemmEx FP32-out (alloc per iter)", dims, func(dim int) func() {
		a := make([]float32, dim*dim)
		b := make([]float32, dim*dim)
		for i := range a { a[i] = 0.001 * float32(i%1000); b[i] = 0.001 * float32(i%997) }
		tA := cuda.FromHostFP16(a, []int{dim, dim})
		tB := cuda.FromHostFP16(b, []int{dim, dim})
		return func() {
			c := cuda.MatMulFP16T(tA, tB, dim, dim, dim)
			cuda.Release(c)
		}
	}, 200)

	// FP16 cublasLt with alloc per iter
	runBench(cuda, "FP16 cublasLtMatmul FP16-out (alloc per iter)", dims, func(dim int) func() {
		a := make([]float32, dim*dim)
		b := make([]float32, dim*dim)
		for i := range a { a[i] = 0.001 * float32(i%1000); b[i] = 0.001 * float32(i%997) }
		tA := cuda.FromHostFP16(a, []int{dim, dim})
		tB := cuda.FromHostFP16(b, []int{dim, dim})
		return func() {
			c := cuda.MatMulFP16(tA, tB, dim, dim, dim)
			cuda.Release(c)
		}
	}, 200)

	// FP16 GemmEx NO alloc (reuse output buffer)
	runBench(cuda, "FP16 cublasGemmEx FP32-out (NO alloc, reuse buf)", dims, func(dim int) func() {
		a := make([]float32, dim*dim)
		b := make([]float32, dim*dim)
		for i := range a { a[i] = 0.001 * float32(i%1000); b[i] = 0.001 * float32(i%997) }
		tA := cuda.FromHostFP16(a, []int{dim, dim})
		tB := cuda.FromHostFP16(b, []int{dim, dim})
		tC := cuda.MatMulFP16T(tA, tB, dim, dim, dim)
		return func() {
			cuda.MatMulFP16TInto(tA, tB, tC, dim, dim, dim)
		}
	}, 200)
}
