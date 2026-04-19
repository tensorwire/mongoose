//go:build linux && cgo

package mongoose

import (
	"fmt"
	"testing"
	"time"
)

func TestGPUResidentVsHostCopy(t *testing.T) {
	cuda := NewCUDA()
	if cuda == nil {
		t.Skip("no CUDA")
	}

	dim := 1024
	a := make([]float32, dim*dim)
	b := make([]float32, dim*dim)
	for i := range a {
		a[i] = 0.001 * float32(i%1000)
		b[i] = 0.001 * float32(i%997)
	}

	// Host-copy
	cuda.MatMul(a, b, dim, dim, dim)
	t0 := time.Now()
	for i := 0; i < 100; i++ {
		cuda.MatMul(a, b, dim, dim, dim)
	}
	hostTime := time.Since(t0)
	hostGflops := float64(2*dim*dim*dim*100) / hostTime.Seconds() / 1e9

	// GPU-resident
	tA := cuda.FromHost(a, []int{dim, dim})
	tB := cuda.FromHost(b, []int{dim, dim})
	tC := cuda.MatMulT(tA, tB, dim, dim, dim)
	cuda.Release(tC)

	t1 := time.Now()
	for i := 0; i < 100; i++ {
		tC = cuda.MatMulT(tA, tB, dim, dim, dim)
		cuda.Release(tC)
	}
	gpuTime := time.Since(t1)
	gpuGflops := float64(2*dim*dim*dim*100) / gpuTime.Seconds() / 1e9

	cuda.Release(tA)
	cuda.Release(tB)

	fmt.Printf("Host-copy:    %v (%.1f GFLOPS)\n", hostTime, hostGflops)
	fmt.Printf("GPU-resident: %v (%.1f GFLOPS)\n", gpuTime, gpuGflops)
	fmt.Printf("Speedup:      %.1fx\n", gpuGflops/hostGflops)

	if gpuGflops <= hostGflops {
		t.Errorf("GPU-resident should be faster than host-copy")
	}
}
