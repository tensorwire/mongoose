package mongoose

import (
	"math"
	"runtime"
	"time"
)

// CPU implements Engine using pure Go float32 operations.
// No CGo, no external dependencies. The Go compiler auto-vectorizes
// the inner loops on arm64 (NEON) and amd64 (SSE/AVX).
type CPU struct{}

func (c *CPU) Name() string { return "cpu" }

// MatMul computes C = A @ B.
// A is [m, k], B is [k, n], result C is [m, n]. All row-major.
func (c *CPU) MatMul(a, b []float32, m, k, n int) []float32 {
	out := make([]float32, m*n)
	for i := 0; i < m; i++ {
		aOff := i * k
		oOff := i * n
		for j := 0; j < n; j++ {
			var sum float32
			for l := 0; l < k; l++ {
				sum += a[aOff+l] * b[l*n+j]
			}
			out[oOff+j] = sum
		}
	}
	return out
}

// RMSNorm computes: x[i] = x[i] / rms * weight[i]
// where rms = sqrt(mean(x^2) + eps)
func (c *CPU) RMSNorm(x, weight []float32, eps float32) {
	n := len(x)
	var ss float32
	for i := 0; i < n; i++ {
		ss += x[i] * x[i]
	}
	ss = ss/float32(n) + eps
	ss = float32(1.0 / math.Sqrt(float64(ss)))
	for i := 0; i < n; i++ {
		x[i] = x[i] * ss * weight[i]
	}
}

// SoftMax computes softmax over x[0:n] in-place.
func (c *CPU) SoftMax(x []float32, n int) {
	// Find max for numerical stability
	maxVal := x[0]
	for i := 1; i < n; i++ {
		if x[i] > maxVal {
			maxVal = x[i]
		}
	}
	// exp and sum
	var sum float32
	for i := 0; i < n; i++ {
		x[i] = float32(math.Exp(float64(x[i] - maxVal)))
		sum += x[i]
	}
	// normalize
	inv := float32(1.0) / sum
	for i := 0; i < n; i++ {
		x[i] *= inv
	}
}

// ReLU applies max(0, x) in-place.
func (c *CPU) ReLU(x []float32) {
	for i := range x {
		if x[i] < 0 {
			x[i] = 0
		}
	}
}

// VRAM returns 0 — CPU backend uses system RAM, not GPU memory.
func (c *CPU) VRAM() uint64 { return 0 }

// Benchmark runs a 512x512 matmul and reports estimated GFLOPS.
func (c *CPU) Benchmark() float64 {
	const dim = 512
	a := make([]float32, dim*dim)
	b := make([]float32, dim*dim)
	for i := range a {
		a[i] = 0.001 * float32(i%1000)
		b[i] = 0.001 * float32(i%997)
	}

	runtime.GC()
	start := time.Now()
	iterations := 10
	for iter := 0; iter < iterations; iter++ {
		c.MatMul(a, b, dim, dim, dim)
	}
	elapsed := time.Since(start)

	// FLOPS = 2 * M * N * K per matmul (multiply + add)
	flops := float64(2*dim*dim*dim*iterations) / elapsed.Seconds()
	return flops / 1e9 // GFLOPS
}
