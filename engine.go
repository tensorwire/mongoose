// Package compute provides the backend interface for tensor operations.
// Phase 1 implements a pure-Go CPU backend. Phase 3 adds Metal, Vulkan, CUDA.
package mongoose

// Engine abstracts tensor operations across hardware backends.
// Each platform implements this interface behind build tags.
type Engine interface {
	// Name returns the backend identifier ("cpu", "metal", "vulkan", "cuda").
	Name() string

	// MatMul computes C = A @ B where A is [M,K] and B is [K,N].
	// All slices are row-major.
	MatMul(a, b []float32, m, k, n int) []float32

	// RMSNorm computes RMSNorm(x, weight, eps) in-place.
	RMSNorm(x, weight []float32, eps float32)

	// SoftMax computes softmax over a vector of length n, in-place.
	SoftMax(x []float32, n int)

	// ReLU applies ReLU activation in-place.
	ReLU(x []float32)

	// VRAM returns available GPU memory in bytes (0 for CPU backend).
	VRAM() uint64

	// Benchmark runs a standard matmul workload and returns estimated GFLOPS.
	Benchmark() float64
}
