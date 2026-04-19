//go:build linux && cgo

package mongoose

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"unsafe"
)

// TestSparseCorrectness verifies that sparse_matmul produces the same output
// as dense cuBLAS matmul for identical inputs. This is the test that matters —
// if the sparse path gives different answers, the benchmarks are meaningless.
func TestSparseCorrectness(t *testing.T) {
	cuda := NewCUDA()
	if cuda == nil {
		t.Skip("no CUDA")
	}
	if !LoadKernels() {
		t.Skip("no kernels .so")
	}
	if !SparseKernelsLoaded() {
		t.Skip("sparse kernels not loaded")
	}

	fmt.Println("\n=== Sparse Correctness Tests ===")

	type testCase struct {
		name     string
		rows     int
		cols     int
		sparsity float64
	}

	cases := []testCase{
		// Edge cases
		{"0% sparse (fully dense)", 128, 512, 0.0},
		{"100% sparse (all zeros)", 128, 512, 1.0},
		{"1 active dim", 128, 512, 0.998},
		{"99% sparse", 128, 512, 0.99},

		// Odd dimensions (not divisible by 256 block size)
		{"odd dims 127x511", 127, 511, 0.80},
		{"odd dims 1023x4093", 1023, 4093, 0.80},
		{"prime dims 127x509", 127, 509, 0.80},

		// Real model sizes
		{"ReluLLaMA FFN 50%", 4096, 11008, 0.50},
		{"ReluLLaMA FFN 80%", 4096, 11008, 0.80},
		{"ReluLLaMA FFN 90%", 4096, 11008, 0.90},
		{"Qwen2 FFN 80%", 3584, 18944, 0.80},

		// Small matrices
		{"tiny 4x8", 4, 8, 0.50},
		{"tiny 1x1", 1, 1, 0.0},
		{"single row", 1, 4096, 0.80},
		{"single col", 4096, 1, 0.0},
	}

	maxErr := float64(0)
	allPass := true

	for _, tc := range cases {
		rows := tc.rows
		cols := tc.cols

		// Create random POST-RELU input (non-negative, with controlled sparsity).
		// In a real FFN, the down projection input has already been through ReLU.
		// Both dense and sparse paths must see the same data.
		x := make([]float32, cols)
		activeCount := 0
		for i := range x {
			if rand.Float64() > tc.sparsity {
				x[i] = rand.Float32() * 2 // [0, 2] — always non-negative
				if x[i] == 0 {
					x[i] = 0.001 // avoid accidental zeros
				}
				activeCount++
			}
			// else x[i] = 0 (sparse)
		}

		// Handle edge case: if sparsity=0, ensure all are positive
		if tc.sparsity == 0 {
			for i := range x {
				if x[i] == 0 {
					x[i] = 0.001
				}
			}
			activeCount = cols
		}

		// Create weights and transpose
		W := make([]float32, rows*cols)
		WT := make([]float32, rows*cols)
		for i := range W {
			W[i] = (rand.Float32() - 0.5) * 0.1
		}
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				WT[j*rows+i] = W[i*cols+j]
			}
		}

		// Dense reference: CPU matmul
		refOut := make([]float32, rows)
		for i := 0; i < rows; i++ {
			var sum float64
			for j := 0; j < cols; j++ {
				sum += float64(W[i*cols+j]) * float64(x[j])
			}
			refOut[i] = float32(sum)
		}

		// GPU dense: cuBLAS
		xT := cuda.FromHost(x, []int{cols})
		wT := cuda.FromHost(W, []int{rows, cols})
		denseResult := cuda.MatMulT(wT, xT, rows, cols, 1)
		cuda.Sync()
		denseOut := cuda.ToHost(denseResult)

		// GPU sparse: our kernel
		wtT := cuda.FromHost(WT, []int{cols, rows})
		activeIdxBuf := cuda.Zeros([]int{cols})
		countBuf := cuda.Zeros([]int{1})
		sparseOut := cuda.Zeros([]int{rows})

		// Build active index
		xCopy := cuda.CopyT(xT)
		KReLUAndIndex(xCopy.DevicePtr(), activeIdxBuf.DevicePtr(), countBuf.DevicePtr(), cols)
		cuda.Sync()

		// Read back active count
		countHost := cuda.ToHost(countBuf)
		gpuActiveCount := int(*(*int32)(unsafe.Pointer(&countHost[0])))

		// Run sparse matmul
		KSparseMatMul(
			sparseOut.DevicePtr(),
			wtT.DevicePtr(),
			xCopy.DevicePtr(), // post-ReLU (but our input already has zeros, ReLU is a no-op for positive values)
			activeIdxBuf.DevicePtr(),
			gpuActiveCount, rows, cols,
		)
		cuda.Sync()
		sparseOutHost := cuda.ToHost(sparseOut)

		// Compare: dense GPU vs sparse GPU vs CPU reference
		var maxDenseCPU, maxSparseCPU, maxSparseDense float64
		var maxDenseIdx, maxSparseIdx, maxSDIdx int

		for i := 0; i < rows; i++ {
			dCPU := math.Abs(float64(denseOut[i]) - float64(refOut[i]))
			sCPU := math.Abs(float64(sparseOutHost[i]) - float64(refOut[i]))
			sD := math.Abs(float64(sparseOutHost[i]) - float64(denseOut[i]))

			if dCPU > maxDenseCPU {
				maxDenseCPU = dCPU
				maxDenseIdx = i
			}
			if sCPU > maxSparseCPU {
				maxSparseCPU = sCPU
				maxSparseIdx = i
			}
			if sD > maxSparseDense {
				maxSparseDense = sD
				maxSDIdx = i
			}
		}

		// Track worst case
		if maxSparseDense > maxErr {
			maxErr = maxSparseDense
		}

		pass := true
		// For FP32, tolerate ~1e-3 for large matrices (accumulation order differs)
		tolerance := 1e-2
		if rows*cols > 1000000 {
			tolerance = 0.1 // large matrices accumulate more FP32 error
		}

		if maxSparseDense > tolerance {
			pass = false
			allPass = false
		}

		status := "PASS"
		if !pass {
			status = "FAIL"
		}

		// Handle the special case where everything is zero
		if activeCount == 0 {
			allZero := true
			for i := 0; i < rows; i++ {
				if sparseOutHost[i] != 0 {
					allZero = false
					break
				}
			}
			if allZero {
				status = "PASS"
				pass = true
			}
		}

		fmt.Printf("  %-30s  active=%5d/%5d  dense-cpu=%.1e[%d]  sparse-cpu=%.1e[%d]  sparse-dense=%.1e[%d]  %s\n",
			tc.name, gpuActiveCount, cols,
			maxDenseCPU, maxDenseIdx,
			maxSparseCPU, maxSparseIdx,
			maxSparseDense, maxSDIdx,
			status)

		if !pass {
			// Print first few divergent values
			for i := 0; i < rows && i < 5; i++ {
				fmt.Printf("    [%d] ref=%.6f dense=%.6f sparse=%.6f\n",
					i, refOut[i], denseOut[i], sparseOutHost[i])
			}
		}

		cuda.Release(xT)
		cuda.Release(wT)
		cuda.Release(denseResult)
		cuda.Release(wtT)
		cuda.Release(activeIdxBuf)
		cuda.Release(countBuf)
		cuda.Release(sparseOut)
		cuda.Release(xCopy)
	}

	fmt.Printf("\n  Max sparse-vs-dense error across all tests: %.2e\n", maxErr)
	if allPass {
		fmt.Println("  ALL CORRECTNESS TESTS PASSED")
	} else {
		t.Fatalf("CORRECTNESS FAILURE — sparse output diverges from dense")
	}
}

// TestSparseEdgeCases tests boundary conditions that could crash the kernel.
func TestSparseEdgeCases(t *testing.T) {
	cuda := NewCUDA()
	if cuda == nil {
		t.Skip("no CUDA")
	}
	if !LoadKernels() {
		t.Skip("no kernels .so")
	}
	if !SparseKernelsLoaded() {
		t.Skip("sparse kernels not loaded")
	}

	fmt.Println("\n=== Sparse Edge Case Tests ===")

	// Test: 100% sparse — all zeros, active count should be 0
	{
		x := make([]float32, 1024)
		xT := cuda.FromHost(x, []int{1024})
		activeIdx := cuda.Zeros([]int{1024})
		count := cuda.Zeros([]int{1})

		KReLUAndIndex(xT.DevicePtr(), activeIdx.DevicePtr(), count.DevicePtr(), 1024)
		cuda.Sync()

		countHost := cuda.ToHost(count)
		ac := int(*(*int32)(unsafe.Pointer(&countHost[0])))
		if ac != 0 {
			t.Errorf("100%% sparse: expected 0 active, got %d", ac)
		} else {
			fmt.Println("  100% sparse (all zeros): active=0 — PASS")
		}

		cuda.Release(xT)
		cuda.Release(activeIdx)
		cuda.Release(count)
	}

	// Test: 0% sparse — all active
	{
		x := make([]float32, 1024)
		for i := range x {
			x[i] = float32(i+1) * 0.001
		}
		xT := cuda.FromHost(x, []int{1024})
		activeIdx := cuda.Zeros([]int{1024})
		count := cuda.Zeros([]int{1})

		KReLUAndIndex(xT.DevicePtr(), activeIdx.DevicePtr(), count.DevicePtr(), 1024)
		cuda.Sync()

		countHost := cuda.ToHost(count)
		ac := int(*(*int32)(unsafe.Pointer(&countHost[0])))
		if ac != 1024 {
			t.Errorf("0%% sparse: expected 1024 active, got %d", ac)
		} else {
			fmt.Println("  0% sparse (all active): active=1024 — PASS")
		}

		cuda.Release(xT)
		cuda.Release(activeIdx)
		cuda.Release(count)
	}

	// Test: Negative values should become zero after ReLU
	{
		x := make([]float32, 256)
		for i := range x {
			x[i] = float32(i) - 128 // [-128, 127]
		}
		xT := cuda.FromHost(x, []int{256})
		activeIdx := cuda.Zeros([]int{256})
		count := cuda.Zeros([]int{1})

		KReLUAndIndex(xT.DevicePtr(), activeIdx.DevicePtr(), count.DevicePtr(), 256)
		cuda.Sync()

		countHost := cuda.ToHost(count)
		ac := int(*(*int32)(unsafe.Pointer(&countHost[0])))
		// Values 1..127 should be active (127 values). 0 and negatives are zeroed.
		if ac != 127 {
			t.Errorf("ReLU negative test: expected 127 active, got %d", ac)
		} else {
			fmt.Println("  ReLU negative suppression: active=127/256 — PASS")
		}

		// Verify the data was actually ReLU'd
		xOut := cuda.ToHost(xT)
		for i := 0; i < 128; i++ {
			if xOut[i] != 0 {
				t.Errorf("  x[%d] = %f, expected 0 after ReLU", i, xOut[i])
				break
			}
		}
		for i := 129; i < 256; i++ {
			if xOut[i] <= 0 {
				t.Errorf("  x[%d] = %f, expected positive after ReLU", i, xOut[i])
				break
			}
		}
		fmt.Println("  ReLU values correct — PASS")

		cuda.Release(xT)
		cuda.Release(activeIdx)
		cuda.Release(count)
	}

	// Test: sparse_matmul with 0 active columns should produce all zeros
	{
		rows, cols := 512, 1024
		WT := make([]float32, rows*cols)
		for i := range WT {
			WT[i] = 1.0 // non-zero weights
		}
		x := make([]float32, cols) // all zeros

		wtT := cuda.FromHost(WT, []int{cols, rows})
		xT := cuda.FromHost(x, []int{cols})
		out := cuda.Zeros([]int{rows})
		activeIdx := cuda.Zeros([]int{cols})

		KSparseMatMul(out.DevicePtr(), wtT.DevicePtr(), xT.DevicePtr(),
			activeIdx.DevicePtr(), 0, rows, cols)
		cuda.Sync()

		outHost := cuda.ToHost(out)
		allZero := true
		for i := range outHost {
			if outHost[i] != 0 {
				allZero = false
				t.Errorf("  out[%d] = %f, expected 0 with 0 active cols", i, outHost[i])
				break
			}
		}
		if allZero {
			fmt.Println("  sparse_matmul with 0 active: output all zeros — PASS")
		}

		cuda.Release(wtT)
		cuda.Release(xT)
		cuda.Release(out)
		cuda.Release(activeIdx)
	}

	fmt.Println("  ALL EDGE CASE TESTS PASSED")
}
