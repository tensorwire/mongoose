//go:build linux && cgo

package mongoose

import (
	"fmt"
	"math/rand"
	"testing"
	"unsafe"
)

func TestSparseBench(t *testing.T) {
	cuda := NewCUDA()
	if cuda == nil {
		t.Skip("no CUDA")
	}
	if !LoadKernels() {
		t.Skip("no kernels .so")
	}
	if !SparseKernelsLoaded() {
		t.Skip("sparse kernels not in .so (recompile)")
	}

	fmt.Println("\n=== Sparse FFN GPU Benchmark ===")

	// Test at multiple sizes and sparsity levels
	type testCase struct {
		rows     int     // output dim (HiddenDim)
		cols     int     // FFN dim
		sparsity float64 // fraction of zeros in input
	}

	cases := []testCase{
		{128, 512, 0.50},    // small model, 50% sparse
		{128, 512, 0.80},    // small model, 80% sparse
		{1024, 4096, 0.50},  // medium model, 50% sparse
		{1024, 4096, 0.80},  // medium model, 80% sparse
		{1024, 4096, 0.90},  // medium model, 90% sparse
		{3584, 18944, 0.50}, // Qwen2-7B size, 50% sparse
		{3584, 18944, 0.80}, // Qwen2-7B size, 80% sparse
		{3584, 18944, 0.90}, // Qwen2-7B size, 90% sparse
	}

	fmt.Printf("%-20s  %8s  %8s  %8s  %8s\n",
		"Config", "Dense µs", "Sparse µs", "Speedup", "FLOP save")
	fmt.Printf("%-20s  %8s  %8s  %8s  %8s\n",
		"--------------------", "--------", "---------", "--------", "---------")

	for _, tc := range cases {
		rows := tc.rows
		cols := tc.cols

		// Create random input with controlled sparsity
		x := make([]float32, cols)
		for i := range x {
			if rand.Float64() > tc.sparsity {
				x[i] = rand.Float32()*2 - 1
			}
			// else x[i] = 0 (sparse)
		}

		// Create weight matrix (rows x cols) and its transpose
		W := make([]float32, rows*cols)
		WT := make([]float32, rows*cols)
		for i := range W {
			W[i] = (rand.Float32() - 0.5) * 0.1
		}
		// Transpose: WT[j*rows+i] = W[i*cols+j]
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				WT[j*rows+i] = W[i*cols+j]
			}
		}

		// Upload to GPU
		xT := cuda.FromHost(x, []int{cols})
		wT := cuda.FromHost(W, []int{rows, cols})
		wtT := cuda.FromHost(WT, []int{cols, rows})
		outDense := cuda.Zeros([]int{rows})
		outSparse := cuda.Zeros([]int{rows})

		// Allocate GPU buffers for sparse index
		activeIdxBuf := cuda.Zeros([]int{cols}) // int32 same size as float32
		countBuf := cuda.Zeros([]int{1})

		// Warm up
		for i := 0; i < 5; i++ {
			cuda.MatMulT(wT, xT, rows, cols, 1)
		}
		cuda.Sync()

		// Dense benchmark: cuBLAS matmul
		iters := 50
		timer := NewCUDATimer()
		timer.Start()
		for i := 0; i < iters; i++ {
			r := cuda.MatMulT(wT, xT, rows, cols, 1)
			cuda.Release(r)
		}
		denseMs := timer.StopMs()
		timer.Destroy()
		denseUs := float64(denseMs) * 1000 / float64(iters)

		// Sparse benchmark: relu_and_index + sparse_matmul
		// First, get the active count by running relu_and_index once on a copy
		xCopy := cuda.CopyT(xT)
		KReLUAndIndex(xCopy.DevicePtr(), activeIdxBuf.DevicePtr(), countBuf.DevicePtr(), cols)
		cuda.Sync()
		countHost := cuda.ToHost(countBuf)
		activeCount := int(*(*int32)(unsafe.Pointer(&countHost[0])))
		cuda.Release(xCopy)

		// Now benchmark the sparse path
		timer2 := NewCUDATimer()
		timer2.Start()
		for i := 0; i < iters; i++ {
			// In a real pipeline, relu_and_index happens once before the matmul
			// For the benchmark, we just time the sparse matmul with the pre-computed index
			KSparseMatMul(
				outSparse.DevicePtr(),
				wtT.DevicePtr(),
				xT.DevicePtr(),
				activeIdxBuf.DevicePtr(),
				activeCount, rows, cols,
			)
		}
		sparseMs := timer2.StopMs()
		timer2.Destroy()
		sparseUs := float64(sparseMs) * 1000 / float64(iters)

		speedup := denseUs / sparseUs
		flopSave := tc.sparsity * 100

		label := fmt.Sprintf("%dx%d @%.0f%%", rows, cols, tc.sparsity*100)
		fmt.Printf("%-20s  %8.1f  %9.1f  %7.2fx  %8.0f%%\n",
			label, denseUs, sparseUs, speedup, flopSave)

		cuda.Release(xT)
		cuda.Release(wT)
		cuda.Release(wtT)
		cuda.Release(outDense)
		cuda.Release(outSparse)
		cuda.Release(activeIdxBuf)
		cuda.Release(countBuf)
	}

	// FP16 sparse benchmark
	if !SparseFP16KernelsLoaded() {
		fmt.Println("\n  (FP16 sparse kernels not loaded — recompile kernels)")
		return
	}

	fmt.Printf("\n=== FP16 Sparse FFN GPU Benchmark ===\n")
	fmt.Printf("%-20s  %9s  %9s  %9s  %9s  %6s\n",
		"Config", "Dense32", "SparseFP16", "vs Dense", "SparseFP32", "16vs32")
	fmt.Printf("%-20s  %9s  %9s  %9s  %9s  %6s\n",
		"--------------------", "---------", "----------", "---------", "----------", "------")

	fp16Cases := []testCase{
		{128, 512, 0.80},
		{1024, 4096, 0.80},
		{1024, 4096, 0.90},
		{4096, 11008, 0.50},
		{4096, 11008, 0.67},
		{4096, 11008, 0.80},
		{4096, 11008, 0.90},
		{4096, 11008, 0.95},
	}

	for _, tc := range fp16Cases {
		rows := tc.rows
		cols := tc.cols

		// FP32 data for sparse32 comparison
		x32 := make([]float32, cols)
		for i := range x32 {
			if rand.Float64() > tc.sparsity {
				x32[i] = rand.Float32()*2 - 1
			}
		}
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

		// Upload FP16
		xT16 := cuda.FromHostFP16(x32, []int{cols})
		wtT16 := cuda.FromHostFP16(WT, []int{cols * rows})

		// FP16 output buffer (half the size)
		fp16OutBytes := rows * 2 // FP16 = 2 bytes
		fp16OutPoolKey := (fp16OutBytes + 3) / 4
		outFP16 := cuda.Zeros([]int{fp16OutPoolKey})

		// FP32 versions for sparse32
		xT32 := cuda.FromHost(x32, []int{cols})
		wtT32 := cuda.FromHost(WT, []int{cols, rows})
		outFP32 := cuda.Zeros([]int{rows})

		activeIdxBuf := cuda.Zeros([]int{cols})
		countBuf := cuda.Zeros([]int{1})

		// Get active count via FP16 relu_and_index
		xCopy16 := cuda.CopyT(xT16)
		KReLUAndIndexFP16(xCopy16.DevicePtr(), activeIdxBuf.DevicePtr(), countBuf.DevicePtr(), cols)
		cuda.Sync()
		countHost := cuda.ToHost(countBuf)
		activeCount := int(*(*int32)(unsafe.Pointer(&countHost[0])))
		cuda.Release(xCopy16)

		iters := 100

		// Dense FP32 (cuBLAS) as baseline
		wT32mat := cuda.FromHost(W, []int{rows, cols})
		timer := NewCUDATimer()
		for i := 0; i < 5; i++ {
			r := cuda.MatMulT(wT32mat, xT32, rows, cols, 1)
			cuda.Release(r)
		}
		cuda.Sync()
		timer.Start()
		for i := 0; i < iters; i++ {
			r := cuda.MatMulT(wT32mat, xT32, rows, cols, 1)
			cuda.Release(r)
		}
		denseFP32Ms := timer.StopMs()
		timer.Destroy()
		denseFP32Us := float64(denseFP32Ms) * 1000 / float64(iters)
		cuda.Release(wT32mat)

		// Sparse FP16
		timer2 := NewCUDATimer()
		timer2.Start()
		for i := 0; i < iters; i++ {
			KSparseMatMulFP16(
				outFP16.DevicePtr(),
				wtT16.DevicePtr(),
				xT16.DevicePtr(),
				activeIdxBuf.DevicePtr(),
				activeCount, rows, cols,
			)
		}
		sparseFP16Ms := timer2.StopMs()
		timer2.Destroy()
		sparseFP16Us := float64(sparseFP16Ms) * 1000 / float64(iters)

		// Sparse FP32 for comparison
		// Get FP32 active index
		xCopy32 := cuda.CopyT(xT32)
		KReLUAndIndex(xCopy32.DevicePtr(), activeIdxBuf.DevicePtr(), countBuf.DevicePtr(), cols)
		cuda.Sync()
		cuda.Release(xCopy32)

		timer3 := NewCUDATimer()
		timer3.Start()
		for i := 0; i < iters; i++ {
			KSparseMatMul(
				outFP32.DevicePtr(),
				wtT32.DevicePtr(),
				xT32.DevicePtr(),
				activeIdxBuf.DevicePtr(),
				activeCount, rows, cols,
			)
		}
		sparseFP32Ms := timer3.StopMs()
		timer3.Destroy()
		sparseFP32Us := float64(sparseFP32Ms) * 1000 / float64(iters)

		speedupVsDense := denseFP32Us / sparseFP16Us
		fp16vsFP32 := sparseFP32Us / sparseFP16Us

		label := fmt.Sprintf("%dx%d @%.0f%%", rows, cols, tc.sparsity*100)
		fmt.Printf("%-20s  %7.1fµs  %7.1fµs  %8.2fx  %7.1fµs  %5.2fx\n",
			label, denseFP32Us, sparseFP16Us, speedupVsDense, sparseFP32Us, fp16vsFP32)

		cuda.Release(xT16)
		cuda.Release(wtT16)
		cuda.Release(outFP16)
		cuda.Release(xT32)
		cuda.Release(wtT32)
		cuda.Release(outFP32)
		cuda.Release(activeIdxBuf)
		cuda.Release(countBuf)
	}
}
