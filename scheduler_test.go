//go:build linux && cgo

package mongoose

import (
	"fmt"
	"testing"
)

func TestSchedulerCalibrate(t *testing.T) {
	cuda := NewCUDA()
	if cuda == nil {
		t.Skip("no CUDA")
	}
	cpu := &CPU{}

	sched := NewScheduler(cuda, cpu)

	// Calibrate matmul at different sizes
	dims := []struct{ m, k, n int }{
		{64, 64, 64},
		{128, 128, 128},
		{256, 256, 256},
		{512, 512, 512},
		{4096, 11008, 1},   // ReluLLaMA FFN mat-vec
	}

	for _, d := range dims {
		key := MatMulKey(d.m, d.k, d.n)
		a := make([]float32, d.m*d.k)
		b := make([]float32, d.k*d.n)
		for i := range a { a[i] = 0.001 * float32(i%1000) }
		for i := range b { b[i] = 0.001 * float32(i%997) }

		sched.CalibrateAll(key, func(eng Engine) {
			eng.MatMul(a, b, d.m, d.k, d.n)
		})
	}

	// Calibrate element-wise ops
	x4096 := make([]float32, 4096)
	w4096 := make([]float32, 4096)
	for i := range x4096 { x4096[i] = float32(i) * 0.001; w4096[i] = 1.0 }

	sched.CalibrateAll(NormKey(4096), func(eng Engine) {
		buf := make([]float32, 4096)
		copy(buf, x4096)
		eng.RMSNorm(buf, w4096, 1e-6)
	})

	x11008 := make([]float32, 11008)
	for i := range x11008 { x11008[i] = float32(i) - 5504 }
	sched.CalibrateAll(ReLUKey(11008), func(eng Engine) {
		buf := make([]float32, 11008)
		copy(buf, x11008)
		eng.ReLU(buf)
	})

	x1024 := make([]float32, 1024)
	for i := range x1024 { x1024[i] = float32(i) * 0.01 }
	sched.CalibrateAll(SoftMaxKey(1024), func(eng Engine) {
		buf := make([]float32, 1024)
		copy(buf, x1024)
		eng.SoftMax(buf, 1024)
	})

	// Print full calibration table
	fmt.Println()
	fmt.Println(sched.String())

	// Simulate scheduling a transformer layer's operations
	fmt.Println("=== Simulated Layer Schedule (ReluLLaMA-7B) ===")
	layerOps := []OpKey{
		NormKey(4096),              // pre-attn norm
		MatMulKey(4096, 4096, 1),   // Q projection (if calibrated) — use closest
		MatMulKey(4096, 4096, 1),   // K projection
		MatMulKey(4096, 4096, 1),   // V projection
		SoftMaxKey(1024),           // attention softmax
		MatMulKey(4096, 4096, 1),   // O projection
		NormKey(4096),              // pre-FFN norm
		MatMulKey(4096, 11008, 1),  // gate projection
		MatMulKey(4096, 11008, 1),  // up projection
		ReLUKey(11008),             // ReLU activation
		MatMulKey(4096, 11008, 1),  // down projection (reversed dims but same key)
	}

	assignments := sched.Assign(layerOps)

	for i, key := range layerOps {
		gpuIdx := assignments[i]
		gpuName := "unknown"
		if gpuIdx < sched.NumGPUs() {
			gpuName = sched.gpus[gpuIdx].name
		}
		us := sched.TimeFor(gpuIdx, key)
		fmt.Printf("  %-25s → GPU %d (%s) [%.1f µs]\n", key, gpuIdx, gpuName, us)
	}

	// Count assignments per GPU
	counts := make(map[int]int)
	for _, a := range assignments {
		counts[a]++
	}
	fmt.Println()
	for gpuIdx, count := range counts {
		fmt.Printf("  GPU %d (%s): %d ops assigned\n", gpuIdx, sched.gpus[gpuIdx].name, count)
	}
}

func TestSchedulerHomogeneous(t *testing.T) {
	// Simulate 2x identical GPUs — should split 50/50
	cpu1 := &CPU{}
	cpu2 := &CPU{}

	sched := NewScheduler(cpu1, cpu2)

	dim := 64
	a := make([]float32, dim*dim)
	b := make([]float32, dim*dim)
	for i := range a { a[i] = float32(i) * 0.001 }
	for i := range b { b[i] = float32(i) * 0.001 }

	key := MatMulKey(dim, dim, dim)
	sched.CalibrateAll(key, func(eng Engine) {
		eng.MatMul(a, b, dim, dim, dim)
	})

	// Create 100 identical ops
	keys := make([]OpKey, 100)
	for i := range keys { keys[i] = key }

	assignments := sched.Assign(keys)

	counts := [2]int{}
	for _, a := range assignments {
		counts[a]++
	}

	fmt.Printf("\n=== Homogeneous 2x CPU: %d/%d split ===\n", counts[0], counts[1])

	// Should be close to 50/50
	if counts[0] < 40 || counts[1] < 40 {
		t.Errorf("unbalanced split: %d/%d (expected ~50/50)", counts[0], counts[1])
	} else {
		fmt.Println("  PASS — balanced split")
	}
}
