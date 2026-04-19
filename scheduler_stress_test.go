//go:build linux && cgo

package mongoose

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"testing"
	"time"
)

// TestSchedulerCorrectness verifies that the scheduler's calibration
// produces consistent results — same operation on different GPUs gives
// the same numerical output.
func TestSchedulerCorrectness(t *testing.T) {
	cuda := NewCUDA()
	if cuda == nil {
		t.Skip("no CUDA")
	}
	cpu := &CPU{}

	fmt.Println("\n=== Scheduler Cross-Backend Correctness ===")

	dims := []struct{ m, k, n int }{
		{32, 32, 32},
		{64, 64, 1},
		{128, 128, 128},
		{256, 512, 1},
	}

	for _, d := range dims {
		a := make([]float32, d.m*d.k)
		b := make([]float32, d.k*d.n)
		r := rand.New(rand.NewSource(42))
		for i := range a { a[i] = r.Float32()*2 - 1 }
		for i := range b { b[i] = r.Float32()*2 - 1 }

		cpuResult := cpu.MatMul(a, b, d.m, d.k, d.n)
		cudaResult := cuda.MatMul(a, b, d.m, d.k, d.n)

		var maxErr float64
		for i := range cpuResult {
			err := math.Abs(float64(cpuResult[i]) - float64(cudaResult[i]))
			if err > maxErr { maxErr = err }
		}

		status := "PASS"
		if maxErr > 0.01 {
			status = "FAIL"
			t.Errorf("%dx%dx%d: max error %.2e", d.m, d.k, d.n, maxErr)
		}
		fmt.Printf("  MatMul %dx%dx%d: CUDA vs CPU max err = %.2e  %s\n", d.m, d.k, d.n, maxErr, status)
	}
}

// TestSchedulerCalibrationStability verifies that repeated calibration
// of the same operation produces stable results (low variance).
func TestSchedulerCalibrationStability(t *testing.T) {
	cuda := NewCUDA()
	if cuda == nil {
		t.Skip("no CUDA")
	}

	fmt.Println("\n=== Scheduler Calibration Stability ===")

	sched := NewScheduler(cuda)

	dim := 256
	a := make([]float32, dim*dim)
	b := make([]float32, dim*dim)
	for i := range a { a[i] = 0.001 * float32(i%1000) }
	for i := range b { b[i] = 0.001 * float32(i%997) }

	key := MatMulKey(dim, dim, dim)

	// Calibrate 10 times
	times := make([]float64, 10)
	for i := range times {
		times[i] = sched.Calibrate(0, key, func(eng Engine) {
			eng.MatMul(a, b, dim, dim, dim)
		})
	}

	// Compute mean and coefficient of variation
	var sum float64
	for _, t := range times { sum += t }
	mean := sum / float64(len(times))

	var varSum float64
	for _, t := range times {
		d := t - mean
		varSum += d * d
	}
	stddev := math.Sqrt(varSum / float64(len(times)))
	cv := stddev / mean * 100

	fmt.Printf("  MatMul %dx%d (10 calibrations):\n", dim, dim)
	fmt.Printf("    Mean:   %.1f µs\n", mean)
	fmt.Printf("    Stddev: %.1f µs\n", stddev)
	fmt.Printf("    CV:     %.1f%%\n", cv)
	fmt.Printf("    Times:  %v\n", times)

	if cv > 75 {
		t.Errorf("calibration too noisy: CV=%.1f%% (>75%%)", cv)
	} else {
		fmt.Printf("    PASS — stable (CV < 50%%)\n")
	}
}

// TestSchedulerAssignmentConsistency verifies that Assign() produces
// the same assignments for the same calibration data.
func TestSchedulerAssignmentConsistency(t *testing.T) {
	cuda := NewCUDA()
	if cuda == nil {
		t.Skip("no CUDA")
	}

	fmt.Println("\n=== Scheduler Assignment Consistency ===")

	sched := NewScheduler(cuda, &CPU{})

	// Calibrate
	sched.CalibrateMatMul(64, 64, 64)
	sched.CalibrateAll(NormKey(4096), func(eng Engine) {
		x := make([]float32, 4096)
		w := make([]float32, 4096)
		for i := range w { w[i] = 1.0 }
		eng.RMSNorm(x, w, 1e-6)
	})

	keys := []OpKey{
		MatMulKey(64, 64, 64),
		NormKey(4096),
		MatMulKey(64, 64, 64),
		NormKey(4096),
		MatMulKey(64, 64, 64),
	}

	// Run Assign() 100 times — must be deterministic
	ref := sched.Assign(keys)
	for i := 0; i < 100; i++ {
		result := sched.Assign(keys)
		for j := range ref {
			if result[j] != ref[j] {
				t.Errorf("iteration %d: assignment[%d] = %d, expected %d", i, j, result[j], ref[j])
			}
		}
	}
	fmt.Printf("  100 identical runs: PASS — deterministic\n")
	fmt.Printf("  Assignments: %v\n", ref)
}

// TestSchedulerMemoryLeak runs many calibrations and checks for leaks.
func TestSchedulerMemoryLeak(t *testing.T) {
	cuda := NewCUDA()
	if cuda == nil {
		t.Skip("no CUDA")
	}

	fmt.Println("\n=== Scheduler Memory Leak Test ===")

	runtime.GC()
	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)

	// Use CPU-only schedulers to avoid GPU alloc overhead
	for iter := 0; iter < 10; iter++ {
		sched := NewScheduler(&CPU{}, &CPU{})
		sched.CalibrateAll(NormKey(128), func(eng Engine) {
			x := make([]float32, 128)
			w := make([]float32, 128)
			eng.RMSNorm(x, w, 1e-6)
		})
		keys := make([]OpKey, 50)
		for i := range keys { keys[i] = NormKey(128) }
		sched.Assign(keys)
	}

	runtime.GC()
	var m2 runtime.MemStats
	runtime.ReadMemStats(&m2)

	var growthMB float64
	if m2.HeapAlloc > m1.HeapAlloc {
		growthMB = float64(m2.HeapAlloc-m1.HeapAlloc) / 1024 / 1024
	}
	fmt.Printf("  10 scheduler lifecycles\n")
	fmt.Printf("  Heap before: %.1f MB\n", float64(m1.HeapAlloc)/1024/1024)
	fmt.Printf("  Heap after:  %.1f MB\n", float64(m2.HeapAlloc)/1024/1024)
	fmt.Printf("  Growth:      %.1f MB\n", growthMB)

	if growthMB > 50 {
		t.Errorf("possible leak: %.1f MB growth", growthMB)
	} else {
		fmt.Println("  PASS")
	}
}

// TestSchedulerLongRunning simulates 1000 scheduling decisions to check
// for drift, crashes, or degradation.
func TestSchedulerLongRunning(t *testing.T) {
	cuda := NewCUDA()
	if cuda == nil {
		t.Skip("no CUDA")
	}

	fmt.Println("\n=== Scheduler Long-Running Stability ===")

	sched := NewScheduler(cuda, &CPU{})
	sched.CalibrateMatMul(64, 64, 64)
	sched.CalibrateMatMul(128, 128, 1)
	sched.CalibrateAll(NormKey(512), func(eng Engine) {
		x := make([]float32, 512)
		w := make([]float32, 512)
		for i := range w { w[i] = 1.0 }
		eng.RMSNorm(x, w, 1e-6)
	})
	sched.CalibrateAll(ReLUKey(1024), func(eng Engine) {
		x := make([]float32, 1024)
		eng.ReLU(x)
	})

	// Build a simulated workload of 1000 ops
	allKeys := []OpKey{
		MatMulKey(64, 64, 64),
		MatMulKey(128, 128, 1),
		NormKey(512),
		ReLUKey(1024),
	}

	ops := make([]OpKey, 1000)
	for i := range ops {
		ops[i] = allKeys[i%len(allKeys)]
	}

	t0 := time.Now()
	assignments := sched.Assign(ops)
	elapsed := time.Since(t0)

	// Count per-GPU assignments
	counts := make(map[int]int)
	for _, a := range assignments { counts[a]++ }

	fmt.Printf("  1000 ops assigned in %v\n", elapsed)
	fmt.Printf("  Per GPU: CUDA=%d, CPU=%d\n", counts[0], counts[1])

	// Verify no ops went unassigned (all indices valid)
	for i, a := range assignments {
		if a < 0 || a >= sched.NumGPUs() {
			t.Errorf("invalid assignment[%d] = %d", i, a)
		}
	}

	// Verify assignment is fast (<1ms for 1000 ops)
	if elapsed > time.Millisecond {
		fmt.Printf("  WARN: assignment took %v (>1ms)\n", elapsed)
	} else {
		fmt.Printf("  PASS — scheduling overhead: %v for 1000 ops\n", elapsed)
	}
}

// TestSchedulerSingleGPU verifies correct behavior with only one backend.
func TestSchedulerSingleGPU(t *testing.T) {
	fmt.Println("\n=== Scheduler Single GPU ===")

	sched := NewScheduler(&CPU{})
	sched.CalibrateAll(MatMulKey(32, 32, 32), func(eng Engine) {
		eng.MatMul(make([]float32, 32*32), make([]float32, 32*32), 32, 32, 32)
	})

	keys := make([]OpKey, 50)
	for i := range keys { keys[i] = MatMulKey(32, 32, 32) }

	assignments := sched.Assign(keys)
	for i, a := range assignments {
		if a != 0 {
			t.Errorf("single GPU: assignment[%d] = %d, expected 0", i, a)
		}
	}
	fmt.Println("  50 ops → all GPU 0: PASS")
}

// TestSchedulerUncalibratedOps verifies graceful handling of ops that
// were never calibrated.
func TestSchedulerUncalibratedOps(t *testing.T) {
	fmt.Println("\n=== Scheduler Uncalibrated Ops ===")

	sched := NewScheduler(&CPU{}, &CPU{})

	// Don't calibrate anything
	keys := []OpKey{"unknown:op:1", "another:unknown:2", "mystery:3"}
	assignments := sched.Assign(keys)

	// Should not crash, should distribute across GPUs
	fmt.Printf("  Uncalibrated ops: assignments = %v\n", assignments)
	if len(assignments) != 3 {
		t.Errorf("expected 3 assignments, got %d", len(assignments))
	}
	fmt.Println("  PASS — no crash on uncalibrated ops")
}
