package mongoose

import (
	"fmt"
	"log"
	"sync"
	"time"
)

// Scheduler assigns operations to GPUs based on measured wall-clock times.
// No theoretical FLOP counts, no hardware-specific tuning. Each GPU times
// itself on the first invocation of each operation shape. The fastest GPU
// for each shape gets that work.
//
// Two H100s: both measure ~113µs on 4096×11008 matmul. Both get matmul work. 2x scaling.
// H100 + Xe: H100 measures 113µs, Xe measures 7100µs on matmul. H100 gets matmuls.
//            H100 measures 4µs, Xe measures 12µs on RMSNorm. H100 still faster, but
//            Xe is fast ENOUGH that it's worth offloading norms to free H100 for matmuls.
type Scheduler struct {
	gpus []scheduledGPU
	mu   sync.RWMutex

	// Calibration cache: opKey → per-GPU measured microseconds
	calibration map[string][]float64 // calibration[opKey][gpuIdx] = microseconds
}

type scheduledGPU struct {
	engine Engine
	name   string
	gflops float64
}

// OpKey uniquely identifies an operation shape for calibration.
// Examples: "matmul:4096:11008:1", "rmsnorm:4096", "relu:11008"
type OpKey = string

func MatMulKey(m, k, n int) OpKey { return fmt.Sprintf("mm:%d:%d:%d", m, k, n) }
func NormKey(dim int) OpKey       { return fmt.Sprintf("norm:%d", dim) }
func ReLUKey(n int) OpKey         { return fmt.Sprintf("relu:%d", n) }
func SoftMaxKey(n int) OpKey      { return fmt.Sprintf("sm:%d", n) }
func CustomKey(op string, dims ...int) OpKey {
	s := op
	for _, d := range dims {
		s += fmt.Sprintf(":%d", d)
	}
	return s
}

// NewScheduler creates a scheduler with the given GPU backends.
func NewScheduler(engines ...Engine) *Scheduler {
	s := &Scheduler{
		calibration: make(map[string][]float64),
	}
	for _, eng := range engines {
		s.gpus = append(s.gpus, scheduledGPU{
			engine: eng,
			name:   eng.Name(),
			gflops: eng.Benchmark(),
		})
	}
	return s
}

// AddGPU adds a GPU to the scheduler.
func (s *Scheduler) AddGPU(eng Engine) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	idx := len(s.gpus)
	s.gpus = append(s.gpus, scheduledGPU{
		engine: eng,
		name:   eng.Name(),
		gflops: eng.Benchmark(),
	})
	return idx
}


// NumGPUs returns the number of GPUs in the scheduler.
func (s *Scheduler) NumGPUs() int { return len(s.gpus) }

// GPUName returns the name of GPU at index i.
func (s *Scheduler) GPUName(i int) string {
	if i >= len(s.gpus) { return "unknown" }
	return s.gpus[i].name
}

// Calibrate times a specific operation on a specific GPU.
// The op function receives the engine and returns after completing the work.
func (s *Scheduler) Calibrate(gpuIdx int, key OpKey, op func(Engine)) float64 {
	if gpuIdx >= len(s.gpus) {
		return 0
	}

	eng := s.gpus[gpuIdx].engine

	// Warmup
	op(eng)

	// Measure (average of 3 runs for stability)
	var totalUs float64
	runs := 3
	for i := 0; i < runs; i++ {
		t0 := time.Now()
		op(eng)
		totalUs += float64(time.Since(t0).Microseconds())
	}
	avgUs := totalUs / float64(runs)

	s.mu.Lock()
	if s.calibration[key] == nil {
		s.calibration[key] = make([]float64, len(s.gpus))
	}
	// Grow slice if GPUs were added after initial calibration
	for len(s.calibration[key]) < len(s.gpus) {
		s.calibration[key] = append(s.calibration[key], 0)
	}
	s.calibration[key][gpuIdx] = avgUs
	s.mu.Unlock()

	return avgUs
}

// CalibrateAll times a specific operation on ALL GPUs.
func (s *Scheduler) CalibrateAll(key OpKey, op func(Engine)) {
	for i := range s.gpus {
		us := s.Calibrate(i, key, op)
		log.Printf("[scheduler] %s on %s: %.1f µs", key, s.gpus[i].name, us)
	}
}

// CalibrateMatMul calibrates a matmul shape on all GPUs via Engine.MatMul.
func (s *Scheduler) CalibrateMatMul(m, k, n int) {
	key := MatMulKey(m, k, n)

	a := make([]float32, m*k)
	b := make([]float32, k*n)
	for i := range a {
		a[i] = 0.001 * float32(i%1000)
	}
	for i := range b {
		b[i] = 0.001 * float32(i%997)
	}

	maxHostCopyBytes := 32 * 1024 * 1024
	matBytes := m * k * 4

	for gpuIdx := range s.gpus {
		g := &s.gpus[gpuIdx]
		if matBytes > maxHostCopyBytes {
			log.Printf("[scheduler] %s on %s: SKIP (matrix %d MB > %d MB limit)",
				key, g.name, matBytes/1024/1024, maxHostCopyBytes/1024/1024)
			continue
		}
		us := s.Calibrate(gpuIdx, key, func(eng Engine) {
			eng.MatMul(a, b, m, k, n)
		})
		log.Printf("[scheduler] %s on %s: %.1f µs", key, g.name, us)
	}
}

// Fastest returns the GPU index that is fastest for the given operation.
// Returns -1 if the operation hasn't been calibrated.
func (s *Scheduler) Fastest(key OpKey) int {
	s.mu.RLock()
	defer s.mu.RUnlock()

	times := s.calibration[key]
	if times == nil {
		return 0 // default to first GPU if uncalibrated
	}

	best := -1
	bestTime := float64(0)
	for i, t := range times {
		if t > 0 && (best < 0 || t < bestTime) {
			best = i
			bestTime = t
		}
	}
	if best < 0 {
		return 0
	}
	return best
}

// TimeFor returns the calibrated time for a GPU on an operation.
func (s *Scheduler) TimeFor(gpuIdx int, key OpKey) float64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if times := s.calibration[key]; times != nil && gpuIdx < len(times) {
		return times[gpuIdx]
	}
	return 0
}

// Assign returns which GPU should handle each operation in a list,
// minimizing total wall time assuming operations execute sequentially.
// This is the core scheduling algorithm.
//
// For independent ops (batched inference): assign each op to its fastest GPU.
// For dependent ops (sequential inference): assign to minimize the critical path.
func (s *Scheduler) Assign(keys []OpKey) []int {
	s.mu.RLock()
	defer s.mu.RUnlock()

	n := len(keys)
	assignments := make([]int, n)

	if len(s.gpus) == 1 {
		// Single GPU — everything goes there
		return assignments
	}

	// Greedy: each op goes to the GPU that finishes it earliest,
	// considering that GPU's current accumulated load.
	load := make([]float64, len(s.gpus)) // accumulated µs per GPU

	for i, key := range keys {
		times := s.calibration[key]
		if times == nil {
			// Uncalibrated — assign to least loaded GPU
			minLoad := 0
			for j := range load {
				if load[j] < load[minLoad] {
					minLoad = j
				}
			}
			assignments[i] = minLoad
			continue
		}

		// Find GPU that finishes this op earliest (load + op_time)
		bestGPU := 0
		bestFinish := load[0] + times[0]
		for j := 1; j < len(s.gpus); j++ {
			t := times[j]
			if t <= 0 {
				continue // uncalibrated on this GPU
			}
			finish := load[j] + t
			if finish < bestFinish {
				bestFinish = finish
				bestGPU = j
			}
		}
		assignments[i] = bestGPU
		load[bestGPU] += times[bestGPU]
	}

	return assignments
}

// String returns a summary of calibration data.
func (s *Scheduler) String() string {
	s.mu.RLock()
	defer s.mu.RUnlock()

	out := fmt.Sprintf("Scheduler: %d GPUs\n", len(s.gpus))
	for i, g := range s.gpus {
		out += fmt.Sprintf("  [%d] %s (%.0f GFLOPS)\n", i, g.name, g.gflops)
	}
	if len(s.calibration) > 0 {
		out += "\nCalibration:\n"
		out += fmt.Sprintf("  %-25s", "Operation")
		for _, g := range s.gpus {
			out += fmt.Sprintf("  %15s", g.name)
		}
		out += "  Best\n"
		for key, times := range s.calibration {
			out += fmt.Sprintf("  %-25s", key)
			best := 0
			for i, t := range times {
				out += fmt.Sprintf("  %12.1f µs", t)
				if t > 0 && (times[best] <= 0 || t < times[best]) {
					best = i
				}
			}
			out += fmt.Sprintf("  → GPU %d\n", best)
		}
	}
	return out
}
