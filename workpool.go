package mongoose

import (
	"fmt"
	"log"
	"sync"
	"time"
)

// WorkPool distributes independent compute tasks across all available GPUs
// proportionally to their measured performance. Every GPU contributes its
// fair share — no tuning, no special-casing.
//
// A 5090 at 100 GFLOPS + an Intel Xe at 1.56 GFLOPS = 101.56 GFLOPS total.
// The Xe contributes 1/64th. Total throughput scales by 1/64th.
// Add a phone GPU at 0.1 GFLOPS and total is 101.66. Universal scaling.
//
// Usage:
//
//	pool := mongoose.NewWorkPool()
//	pool.AddEngine("cuda/5090", cudaEngine, 100.0)
//	pool.AddEngine("webgpu/xe", xeEngine, 1.56)
//	results := pool.Run(tasks)
type WorkPool struct {
	workers []worker
	mu      sync.Mutex
}

type worker struct {
	name   string
	eng    Engine
	gflops float64
	share  float64 // fraction of total work [0, 1]
}

// WorkItem is a unit of work to be dispatched to any GPU.
type WorkItem struct {
	ID   int
	A, B []float32
	M, K, N int
}

// WorkResult is the output from a completed work item.
type WorkResult struct {
	ID     int
	C      []float32
	Worker string
	Micros int64 // execution time in microseconds
}

// NewWorkPool creates an empty work pool.
func NewWorkPool() *WorkPool {
	return &WorkPool{}
}

// AddEngine adds a compute backend to the pool with its measured GFLOPS.
func (wp *WorkPool) AddEngine(name string, eng Engine, gflops float64) {
	wp.mu.Lock()
	defer wp.mu.Unlock()
	wp.workers = append(wp.workers, worker{name: name, eng: eng, gflops: gflops})
	wp.recalcShares()
}

func (wp *WorkPool) recalcShares() {
	var total float64
	for _, w := range wp.workers {
		total += w.gflops
	}
	for i := range wp.workers {
		wp.workers[i].share = wp.workers[i].gflops / total
	}
}

// NumWorkers returns the number of backends in the pool.
func (wp *WorkPool) NumWorkers() int {
	return len(wp.workers)
}

// TotalGFLOPS returns the sum of all workers' measured performance.
func (wp *WorkPool) TotalGFLOPS() float64 {
	var total float64
	for _, w := range wp.workers {
		total += w.gflops
	}
	return total
}

// Run distributes work items across all backends using pre-partitioned assignment.
// Each worker gets a contiguous slice of items proportional to its GFLOPS.
// No contention, no work-stealing, no scheduling overhead. Workers run fully
// independent — fast workers never block on slow workers.
func (wp *WorkPool) Run(items []WorkItem) []WorkResult {
	n := len(items)
	if n == 0 {
		return nil
	}
	results := make([]WorkResult, n)

	// Pre-partition: assign contiguous slices proportional to GFLOPS
	var totalGFLOPS float64
	for _, w := range wp.workers {
		totalGFLOPS += w.gflops
	}

	type partition struct {
		start, end int
		w          *worker
	}
	parts := make([]partition, len(wp.workers))
	assigned := 0
	for i := range wp.workers {
		share := int(wp.workers[i].gflops / totalGFLOPS * float64(n))
		if share < 1 && i < len(wp.workers)-1 {
			share = 1
		}
		if assigned+share > n {
			share = n - assigned
		}
		parts[i] = partition{start: assigned, end: assigned + share, w: &wp.workers[i]}
		assigned += share
	}
	// Give remainder to fastest
	if assigned < n {
		fastest := 0
		for i := range wp.workers {
			if wp.workers[i].gflops > wp.workers[fastest].gflops {
				fastest = i
			}
		}
		parts[fastest].end += n - assigned
	}

	// Launch all workers in parallel — each processes its own slice
	var wg sync.WaitGroup
	for _, p := range parts {
		if p.start >= p.end {
			continue
		}
		wg.Add(1)
		go func(p partition) {
			defer wg.Done()
			for idx := p.start; idx < p.end; idx++ {
				item := items[idx]
				t0 := time.Now()
				c := p.w.eng.MatMul(item.A, item.B, item.M, item.K, item.N)
				micros := time.Since(t0).Microseconds()
				results[item.ID] = WorkResult{
					ID:     item.ID,
					C:      c,
					Worker: p.w.name,
					Micros: micros,
				}
			}
		}(p)
	}

	wg.Wait()
	return results
}

// RunTimed is like Run but also returns total wall time and per-worker stats.
func (wp *WorkPool) RunTimed(items []WorkItem) ([]WorkResult, time.Duration, map[string]int) {
	workerCounts := make(map[string]int)
	n := len(items)
	if n == 0 {
		return nil, 0, workerCounts
	}
	results := make([]WorkResult, n)

	var totalGFLOPS float64
	for _, w := range wp.workers {
		totalGFLOPS += w.gflops
	}

	type partition struct {
		start, end int
		w          *worker
	}
	parts := make([]partition, len(wp.workers))
	assigned := 0
	for i := range wp.workers {
		share := int(wp.workers[i].gflops / totalGFLOPS * float64(n))
		if share < 1 && i < len(wp.workers)-1 {
			share = 1
		}
		if assigned+share > n {
			share = n - assigned
		}
		parts[i] = partition{start: assigned, end: assigned + share, w: &wp.workers[i]}
		assigned += share
	}
	if assigned < n {
		fastest := 0
		for i := range wp.workers {
			if wp.workers[i].gflops > wp.workers[fastest].gflops {
				fastest = i
			}
		}
		parts[fastest].end += n - assigned
	}

	t0 := time.Now()
	var wg sync.WaitGroup
	for _, p := range parts {
		if p.start >= p.end {
			continue
		}
		wg.Add(1)
		go func(p partition) {
			defer wg.Done()
			for idx := p.start; idx < p.end; idx++ {
				item := items[idx]
				t := time.Now()
				c := p.w.eng.MatMul(item.A, item.B, item.M, item.K, item.N)
				micros := time.Since(t).Microseconds()
				results[item.ID] = WorkResult{
					ID: item.ID, C: c, Worker: p.w.name, Micros: micros,
				}
			}
		}(p)
	}

	wg.Wait()
	elapsed := time.Since(t0)

	for _, p := range parts {
		workerCounts[p.w.name] = p.end - p.start
	}

	return results, elapsed, workerCounts
}

// Benchmark runs a standardized workload and reports per-worker and combined throughput.
func (wp *WorkPool) Benchmark(dim, batchSize int) {
	log.Printf("[workpool] Benchmarking %d workers, batch=%d, dim=%d", len(wp.workers), batchSize, dim)

	// Create batch of identical matmuls
	a := make([]float32, dim*dim)
	b := make([]float32, dim*dim)
	for i := range a {
		a[i] = 0.001 * float32(i%1000)
		b[i] = 0.001 * float32(i%997)
	}

	items := make([]WorkItem, batchSize)
	for i := range items {
		items[i] = WorkItem{ID: i, A: a, B: b, M: dim, K: dim, N: dim}
	}

	// Run with just the fastest worker (baseline)
	fastest := 0
	for i := range wp.workers {
		if wp.workers[i].gflops > wp.workers[fastest].gflops {
			fastest = i
		}
	}

	singlePool := NewWorkPool()
	singlePool.AddEngine(wp.workers[fastest].name, wp.workers[fastest].eng, wp.workers[fastest].gflops)
	_, baseTime, baseCounts := singlePool.RunTimed(items)

	// Run with all workers
	_, allTime, allCounts := wp.RunTimed(items)

	// Report
	flopsPerOp := float64(2 * dim * dim * dim)
	baseGFLOPS := flopsPerOp * float64(batchSize) / baseTime.Seconds() / 1e9
	allGFLOPS := flopsPerOp * float64(batchSize) / allTime.Seconds() / 1e9
	speedup := baseTime.Seconds() / allTime.Seconds()

	fmt.Printf("\n=== WorkPool Benchmark: %dx%d, batch=%d ===\n", dim, dim, batchSize)
	fmt.Printf("\n  Baseline (fastest worker only: %s):\n", wp.workers[fastest].name)
	fmt.Printf("    Time:  %v\n", baseTime)
	fmt.Printf("    GFLOPS: %.1f\n", baseGFLOPS)
	for name, count := range baseCounts {
		fmt.Printf("    %s: %d items\n", name, count)
	}

	fmt.Printf("\n  All workers combined:\n")
	fmt.Printf("    Time:  %v\n", allTime)
	fmt.Printf("    GFLOPS: %.1f\n", allGFLOPS)
	fmt.Printf("    Speedup: %.4fx\n", speedup)
	for name, count := range allCounts {
		pct := float64(count) / float64(batchSize) * 100
		fmt.Printf("    %s: %d items (%.1f%%)\n", name, count, pct)
	}

	fmt.Printf("\n  Theoretical:\n")
	fmt.Printf("    Sum of worker GFLOPS: %.1f\n", wp.TotalGFLOPS())
	fmt.Printf("    Fastest worker:       %.1f GFLOPS\n", wp.workers[fastest].gflops)
	fmt.Printf("    Expected speedup:     %.4fx\n", wp.TotalGFLOPS()/wp.workers[fastest].gflops)
	fmt.Printf("    Actual speedup:       %.4fx\n", speedup)
	fmt.Printf("    Efficiency:           %.1f%%\n", speedup/(wp.TotalGFLOPS()/wp.workers[fastest].gflops)*100)
}
