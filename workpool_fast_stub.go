//go:build !linux || !cgo

package mongoose

import (
	"fmt"
	"time"
)

type FastWorkPool struct{ numWorkers int }
type FastWorkResult struct {
	TotalItems int
	WallTime   time.Duration
	PerWorker  []int
}

func RunFastCPUPool(numWorkers int, dim int, batchSize int) FastWorkResult {
	return FastWorkResult{}
}

func BenchmarkFastVsGo(dim, batchSize, numWorkers int) {
	fmt.Println("FastWorkPool not available (requires linux+cgo)")
}
