package mongoose

import (
	"fmt"
	"log"
)

// HybridEngine coordinates multiple GPU backends (CUDA, WebGPU, Metal, CPU)
// for heterogeneous multi-GPU inference. Each backend handles a range of layers;
// activations are passed between backends through host memory.
//
// This is the architecture that makes vendor-agnostic GPU compute real.
// An NVIDIA GPU and an Intel iGPU on the same machine, sharing the load.
// A Mac with Metal and a Linux box with CUDA, in the same mesh.
// Any GPU, any vendor, any stack. They all scale together.
type HybridEngine struct {
	backends []BackendSlot
	total    int // total layer capacity
}

// BackendSlot assigns a compute backend to a range of layers.
type BackendSlot struct {
	Engine     Engine
	// reserved for extensions
	Name       string
	LayerStart int
	LayerEnd   int // exclusive
	GFLOPS     float64 // measured performance for proportional assignment
}

// NewHybridEngine creates a hybrid engine from available backends.
// It auto-detects all available compute and assigns layers proportionally
// to each backend's measured performance.
func NewHybridEngine() *HybridEngine {
	h := &HybridEngine{}

	// Discover all available backends
	var slots []BackendSlot

	// CUDA (highest priority — discrete NVIDIA GPU)
	if cuda := NewCUDA(); cuda != nil {
		gflops := cuda.Benchmark()
		slots = append(slots, BackendSlot{
			Engine: cuda,
			Name:   cuda.Name(),
			GFLOPS:    gflops,
		})
		log.Printf("[hybrid] CUDA: %s (%.0f GFLOPS)", cuda.Name(), gflops)
	}

	// WebGPU/Vulkan — may find additional GPUs (Intel, AMD, etc.)
	// Note: WebGPU requires CGO_ENABLED=0 build. In a CGO build,
	// the WebGPU stub is used and NewWebGPU returns nil.
	// For the hybrid engine to use both CUDA+WebGPU, we need
	// the WebGPU backend to be available via a separate process.
	// For now, detect it if available.
	if wgpu := NewWebGPU(); wgpu != nil {
		gflops := wgpu.Benchmark()
		slots = append(slots, BackendSlot{
			Engine: wgpu,
			Name:   wgpu.Name(),
			GFLOPS: gflops,
		})
		log.Printf("[hybrid] WebGPU: %s (%.0f GFLOPS)", wgpu.Name(), gflops)
	}

	// CPU fallback
	cpu := &CPU{}
	cpuGflops := cpu.Benchmark()
	slots = append(slots, BackendSlot{
		Engine: cpu,
		Name:   cpu.Name(),
		GFLOPS: cpuGflops,
	})
	log.Printf("[hybrid] CPU: %s (%.1f GFLOPS)", cpu.Name(), cpuGflops)

	h.backends = slots
	return h
}

// NewHybridEngineFrom creates a hybrid engine from explicitly provided backends.
// Use this when you have specific backends to combine (e.g., CUDA + WebGPU subprocess).
func NewHybridEngineFrom(backends ...BackendSlot) *HybridEngine {
	return &HybridEngine{backends: backends}
}

// AssignLayers distributes numLayers across backends proportionally to performance.
// Returns the updated HybridEngine with LayerStart/LayerEnd set on each slot.
func (h *HybridEngine) AssignLayers(numLayers int) {
	h.total = numLayers

	if len(h.backends) == 0 {
		return
	}
	if len(h.backends) == 1 {
		h.backends[0].LayerStart = 0
		h.backends[0].LayerEnd = numLayers
		return
	}

	// Sum total GFLOPS
	var totalGFLOPS float64
	for _, b := range h.backends {
		totalGFLOPS += b.GFLOPS
	}

	// Assign proportionally, ensuring at least 1 layer per backend
	assigned := 0
	for i := range h.backends {
		share := int(float64(numLayers) * h.backends[i].GFLOPS / totalGFLOPS)
		if share < 1 {
			share = 1
		}
		if assigned+share > numLayers {
			share = numLayers - assigned
		}
		h.backends[i].LayerStart = assigned
		h.backends[i].LayerEnd = assigned + share
		assigned += share
	}

	// Give any remainder to the fastest backend
	if assigned < numLayers {
		fastest := 0
		for i := range h.backends {
			if h.backends[i].GFLOPS > h.backends[fastest].GFLOPS {
				fastest = i
			}
		}
		h.backends[fastest].LayerEnd += numLayers - assigned
	}

	// Log assignment
	for _, b := range h.backends {
		layers := b.LayerEnd - b.LayerStart
		log.Printf("[hybrid] %s: layers %d-%d (%d layers, %.0f GFLOPS)",
			b.Name, b.LayerStart, b.LayerEnd-1, layers, b.GFLOPS)
	}
}

// BackendForLayer returns the backend that owns a given layer.
func (h *HybridEngine) BackendForLayer(layer int) *BackendSlot {
	for i := range h.backends {
		if layer >= h.backends[i].LayerStart && layer < h.backends[i].LayerEnd {
			return &h.backends[i]
		}
	}
	return nil
}

// Backends returns all backend slots.
func (h *HybridEngine) Backends() []BackendSlot {
	return h.backends
}

// NumBackends returns the number of active backends.
func (h *HybridEngine) NumBackends() int {
	return len(h.backends)
}

// String returns a summary of the hybrid topology.
func (h *HybridEngine) String() string {
	s := fmt.Sprintf("HybridEngine: %d backends, %d total layers\n", len(h.backends), h.total)
	for _, b := range h.backends {
		s += fmt.Sprintf("  %s: layers [%d, %d) — %.0f GFLOPS\n",
			b.Name, b.LayerStart, b.LayerEnd, b.GFLOPS)
	}
	return s
}

