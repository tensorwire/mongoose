//go:build linux && cgo

package mongoose

/*
#include <cuda_runtime.h>
#include <stdlib.h>

void* tw_arena_alloc(size_t bytes) {
    void* ptr;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) return NULL;
    return ptr;
}

void tw_arena_free(void* ptr) {
    cudaFree(ptr);
}

size_t tw_vram_free() {
    size_t free = 0, total = 0;
    cudaMemGetInfo(&free, &total);
    return free;
}
*/
import "C"

import (
	"log"
	"sort"
	"sync"
	"unsafe"
)

// GPUArena is a GPU memory allocator modeled after PyTorch's CUDACachingAllocator.
//
// Architecture:
//   - Allocates one large CUDA arena at init (80% of free VRAM)
//   - Sub-allocates with best-fit from a sorted free list
//   - Merges adjacent free blocks on deallocation (prevents fragmentation)
//   - Never calls cudaMalloc/cudaFree during training — just pointer math
//   - Alignment: all allocations 256-byte aligned (GPU cache line)
//
// This replaces the per-size pool that caused 22GB phantom usage on 0.5B models.
type GPUArena struct {
	base     uintptr     // arena start address
	size     int         // total arena bytes
	free     []freeBlock // sorted by address, merged on dealloc
	mu       sync.Mutex
	stats    ArenaStats
}

type freeBlock struct {
	offset int // byte offset from arena base
	size   int // bytes
}

type ArenaStats struct {
	TotalBytes     int
	UsedBytes      int
	FreeBytes      int
	AllocCount     int64
	FreeCount      int64
	MergeCount     int64
	FragmentCount  int // number of free blocks (lower = less fragmented)
}

const arenaAlign = 256 // GPU cache line alignment

// NewGPUArena allocates a CUDA arena using a fraction of available VRAM.
func NewGPUArena(fraction float64) *GPUArena {
	freeVRAM := int(C.tw_vram_free())
	arenaSize := int(float64(freeVRAM) * fraction)
	return newArena(arenaSize, freeVRAM)
}

// NewGPUArenaBytes allocates a CUDA arena of exactly the given size (rounded to MB).
func NewGPUArenaBytes(bytes int) *GPUArena {
	freeVRAM := int(C.tw_vram_free())
	return newArena(bytes, freeVRAM)
}

func newArena(arenaSize, freeVRAM int) *GPUArena {
	// Round down to MB boundary
	arenaSize = (arenaSize / (1024 * 1024)) * (1024 * 1024)
	if arenaSize < 64*1024*1024 {
		log.Printf("[arena] requested %dMB, minimum 64MB", arenaSize/(1024*1024))
		return nil
	}
	// Don't exceed 90% of free VRAM
	maxSize := freeVRAM * 90 / 100
	if arenaSize > maxSize {
		arenaSize = (maxSize / (1024 * 1024)) * (1024 * 1024)
	}

	ptr := C.tw_arena_alloc(C.size_t(arenaSize))
	if ptr == nil {
		log.Printf("[arena] cudaMalloc(%dMB) failed", arenaSize/(1024*1024))
		return nil
	}

	a := &GPUArena{
		base: uintptr(ptr),
		size: arenaSize,
		free: []freeBlock{{offset: 0, size: arenaSize}},
	}
	a.stats.TotalBytes = arenaSize
	a.stats.FreeBytes = arenaSize

	log.Printf("[arena] %dMB GPU arena (%dMB free VRAM)",
		arenaSize/(1024*1024), freeVRAM/(1024*1024))
	return a
}

// Alloc returns a GPU pointer of at least `bytes` size, 256-byte aligned.
// Uses best-fit search over the free list. O(n) in free blocks but n is
// typically small (<100) due to merging.
func (a *GPUArena) Alloc(bytes int) unsafe.Pointer {
	// Round up to alignment
	bytes = (bytes + arenaAlign - 1) & ^(arenaAlign - 1)

	a.mu.Lock()
	defer a.mu.Unlock()

	// Best-fit: find smallest free block that fits
	bestIdx := -1
	bestSize := int(1<<62)
	for i, b := range a.free {
		if b.size >= bytes && b.size < bestSize {
			bestIdx = i
			bestSize = b.size
			if b.size == bytes {
				break // exact match
			}
		}
	}

	if bestIdx < 0 {
		// No block large enough — arena exhausted
		log.Printf("[arena] OOM: need %dMB, largest free block %dMB, %d fragments",
			bytes/(1024*1024), a.largestFree()/(1024*1024), len(a.free))
		return nil
	}

	block := a.free[bestIdx]
	ptr := unsafe.Pointer(a.base + uintptr(block.offset))

	// Split or consume the free block
	if block.size == bytes {
		// Exact fit — remove block
		a.free = append(a.free[:bestIdx], a.free[bestIdx+1:]...)
	} else {
		// Split — shrink the free block
		a.free[bestIdx].offset += bytes
		a.free[bestIdx].size -= bytes
	}

	a.stats.UsedBytes += bytes
	a.stats.FreeBytes -= bytes
	a.stats.AllocCount++
	a.stats.FragmentCount = len(a.free)
	return ptr
}

// Free returns a GPU allocation to the arena. Merges with adjacent free blocks.
func (a *GPUArena) Free(ptr unsafe.Pointer, bytes int) {
	bytes = (bytes + arenaAlign - 1) & ^(arenaAlign - 1)
	offset := int(uintptr(ptr) - a.base)

	a.mu.Lock()
	defer a.mu.Unlock()

	// Insert into sorted free list and merge neighbors
	newBlock := freeBlock{offset: offset, size: bytes}

	// Find insertion point (sorted by offset)
	idx := sort.Search(len(a.free), func(i int) bool {
		return a.free[i].offset > offset
	})

	// Insert
	a.free = append(a.free, freeBlock{})
	copy(a.free[idx+1:], a.free[idx:])
	a.free[idx] = newBlock

	// Merge with right neighbor
	if idx+1 < len(a.free) && a.free[idx].offset+a.free[idx].size == a.free[idx+1].offset {
		a.free[idx].size += a.free[idx+1].size
		a.free = append(a.free[:idx+1], a.free[idx+2:]...)
		a.stats.MergeCount++
	}

	// Merge with left neighbor
	if idx > 0 && a.free[idx-1].offset+a.free[idx-1].size == a.free[idx].offset {
		a.free[idx-1].size += a.free[idx].size
		a.free = append(a.free[:idx], a.free[idx+1:]...)
		a.stats.MergeCount++
	}

	a.stats.UsedBytes -= bytes
	a.stats.FreeBytes += bytes
	a.stats.FreeCount++
	a.stats.FragmentCount = len(a.free)
}

// AllocFloats allocates n float32s (n*4 bytes, aligned).
func (a *GPUArena) AllocFloats(n int) unsafe.Pointer {
	return a.Alloc(n * 4)
}

// FreeFloats returns n float32s to the arena.
func (a *GPUArena) FreeFloats(ptr unsafe.Pointer, n int) {
	a.Free(ptr, n*4)
}

// Stats returns current arena statistics.
func (a *GPUArena) Stats() ArenaStats {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.stats
}

func (a *GPUArena) largestFree() int {
	max := 0
	for _, b := range a.free {
		if b.size > max {
			max = b.size
		}
	}
	return max
}

// Destroy releases the entire arena back to CUDA.
func (a *GPUArena) Destroy() {
	if a.base != 0 {
		C.tw_arena_free(unsafe.Pointer(a.base))
		a.base = 0
		log.Printf("[arena] destroyed (%d allocs, %d frees, %d merges)",
			a.stats.AllocCount, a.stats.FreeCount, a.stats.MergeCount)
	}
}
