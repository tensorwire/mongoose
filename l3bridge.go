//go:build linux && cgo

package mongoose

/*
#include <stdlib.h>
#include <string.h>

// Pinned host memory — page-locked, DMA-accessible, L3-resident.
// Both CPU and GPU access through cache coherency. Zero copy.
void* tw_gpu_alloc_pinned(size_t bytes);
int tw_register_host_memory(void* ptr, size_t bytes);
*/
import "C"

import (
	"fmt"
	"log"
	"unsafe"
)

// L3Bridge is a region of pinned host memory accessible to both CPU and GPU
// through L3 cache coherency. CPU writes to slices, GPU reads via device pointer.
// No memcpy, no DMA transfer — both see the same physical pages.
//
// Usage:
//   bridge := cuda.AllocL3Bridge(4 * 1024 * 1024)  // 4MB
//   masks := bridge.Float32(0, nRows)                // CPU slice at offset 0
//   maskPtr := bridge.DevicePtr(0)                   // GPU pointer to same memory
//   // CPU writes masks[i] = 1.0
//   // GPU kernel reads maskPtr[i] — sees 1.0 through L3
type L3Bridge struct {
	Ptr  unsafe.Pointer
	Size int
}

// AllocL3Bridge allocates pinned host memory via cudaHostAlloc.
// This memory is page-locked and accessible to GPU via DMA through L3 cache.
func (c *CUDA) AllocL3Bridge(bytes int) *L3Bridge {
	ptr := C.tw_gpu_alloc_pinned(C.size_t(bytes))
	if ptr == nil {
		return nil
	}
	C.memset(ptr, 0, C.size_t(bytes))
	log.Printf("[cuda] L3 bridge: %d MB pinned host memory", bytes/(1024*1024))
	return &L3Bridge{Ptr: ptr, Size: bytes}
}

// RegisterL3Bridge registers external host memory (e.g. from Xe daemon) with CUDA.
func (c *CUDA) RegisterL3Bridge(bridge *L3Bridge) error {
	if bridge == nil {
		return nil
	}
	ret := C.tw_register_host_memory(bridge.Ptr, C.size_t(bridge.Size))
	if ret != 0 {
		return fmt.Errorf("cudaHostRegister failed: %d", ret)
	}
	log.Printf("[cuda] L3 bridge registered — %d MB", bridge.Size/(1024*1024))
	return nil
}

// Float32 returns a float32 slice view into the bridge at a byte offset.
// CPU reads/writes this slice. GPU sees the same data through L3 cache.
func (b *L3Bridge) Float32(byteOffset, count int) []float32 {
	return unsafe.Slice((*float32)(unsafe.Pointer(uintptr(b.Ptr)+uintptr(byteOffset))), count)
}

// DevicePtr returns a device-accessible pointer at a byte offset into the bridge.
// Pass this to GPU kernels — they read from L3 cache, not device memory.
func (b *L3Bridge) DevicePtr(byteOffset int) unsafe.Pointer {
	return unsafe.Pointer(uintptr(b.Ptr) + uintptr(byteOffset))
}
