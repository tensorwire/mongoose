//go:build !linux || !cgo

package mongoose

import "unsafe"

type GPUArena struct{}
type ArenaStats struct {
	TotalBytes, UsedBytes, FreeBytes int
	AllocCount, FreeCount, MergeCount int64
	FragmentCount int
}

func NewGPUArena(fraction float64) *GPUArena { return nil }
func NewGPUArenaBytes(bytes int) *GPUArena { return nil }
func (a *GPUArena) Alloc(bytes int) unsafe.Pointer { return nil }
func (a *GPUArena) Free(ptr unsafe.Pointer, bytes int) {}
func (a *GPUArena) AllocFloats(n int) unsafe.Pointer { return nil }
func (a *GPUArena) FreeFloats(ptr unsafe.Pointer, n int) {}
func (a *GPUArena) Stats() ArenaStats { return ArenaStats{} }
func (a *GPUArena) Destroy() {}
