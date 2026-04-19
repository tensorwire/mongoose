package mongoose

import "unsafe"

// L3Bridge is a region of pinned host memory accessible to both CPU and GPU
// through L3 cache coherency. CPU writes to slices, GPU reads via device pointer.
// No memcpy, no DMA transfer — both see the same physical pages.
type L3Bridge struct {
	Ptr  unsafe.Pointer
	Size int
}

// Float32 returns a float32 slice view into the bridge at a byte offset.
func (b *L3Bridge) Float32(byteOffset, count int) []float32 {
	if b == nil || b.Ptr == nil {
		return make([]float32, count)
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(uintptr(b.Ptr)+uintptr(byteOffset))), count)
}

// DevicePtr returns a device-accessible pointer at a byte offset.
func (b *L3Bridge) DevicePtr(byteOffset int) unsafe.Pointer {
	if b == nil || b.Ptr == nil {
		return nil
	}
	return unsafe.Pointer(uintptr(b.Ptr) + uintptr(byteOffset))
}
