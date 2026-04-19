//go:build !linux || !cgo

package mongoose

import "unsafe"

type L3Bridge struct {
	Ptr  unsafe.Pointer
	Size int
}

func (c *CUDA) AllocL3Bridge(bytes int) *L3Bridge   { return nil }
func (c *CUDA) RegisterL3Bridge(bridge *L3Bridge) error { return nil }
func (b *L3Bridge) Float32(byteOffset, count int) []float32 { return nil }
func (b *L3Bridge) DevicePtr(byteOffset int) unsafe.Pointer { return nil }
