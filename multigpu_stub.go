//go:build !linux || !cgo

package mongoose

import "unsafe"

type MultiCUDA struct{ DeviceCount int }
type DeviceInfo struct {
	ID, SMCount, ComputeMajor, ComputeMinor int
	Name                                     string
	TotalMem, FreeMem                        uint64
	CanPeer                                  []bool
}
type DeviceTensor struct {
	DeviceID int
	Ptr      unsafe.Pointer
	Shape    []int
	Size     int
}

func NewMultiCUDA() *MultiCUDA                                       { return nil }
func (mc *MultiCUDA) Device(id int) DeviceInfo                       { return DeviceInfo{} }
func (mc *MultiCUDA) Alloc(dev, size int) unsafe.Pointer             { return nil }
func (mc *MultiCUDA) Free(dev, size int, ptr unsafe.Pointer)         {}
func (mc *MultiCUDA) Upload(dev int, data []float32) *DeviceTensor   { return nil }
func (mc *MultiCUDA) Download(dt *DeviceTensor) []float32            { return nil }
func (mc *MultiCUDA) Release(dt *DeviceTensor)                       {}
func (mc *MultiCUDA) PeerCopy(src *DeviceTensor, dst int) *DeviceTensor { return nil }
func (mc *MultiCUDA) MatMulOnDevice(dev int, a, b unsafe.Pointer, m, k, n int) unsafe.Pointer {
	return nil
}
func (mc *MultiCUDA) MatMulFP16OnDevice(dev int, a, b unsafe.Pointer, m, k, n int) unsafe.Pointer {
	return nil
}
func (mc *MultiCUDA) SyncDevice(dev int) {}
func (mc *MultiCUDA) SyncAll()           {}
func (mc *MultiCUDA) ParallelMatMul(ops []ParallelOp) {}
func (mc *MultiCUDA) ParallelMatMulBatch(ops []ParallelOp, iters int) {}
func (mc *MultiCUDA) String() string     { return "MultiCUDA: unavailable" }

type ParallelOp struct {
	DeviceID   int
	A, B, C    unsafe.Pointer
	M, K, N    int
}
