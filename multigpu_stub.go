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
func (mc *MultiCUDA) PeerCopyInto(srcDev int, src unsafe.Pointer, dstDev int, dst unsafe.Pointer, bytes int) {}
func (mc *MultiCUDA) PeerCopy(src *DeviceTensor, dst int) *DeviceTensor { return nil }
func (mc *MultiCUDA) MatMulOnDevice(dev int, a, b unsafe.Pointer, m, k, n int) unsafe.Pointer {
	return nil
}
func (mc *MultiCUDA) MatMulFP16TransBOnDevice(dev int, a, b, out unsafe.Pointer, m, k, n int) {}
func (mc *MultiCUDA) MatMulFP16FP32OutOnDevice(dev int, a, b unsafe.Pointer, out unsafe.Pointer, m, k, n int) {}
func (mc *MultiCUDA) MatMulFP16TransAFP32OutOnDevice(dev int, a, b unsafe.Pointer, out unsafe.Pointer, m, k, n int) {}
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

type ParallelFP16Op struct {
	DeviceID   int
	A, B, C    unsafe.Pointer
	M, K, N    int
	TransB     bool
}

func (mc *MultiCUDA) ParallelMatMulFP16(ops []ParallelFP16Op) {}
func SetDevice(dev int)                                                                              {}
func (mc *MultiCUDA) AllocFP16OnDevice(dev, nElements int) unsafe.Pointer                            { return nil }
func (mc *MultiCUDA) AllocBytesOnDevice(dev, bytes int) unsafe.Pointer                               { return nil }
func (mc *MultiCUDA) ZerosFP32OnDevice(dev, nFloats int) *Tensor                                     { return nil }
func (mc *MultiCUDA) ZeroOnDevice(dev int, ptr unsafe.Pointer, bytes int)                             {}
func (mc *MultiCUDA) UploadFP32OnDevice(dev int, dst unsafe.Pointer, src []float32)                   {}
func (mc *MultiCUDA) DownloadFP32FromDevice(dev int, src unsafe.Pointer, nFloats int) []float32       { return nil }
func (mc *MultiCUDA) FromHostFP32OnDevice(dev int, data []float32, shape []int) *Tensor               { return nil }
func (mc *MultiCUDA) FromHostFP16OnDevice(dev int, data []float32, shape []int, f func([]float32, []int) *Tensor) *Tensor { return nil }
func (mc *MultiCUDA) SparsePeerCopy(srcDev int, src unsafe.Pointer, dstDev int, dst unsafe.Pointer, rows []int32, cols, elemSize int) {}
func (mc *MultiCUDA) PeerCopyAsync(srcDev int, src unsafe.Pointer, dstDev int, dst unsafe.Pointer, bytes int) {}
