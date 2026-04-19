//go:build !linux || !cgo

package mongoose

import "unsafe"

type Xe struct{}

func NewXe() *Xe                                     { return nil }
func (x *Xe) Name() string                           { return "xe/unavailable" }
func (x *Xe) MatMul(a, b []float32, m, k, n int) []float32 { return nil }
func (x *Xe) RMSNorm(data, weight []float32, eps float32) {}
func (x *Xe) SoftMax(data []float32, n int)           {}
func (x *Xe) ReLU(data []float32)                     {}
func (x *Xe) VRAM() uint64                            { return 0 }
func (x *Xe) Benchmark() float64                      { return 0 }
func (x *Xe) Sync()                                   {}
func XeSharedAlloc(bytes int) unsafe.Pointer           { return nil }
func XeFree(ptr unsafe.Pointer)                        {}
func XeInitialized() bool                              { return false }
func XeLoadSPIRV(spirvData []byte, kernelName string) int { return -1 }
func XeDispatchRMSNorm(shaderIdx int, x, out, weight unsafe.Pointer, dim, seqLen int, eps float32) {}
func XeDispatchSiLUGate(shaderIdx int, gate, up, out unsafe.Pointer, n int) {}
func XeDispatchAddInPlace(shaderIdx int, a, b unsafe.Pointer, n int) {}
func XeSync()                                          {}
