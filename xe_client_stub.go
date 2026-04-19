//go:build !linux

package mongoose

import "unsafe"

type XeDaemon struct{}

func NewXeDaemon() *XeDaemon                    { return nil }
func (d *XeDaemon) Close()                      {}
func (d *XeDaemon) Name() string                { return "xe/unavailable" }
func (d *XeDaemon) VRAM() uint64                { return 0 }
func (d *XeDaemon) LoadSPIRV(path, ep string) int { return -1 }
func (d *XeDaemon) Alloc(bytes int) unsafe.Pointer { return nil }
func (d *XeDaemon) Free(ptr unsafe.Pointer)      {}
func (d *XeDaemon) DispatchRMSNorm(ki int, x, out, w unsafe.Pointer, dim, seq int, eps float32) {}
func (d *XeDaemon) DispatchSiLU(ki int, gate, up, out unsafe.Pointer, n int)                    {}
func (d *XeDaemon) DispatchAdd(ki int, a, b unsafe.Pointer, n int)                              {}
func (d *XeDaemon) Sync()                       {}
func (d *XeDaemon) HasArena() bool               { return false }
func (d *XeDaemon) GoRegion(off, n int) []float32     { return nil }
func (d *XeDaemon) GoRegionInt32(off, n int) []int32   { return nil }
func (d *XeDaemon) XeRegion(off, n int) []float32     { return nil }
func (d *XeDaemon) DispatchCrossEntropy(ki int, lo, to, oo, go2 uint32, np, vs uint32, in2 float32) {}
