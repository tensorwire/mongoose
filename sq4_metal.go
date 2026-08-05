
//go:build darwin && cgo

package mongoose

/*
#cgo LDFLAGS: -framework Metal -framework Foundation
#include <stdlib.h>

extern int mtl_sq4_init(const char* path);
extern int mtl_sq4_ready(void);
extern void mtl_sq4_matvec(void* act, void* packed, void* bands, void* out, int rows, int cols);
extern void mtl_sq4_matvec_fused(void* act, void* packed, void* bands, void* out, int rows, int cols, void* outlierIdx, void* outlierVal, int outlierCount);
extern void mtl_sq4_outlier_correct(void* outlierIdx, void* outlierVal, void* packed, void* bands, void* act, void* out, int outlierCount, int cols);

extern void* mtl_alloc(unsigned long bytes);
extern void* mtl_shared_ptr(void* buf);
*/
import "C"
import _ "unsafe"
import (
	"log"
	"os"
	"path/filepath"
	"unsafe"
)

func findSQ4Metallib() string {
	const name = "sq4_matvec.metallib"
	paths := []string{
		filepath.Join(".", name),
		filepath.Join("kernels", name),
	}
	if exe, err := os.Executable(); err == nil {
		dir := filepath.Dir(exe)
		paths = append(paths,
			filepath.Join(dir, name),
			filepath.Join(dir, "kernels", name),
		)
	}
	if home := os.Getenv("HOME"); home != "" {
		paths = append(paths,
			filepath.Join(home, "go/src/github.com/tensorwire/mongoose/kernels", name),
		)
	}
	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			if abs, err := filepath.Abs(p); err == nil {
				return abs
			}
			return p
		}
	}
	return ""
}

func SQ4MetalInit() bool {
	path := findSQ4Metallib()
	if path == "" {
		return false
	}
	cPath := (C.CString)(path)
	defer func() func() {
		_cgo0 := unsafe.Pointer(cPath)
		return func() { _cgoCheckPointer(_cgo0, nil); C.free(_cgo0) }
	}()()
	return (C.mtl_sq4_init)(cPath) != 0
}

func HasSQ4MetalKernels() bool {
	return (C.mtl_sq4_ready)() != 0
}

type SQ4Metal struct {
	eng *Metal
}

func NewSQ4Metal(eng *Metal) *SQ4Metal {
	if !SQ4MetalInit() {
		log.Println("[SQ4Metal] sq4_matvec.metallib not found")
		return nil
	}
	return &SQ4Metal{eng: eng}
}

func (s *SQ4Metal) Matvec(act, packed, bands, out *Tensor, rows, cols int) {
	func() {
		_cgo0 := MtlBufPtr(act)
		_cgo1 := MtlBufPtr(packed)
		_cgo2 := MtlBufPtr(bands)
		_cgo3 := MtlBufPtr(out)
		var _cgo4 C.int = C.int(rows)
		var _cgo5 C.int = C.int(cols)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		_cgoCheckPointer(_cgo3, nil)
		C.mtl_sq4_matvec(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5)
	}()
}

func (s *SQ4Metal) MatvecFused(act, packed, bands, out *Tensor, rows, cols int, outlierIdx, outlierVal *Tensor, outlierCount int) {
	func() {
		_cgo0 := MtlBufPtr(act)
		_cgo1 := MtlBufPtr(packed)
		_cgo2 := MtlBufPtr(bands)
		_cgo3 := MtlBufPtr(out)
		var _cgo4 C.int = C.int(rows)
		var _cgo5 C.int = C.int(cols)
		_cgo6 := MtlBufPtr(outlierIdx)
		_cgo7 := MtlBufPtr(outlierVal)
		var _cgo8 C.int = C.int(outlierCount)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		_cgoCheckPointer(_cgo3, nil)
		_cgoCheckPointer(_cgo6, nil)
		_cgoCheckPointer(_cgo7, nil)
		C.mtl_sq4_matvec_fused(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5, _cgo6, _cgo7, _cgo8)
	}()
}

func (s *SQ4Metal) OutlierCorrect(outlierIdx, outlierVal, packed, bands, act, out *Tensor, outlierCount, cols int) {
	func() {
		_cgo0 := MtlBufPtr(outlierIdx)
		_cgo1 := MtlBufPtr(outlierVal)
		_cgo2 := MtlBufPtr(packed)
		_cgo3 := MtlBufPtr(bands)
		_cgo4 := MtlBufPtr(act)
		_cgo5 := MtlBufPtr(out)
		var _cgo6 C.int = C.int(outlierCount)
		var _cgo7 C.int = C.int(cols)
		_cgoCheckPointer(_cgo0, nil)
		_cgoCheckPointer(_cgo1, nil)
		_cgoCheckPointer(_cgo2, nil)
		_cgoCheckPointer(_cgo3, nil)
		_cgoCheckPointer(_cgo4, nil)
		_cgoCheckPointer(_cgo5, nil)
		C.mtl_sq4_outlier_correct(_cgo0, _cgo1, _cgo2, _cgo3, _cgo4, _cgo5, _cgo6, _cgo7)
	}()
}
