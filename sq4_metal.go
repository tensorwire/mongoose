//go:build darwin && cgo

package mongoose

/*
#cgo LDFLAGS: -framework Metal -framework Foundation
#include <stdlib.h>

extern int mtl_sq4_init(const char* path);
extern int mtl_sq4_ready(void);
extern void mtl_sq4_matvec(void* act, void* packed, void* bands, void* out, int rows, int cols);
extern void mtl_sq4_outlier_correct(void* outlierIdx, void* outlierVal, void* packed, void* bands, void* act, void* out, int outlierCount, int cols);

extern void* mtl_alloc(unsigned long bytes);
extern void* mtl_shared_ptr(void* buf);
*/
import "C"
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
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	return C.mtl_sq4_init(cPath) != 0
}

func HasSQ4MetalKernels() bool {
	return C.mtl_sq4_ready() != 0
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
	C.mtl_sq4_matvec(MtlBufPtr(act), MtlBufPtr(packed), MtlBufPtr(bands), MtlBufPtr(out),
		C.int(rows), C.int(cols))
}

func (s *SQ4Metal) OutlierCorrect(outlierIdx, outlierVal, packed, bands, act, out *Tensor, outlierCount, cols int) {
	C.mtl_sq4_outlier_correct(MtlBufPtr(outlierIdx), MtlBufPtr(outlierVal),
		MtlBufPtr(packed), MtlBufPtr(bands), MtlBufPtr(act), MtlBufPtr(out),
		C.int(outlierCount), C.int(cols))
}
