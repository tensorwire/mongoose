//go:build darwin && cgo

package mongoose

/*
#include <stdlib.h>
#include <string.h>

typedef void* MTLBufferRef;

MTLBufferRef mtl_alloc(size_t bytes);
void mtl_free(MTLBufferRef buf);
void mtl_upload(MTLBufferRef buf, const void* src, size_t bytes);
void mtl_download(void* dst, MTLBufferRef buf, size_t bytes);
void mtl_zero(MTLBufferRef buf, size_t bytes);
int mtl_sgemm(MTLBufferRef A, MTLBufferRef B, MTLBufferRef C, int m, int k, int n);
int mtl_sgemm_transB(MTLBufferRef A, MTLBufferRef B, MTLBufferRef C, int m, int k, int n);
int mtl_sgemm_transA(MTLBufferRef A, MTLBufferRef B, MTLBufferRef C, int m, int k, int n);
void* mtl_shared_ptr(MTLBufferRef buf);
int mtl_graph_sgemm(void* aRef, void* bRef, void* cRef, int m, int k, int n, int transA, int transB);
*/
import "C"

import (
	"unsafe"
)

type mtlPtr struct {
	buf C.MTLBufferRef
}

func (p *mtlPtr) RawPtr() unsafe.Pointer {
	return unsafe.Pointer(p.buf)
}

func (m *Metal) FromHost(data []float32, shape []int) *Tensor {
	size := 1
	for _, s := range shape {
		size *= s
	}
	buf := C.mtl_alloc(C.size_t(size * 4))
	C.mtl_upload(buf, unsafe.Pointer(&data[0]), C.size_t(size*4))
	return &Tensor{
		Shape:  shape,
		Size:   size,
		device: &mtlPtr{buf: buf},
		eng:    m,
	}
}

func (m *Metal) Zeros(shape []int) *Tensor {
	size := 1
	for _, s := range shape {
		size *= s
	}
	buf := C.mtl_alloc(C.size_t(size * 4))
	C.mtl_zero(buf, C.size_t(size*4))
	return &Tensor{
		Shape:  shape,
		Size:   size,
		device: &mtlPtr{buf: buf},
		eng:    m,
	}
}

func (m *Metal) ToHost(t *Tensor) []float32 {
	mp := t.device.(*mtlPtr)
	data := make([]float32, t.Size)
	C.mtl_download(unsafe.Pointer(&data[0]), mp.buf, C.size_t(t.Size*4))
	return data
}

func (m *Metal) Release(t *Tensor) {
	if t.device != nil {
		mp := t.device.(*mtlPtr)
		C.mtl_free(mp.buf)
		t.device = nil
	}
}

// AllocRaw allocates an MTLBuffer of exactly nBytes, zeroed. Returns a Tensor
// with Size = nElements (caller provides this for dispatch sizing).
func (m *Metal) AllocRaw(nBytes, nElements int, shape []int) *Tensor {
	buf := C.mtl_alloc(C.size_t(nBytes))
	C.mtl_zero(buf, C.size_t(nBytes))
	return &Tensor{Shape: shape, Size: nElements, device: &mtlPtr{buf: buf}, eng: m}
}

// UploadRaw uploads arbitrary bytes into a Tensor's underlying MTLBuffer.
func (m *Metal) UploadRaw(t *Tensor, data unsafe.Pointer, nBytes int) {
	mp := t.device.(*mtlPtr)
	C.mtl_upload(mp.buf, data, C.size_t(nBytes))
}

// MatMulT computes C = A @ B. A[m,k], B[k,n] → C[m,n].
func (m *Metal) MatMulT(a, b *Tensor, rows, k, n int) *Tensor {
	size := rows * n
	bufC := C.mtl_alloc(C.size_t(size * 4))
	C.mtl_sgemm(
		a.device.(*mtlPtr).buf,
		b.device.(*mtlPtr).buf,
		bufC,
		C.int(rows), C.int(k), C.int(n))
	return &Tensor{Shape: []int{rows, n}, Size: size, device: &mtlPtr{buf: bufC}, eng: m}
}

// MatMulTransposeBT computes C[m,n] = A[m,k] @ B[n,k]^T.
func (m *Metal) MatMulTransposeBT(a, b *Tensor, rows, k, n int) *Tensor {
	size := rows * n
	bufC := C.mtl_alloc(C.size_t(size * 4))
	C.mtl_sgemm_transB(
		a.device.(*mtlPtr).buf,
		b.device.(*mtlPtr).buf,
		bufC,
		C.int(rows), C.int(k), C.int(n))
	return &Tensor{Shape: []int{rows, n}, Size: size, device: &mtlPtr{buf: bufC}, eng: m}
}

// MatMulTransposeAT computes C[k,n] = A[m,k]^T @ B[m,n].
func (m *Metal) MatMulTransposeAT(a, b *Tensor, rows, k, n int) *Tensor {
	size := k * n
	bufC := C.mtl_alloc(C.size_t(size * 4))
	C.mtl_sgemm_transA(
		a.device.(*mtlPtr).buf,
		b.device.(*mtlPtr).buf,
		bufC,
		C.int(rows), C.int(k), C.int(n))
	return &Tensor{Shape: []int{k, n}, Size: size, device: &mtlPtr{buf: bufC}, eng: m}
}

// AddInPlace computes a += b on GPU.
func (m *Metal) AddInPlace(a, b *Tensor) {
	aData := m.ToHost(a)
	bData := m.ToHost(b)
	for i := range aData {
		aData[i] += bData[i]
	}
	C.mtl_upload(a.device.(*mtlPtr).buf, unsafe.Pointer(&aData[0]), C.size_t(a.Size*4))
}

func (m *Metal) AddT(a, b *Tensor) *Tensor {
	aData := m.ToHost(a)
	bData := m.ToHost(b)
	out := make([]float32, a.Size)
	for i := range out {
		out[i] = aData[i] + bData[i]
	}
	return m.FromHost(out, a.Shape)
}

func (m *Metal) ScaleT(a *Tensor, s float32) *Tensor {
	data := m.ToHost(a)
	for i := range data {
		data[i] *= s
	}
	return m.FromHost(data, a.Shape)
}

func (m *Metal) ReLUT(a *Tensor) *Tensor {
	data := m.ToHost(a)
	for i := range data {
		if data[i] < 0 {
			data[i] = 0
		}
	}
	return m.FromHost(data, a.Shape)
}

func (m *Metal) ReLUBackwardT(dOut, fwdInput *Tensor) *Tensor {
	dData := m.ToHost(dOut)
	fData := m.ToHost(fwdInput)
	for i := range dData {
		if fData[i] <= 0 {
			dData[i] = 0
		}
	}
	return m.FromHost(dData, dOut.Shape)
}

func (m *Metal) TransposeT(a *Tensor, rows, cols int) *Tensor {
	data := m.ToHost(a)
	out := make([]float32, len(data))
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			out[j*rows+i] = data[i*cols+j]
		}
	}
	return m.FromHost(out, []int{cols, rows})
}

func (m *Metal) CopyT(src *Tensor) *Tensor {
	data := m.ToHost(src)
	return m.FromHost(data, src.Shape)
}

// MatVecResidentW computes out = W @ x where W is a pre-uploaded GPU tensor
// and x is a host slice. Uses the MPSGraph cached path for minimal dispatch overhead.
func (m *Metal) MatVecResidentW(W *Tensor, x []float32, rows, cols int) []float32 {
	mp := W.device.(*mtlPtr)
	bufX := m.poolGet(cols)
	bufC := m.poolGet(rows)

	ptrX := C.mtl_shared_ptr(bufX)
	C.memcpy(ptrX, unsafe.Pointer(&x[0]), C.size_t(cols*4))

	C.mtl_graph_sgemm(unsafe.Pointer(mp.buf), unsafe.Pointer(bufX), unsafe.Pointer(bufC),
		C.int(rows), C.int(cols), C.int(1), C.int(0), C.int(0))

	out := make([]float32, rows)
	ptrC := C.mtl_shared_ptr(bufC)
	C.memcpy(unsafe.Pointer(&out[0]), ptrC, C.size_t(rows*4))

	m.poolPut(cols, bufX)
	m.poolPut(rows, bufC)
	return out
}

// MtlBufPtr extracts the raw MTLBuffer pointer from a Tensor for use with fused compute kernels.
func MtlBufPtr(t *Tensor) unsafe.Pointer {
	return unsafe.Pointer(t.device.(*mtlPtr).buf)
}

// UploadInto overwrites an existing Metal tensor's buffer with new float32 data.
// Uses direct memcpy to shared memory — no allocation, no command buffer needed.
func (m *Metal) UploadInto(dst *Tensor, data []float32) {
	C.mtl_upload(dst.device.(*mtlPtr).buf, unsafe.Pointer(&data[0]), C.size_t(len(data)*4))
}

// SharedPtr returns the raw CPU-accessible pointer for a StorageModeShared GPU buffer.
// On Apple Silicon unified memory this is the same physical pages the GPU reads —
// writes here are visible to the GPU with no explicit copy or flush.
func (m *Metal) SharedPtr(t *Tensor) unsafe.Pointer {
	return C.mtl_shared_ptr(t.device.(*mtlPtr).buf)
}
