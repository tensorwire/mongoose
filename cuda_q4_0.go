//go:build linux && cgo

package mongoose

/*
void tw_gpu_upload(void* dst, const void* src, size_t bytes);
void* tw_gpu_alloc(size_t bytes);
*/
import "C"

import "unsafe"

// UploadQ4_0 uploads raw GGUF block_q4_0 weight data to GPU memory.
// Format: nBlocks*18 bytes per row, where nBlocks = cols/32.
func (eng *CUDA) UploadQ4_0(q4Data []byte, rows, cols int) *Tensor {
	nBlocks := cols / 32
	bytesPerRow := nBlocks * 18
	totalBytes := rows * bytesPerRow

	gpuPtr := C.tw_gpu_alloc(C.size_t(totalBytes))
	C.tw_gpu_upload(gpuPtr, unsafe.Pointer(&q4Data[0]), C.size_t(totalBytes))

	t := TensorFromDevicePtr(gpuPtr, totalBytes)
	t.eng = eng
	return t
}
