//go:build darwin && cgo

package mongoose

/*
#cgo LDFLAGS: -framework Metal -framework Foundation
#include <stdlib.h>
#include <stdint.h>

extern int mtl_sq4_infer_build(int dim, int kvDim, int headDim,
    int nHeads, int nKVHeads, int ffnDim,
    int vocabSize, int nLayers, int maxSeq,
    float ropeTheta, float rmsEps);
extern int mtl_sq4_infer_ready(void);
extern void mtl_sq4_infer_alloc_slabs(int totalPackedBytes, int totalBandsFloats, int totalOutliers);
extern void mtl_sq4_infer_upload_packed(int byteOffset, const uint8_t* data, int nBytes);
extern void mtl_sq4_infer_upload_bands(int floatOffset, const float* data, int nFloats);
extern void mtl_sq4_infer_upload_outliers(int offset, const uint32_t* idx, const float* val, int count);
extern void mtl_sq4_infer_set_fp32(int idx, const float* data, int nFloats);
extern void mtl_sq4_infer_set_sq4_desc(int idx, int packedOffset, int packedBytes,
    int bandsOffset, int outlierOffset, int outlierCount, int rows, int cols);
extern int mtl_sq4_infer_step(const float* hiddenIn, const float* cosData, const float* sinData,
    int pos, float* logitsOut);
extern void mtl_sq4_infer_reset_kv(void);
*/
import "C"
import "unsafe"

type SQ4InferMetal struct {
	eng *Metal
}

func NewSQ4InferMetal(eng *Metal) *SQ4InferMetal {
	return &SQ4InferMetal{eng: eng}
}

func (s *SQ4InferMetal) Build(dim, kvDim, headDim, nHeads, nKVHeads, ffnDim, vocabSize, nLayers, maxSeq int,
	ropeTheta, rmsEps float32) int {
	return int(C.mtl_sq4_infer_build(C.int(dim), C.int(kvDim), C.int(headDim),
		C.int(nHeads), C.int(nKVHeads), C.int(ffnDim),
		C.int(vocabSize), C.int(nLayers), C.int(maxSeq),
		C.float(ropeTheta), C.float(rmsEps)))
}

func SQ4InferReady() bool {
	return C.mtl_sq4_infer_ready() != 0
}

func (s *SQ4InferMetal) AllocSlabs(totalPackedBytes, totalBandsFloats, totalOutliers int) {
	C.mtl_sq4_infer_alloc_slabs(C.int(totalPackedBytes), C.int(totalBandsFloats), C.int(totalOutliers))
}

func (s *SQ4InferMetal) UploadPacked(byteOffset int, data []byte) {
	C.mtl_sq4_infer_upload_packed(C.int(byteOffset), (*C.uint8_t)(unsafe.Pointer(&data[0])), C.int(len(data)))
}

func (s *SQ4InferMetal) UploadBands(floatOffset int, data []float32) {
	C.mtl_sq4_infer_upload_bands(C.int(floatOffset), (*C.float)(unsafe.Pointer(&data[0])), C.int(len(data)))
}

func (s *SQ4InferMetal) UploadOutliers(offset int, idx []uint32, val []float32) {
	if len(idx) == 0 {
		return
	}
	C.mtl_sq4_infer_upload_outliers(C.int(offset),
		(*C.uint32_t)(unsafe.Pointer(&idx[0])),
		(*C.float)(unsafe.Pointer(&val[0])),
		C.int(len(idx)))
}

func (s *SQ4InferMetal) SetFP32(idx int, data []float32) {
	C.mtl_sq4_infer_set_fp32(C.int(idx), (*C.float)(unsafe.Pointer(&data[0])), C.int(len(data)))
}

func (s *SQ4InferMetal) SetSQ4Desc(idx, packedOffset, packedBytes, bandsOffset, outlierOffset, outlierCount, rows, cols int) {
	C.mtl_sq4_infer_set_sq4_desc(C.int(idx), C.int(packedOffset), C.int(packedBytes),
		C.int(bandsOffset), C.int(outlierOffset), C.int(outlierCount), C.int(rows), C.int(cols))
}

func (s *SQ4InferMetal) Step(hidden, cosSlice, sinSlice []float32, pos int, logitsOut []float32) int {
	return int(C.mtl_sq4_infer_step(
		(*C.float)(unsafe.Pointer(&hidden[0])),
		(*C.float)(unsafe.Pointer(&cosSlice[0])),
		(*C.float)(unsafe.Pointer(&sinSlice[0])),
		C.int(pos),
		(*C.float)(unsafe.Pointer(&logitsOut[0]))))
}

func (s *SQ4InferMetal) ResetKV() {
	C.mtl_sq4_infer_reset_kv()
}
