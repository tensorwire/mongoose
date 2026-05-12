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
extern void mtl_sq4_infer_alloc_slabs(long long totalPackedBytes,
    int totalBandsFloats, int totalOutliers);
extern void mtl_sq4_infer_upload_packed(long long packedOffset, const uint8_t* mag, int magBytes,
    const uint8_t* sign_data, int signBytes, int nWeights);
extern void mtl_sq4_infer_finalize_slabs(void);
extern void mtl_sq4_infer_upload_embed(const float* data, int nFloats);
extern void mtl_sq4_infer_upload_bands(int floatOffset, const float* data, int nFloats);
extern void mtl_sq4_infer_upload_outliers(int offset, const uint32_t* idx, const float* val, int count);
extern void mtl_sq4_infer_set_fp32(int idx, const float* data, int nFloats);
extern void mtl_sq4_infer_set_sq4_desc(int idx, long long packedOffset,
    int bandsOffset, int outlierOffset, int outlierCount, int rows, int cols);
extern int mtl_sq4_infer_step(int tokenID, int pos, float* logitsOut);
extern int mtl_sq4_infer_step_sample(int tokenID, int pos);
extern int mtl_sq4_infer_prefill(const int* tokenIDs, int nTokens, float* logitsOut);
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
	C.mtl_sq4_infer_alloc_slabs(C.longlong(totalPackedBytes), C.int(totalBandsFloats), C.int(totalOutliers))
}

func (s *SQ4InferMetal) FinalizeSlabs() {
	C.mtl_sq4_infer_finalize_slabs()
}

func (s *SQ4InferMetal) UploadEmbed(data []float32) {
	C.mtl_sq4_infer_upload_embed((*C.float)(unsafe.Pointer(&data[0])), C.int(len(data)))
}

// UploadPacked takes spec-format mag+sign and packs into nibbles on CPU, uploads to GPU slab.
func (s *SQ4InferMetal) UploadPacked(packedOffset int, mag []byte, sign []byte, nWeights int) {
	C.mtl_sq4_infer_upload_packed(C.longlong(packedOffset),
		(*C.uint8_t)(unsafe.Pointer(&mag[0])), C.int(len(mag)),
		(*C.uint8_t)(unsafe.Pointer(&sign[0])), C.int(len(sign)),
		C.int(nWeights))
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

func (s *SQ4InferMetal) SetSQ4Desc(idx, packedOffset, bandsOffset, outlierOffset, outlierCount, rows, cols int) {
	C.mtl_sq4_infer_set_sq4_desc(C.int(idx), C.longlong(packedOffset),
		C.int(bandsOffset), C.int(outlierOffset), C.int(outlierCount), C.int(rows), C.int(cols))
}

func (s *SQ4InferMetal) Step(tokenID, pos int, logitsOut []float32) int {
	return int(C.mtl_sq4_infer_step(
		C.int(tokenID), C.int(pos),
		(*C.float)(unsafe.Pointer(&logitsOut[0]))))
}

// StepSample does forward + argmax on GPU. Returns sampled token ID. No D2H logits copy.
func (s *SQ4InferMetal) StepSample(tokenID, pos int) int {
	return int(C.mtl_sq4_infer_step_sample(C.int(tokenID), C.int(pos)))
}

// Prefill processes all prompt tokens in one batched pass using Metal 4 GEMM.
// Returns logits for the last token.
func (s *SQ4InferMetal) Prefill(tokenIDs []int, logitsOut []float32) int {
	if len(tokenIDs) == 0 {
		return -1
	}
	cTokens := make([]C.int, len(tokenIDs))
	for i, t := range tokenIDs {
		cTokens[i] = C.int(t)
	}
	return int(C.mtl_sq4_infer_prefill((*C.int)(unsafe.Pointer(&cTokens[0])),
		C.int(len(tokenIDs)),
		(*C.float)(unsafe.Pointer(&logitsOut[0]))))
}

func (s *SQ4InferMetal) ResetKV() {
	C.mtl_sq4_infer_reset_kv()
}
