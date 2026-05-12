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
extern void mtl_sq4_infer_set_fp32(int idx, const float* data, int nFloats);
extern void mtl_sq4_infer_set_sq4(int idx, const uint8_t* packed, int packedBytes,
    const float* bands, const uint32_t* outlierIdx, const float* outlierVal,
    int outlierCount, int rows, int cols);
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

func (s *SQ4InferMetal) SetFP32(idx int, data []float32) {
	C.mtl_sq4_infer_set_fp32(C.int(idx), (*C.float)(unsafe.Pointer(&data[0])), C.int(len(data)))
}

func (s *SQ4InferMetal) SetSQ4(idx int, packed []byte, bands [8]float32,
	outlierIdx []uint32, outlierVal []float32, rows, cols int) {
	var oIdx *C.uint32_t
	var oVal *C.float
	oc := len(outlierIdx)
	if oc > 0 {
		oIdx = (*C.uint32_t)(unsafe.Pointer(&outlierIdx[0]))
		oVal = (*C.float)(unsafe.Pointer(&outlierVal[0]))
	}
	C.mtl_sq4_infer_set_sq4(C.int(idx),
		(*C.uint8_t)(unsafe.Pointer(&packed[0])), C.int(len(packed)),
		(*C.float)(unsafe.Pointer(&bands[0])),
		oIdx, oVal, C.int(oc), C.int(rows), C.int(cols))
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
