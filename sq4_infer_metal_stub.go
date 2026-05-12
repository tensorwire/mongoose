//go:build !darwin || !cgo

package mongoose

type SQ4InferMetal struct{}

func NewSQ4InferMetal(eng *Metal) *SQ4InferMetal { return nil }
func (s *SQ4InferMetal) Build(dim, kvDim, headDim, nHeads, nKVHeads, ffnDim, vocabSize, nLayers, maxSeq int, ropeTheta, rmsEps float32) int { return -1 }
func SQ4InferReady() bool { return false }
func (s *SQ4InferMetal) AllocSlabs(totalPackedBytes, totalBandsFloats, totalOutliers int) {}
func (s *SQ4InferMetal) UploadPacked(byteOffset int, data []byte) {}
func (s *SQ4InferMetal) UploadBands(floatOffset int, data []float32) {}
func (s *SQ4InferMetal) UploadOutliers(offset int, idx []uint32, val []float32) {}
func (s *SQ4InferMetal) SetFP32(idx int, data []float32) {}
func (s *SQ4InferMetal) SetSQ4Desc(idx, packedOffset, packedBytes, bandsOffset, outlierOffset, outlierCount, rows, cols int) {}
func (s *SQ4InferMetal) Step(hidden, cosSlice, sinSlice []float32, pos int, logitsOut []float32) int { return -1 }
func (s *SQ4InferMetal) ResetKV() {}
