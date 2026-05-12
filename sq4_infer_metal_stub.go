//go:build !darwin || !cgo

package mongoose

type SQ4InferMetal struct{}

func NewSQ4InferMetal(eng *Metal) *SQ4InferMetal                          { return nil }
func (s *SQ4InferMetal) Build(dim, kvDim, headDim, nHeads, nKVHeads, ffnDim, vocabSize, nLayers, maxSeq int, ropeTheta, rmsEps float32) int { return -1 }
func SQ4InferReady() bool                                                  { return false }
func (s *SQ4InferMetal) AllocSlabs(totalPackedBytes, totalBandsFloats, totalOutliers int) {}
func (s *SQ4InferMetal) FinalizeSlabs()                                    {}
func (s *SQ4InferMetal) UploadPacked(packedOffset int, mag []byte, sign []byte, nWeights int) {}
func (s *SQ4InferMetal) UploadBands(floatOffset int, data []float32)       {}
func (s *SQ4InferMetal) UploadOutliers(offset int, idx []uint32, val []float32) {}
func (s *SQ4InferMetal) UploadEmbed(data []float32)                        {}
func (s *SQ4InferMetal) SetFP32(idx int, data []float32)                   {}
func (s *SQ4InferMetal) SetSQ4Desc(idx, packedOffset, bandsOffset, outlierOffset, outlierCount, rows, cols int) {}
func (s *SQ4InferMetal) Step(tokenID, pos int, logitsOut []float32) int    { return -1 }
func (s *SQ4InferMetal) StepSample(tokenID, pos int) int                   { return -1 }
func (s *SQ4InferMetal) Prefill(tokenIDs []int, logitsOut []float32)       {}
func (s *SQ4InferMetal) ResetKV()                                          {}
