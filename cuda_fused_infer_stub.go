//go:build !linux || !cgo

package mongoose

import "unsafe"

type CUDAFusedInference struct{
	WeightsFP16 bool
}

func NewCUDAFusedInference(eng *CUDA, dim, kvDim, headDim, nHeads, nKVHeads, ffnDim, vocabSize, nLayers, maxSeq int, ropeTheta float64) *CUDAFusedInference {
	return nil
}
func (f *CUDAFusedInference) NumWeights() int { return 0 }
func (f *CUDAFusedInference) SetWeight(idx int, data []float32) {}
func (f *CUDAFusedInference) SetWeightQ4_0(idx int, q4Data []byte) {}
func (f *CUDAFusedInference) SetWeightFP16(idx int, data []float32) {}
func (f *CUDAFusedInference) SetWeightRawFP16(idx int, fp16Bytes []byte, nElems int, normData []float32) {}
func (f *CUDAFusedInference) StepLayerResident(layer, pos int, hiddenIn, preOut, postOut, logitsOut []float32) {}
func (f *CUDAFusedInference) StepLayerResidentFP16(layer, pos int, hiddenIn, preOut, postOut, logitsOut []float32) {}
func (f *CUDAFusedInference) PartialStepFP16(hiddenIn []float32, pos, layerStart, layerEnd int, hiddenOut, logitsOut []float32) int { return -1 }
func (f *CUDAFusedInference) PartialStep(hiddenIn []float32, pos, layerStart, layerEnd int, hiddenOut, logitsOut []float32) int { return -1 }
func (f *CUDAFusedInference) PartialStepContinue(pos, layerStart, layerEnd int, hiddenOut, logitsOut []float32) int { return -1 }
func (f *CUDAFusedInference) PartialStepQ4(hiddenIn []float32, pos, layerStart, layerEnd int, hiddenOut, logitsOut []float32) int { return -1 }
func (f *CUDAFusedInference) PartialStepQ4Continue(pos, layerStart, layerEnd int, hiddenOut, logitsOut []float32) int { return -1 }
func (f *CUDAFusedInference) ResetKV() {}
func (f *CUDAFusedInference) QueueHiddenToL3() {}
func (f *CUDAFusedInference) WaitL3() {}
func (f *CUDAFusedInference) GetL3Hidden() []float32 { return nil }
func (f *CUDAFusedInference) HasL3() bool { return false }
func (f *CUDAFusedInference) SwapHiddenBuffer() {}
func (f *CUDAFusedInference) BuildFusedDispatch() {}
func (f *CUDAFusedInference) DisableFusedDispatch() {}
func (f *CUDAFusedInference) SetLayerRange(layerStart, layerEnd int) {}
func (f *CUDAFusedInference) HiddenIdx() int { return 0 }
func (f *CUDAFusedInference) SignalMailbox() {}
func (f *CUDAFusedInference) SignalMailboxOnStream1() {}
func (f *CUDAFusedInference) WaitMailbox() {}
func (f *CUDAFusedInference) HasMailbox() bool { return false }
func (f *CUDAFusedInference) ReadHiddenBlocking(dst []float32) {}
func (f *CUDAFusedInference) SetVNodeCount(n int) {}
func (f *CUDAFusedInference) VNodeCount() int { return 1 }
func (f *CUDAFusedInference) RecordStageDone() {}
func (f *CUDAFusedInference) WaitStageDone() {}
func (f *CUDAFusedInference) SyncStream1() {}
func (f *CUDAFusedInference) QueueHiddenToL3OnStream1() {}
func (f *CUDAFusedInference) ComputeStream1Ptr() unsafe.Pointer { return nil }
func (f *CUDAFusedInference) PartialStepQ4ContinueOnStream1(pos, layerStart, layerEnd int, hiddenOut, logitsOut []float32) int { return -1 }
func (f *CUDAFusedInference) ExportHidden() []float32 { return nil }
