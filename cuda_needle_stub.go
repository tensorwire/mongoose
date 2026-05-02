//go:build !linux || !cgo

package mongoose

type NeedleTrainState struct{}
type needleLayerGPU struct{}

func NewNeedleTrainState(ci *CUDAFusedInference, nLayers, dim, nHot int, lr, beta1, wd float32) *NeedleTrainState {
	return nil
}
func (s *NeedleTrainState) ForwardOneLayer(layer, pos int, hiddenIn, logitsOut []float32) float32 { return 0 }
func (s *NeedleTrainState) NeedlePoke(layer int, avgG float32) {}
func (s *NeedleTrainState) UpdateCoherence(layer int, c float32) {}
func (s *NeedleTrainState) SnapshotHistory() {}
func (s *NeedleTrainState) AvgCoherence() float32 { return 0 }
func (s *NeedleTrainState) StallCount() int { return 0 }
