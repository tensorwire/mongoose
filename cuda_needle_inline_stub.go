//go:build !linux || !cgo

package mongoose

type NeedleInlineState struct{}
type inlineLayerState struct{}

func NewNeedleInlineState(ci *CUDAFusedInference, nLayers, nActive int, lr, beta1, wd float32) *NeedleInlineState {
	return nil
}
func (s *NeedleInlineState) SetMask(layer, matIdx int, goodness []float32, nRows int) {}
func (s *NeedleInlineState) ForwardOneLayer(layer, pos int, hiddenIn, logitsOut []float32) float32 { return 0 }
func (s *NeedleInlineState) PokeInline(layer int, signalScale float32) {}
func (s *NeedleInlineState) AvgCoherence() float32 { return 0 }
func (s *NeedleInlineState) StallCount() int { return 0 }
