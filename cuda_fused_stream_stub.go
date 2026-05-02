//go:build !linux || !cgo

package mongoose

import "unsafe"

type PhaseTrace struct {
	Coherence []float32
	Energy    []float32
}

type needleWeightState struct{}
type NeedleLayerState struct{}

type CUDAStreamInfer struct {
	Trace        *PhaseTrace
	TraceEnabled bool
	NeedleActive bool
	Needle       []NeedleLayerState
}

func (f *CUDAFusedInference) StreamBuild() *CUDAStreamInfer { return nil }

func (s *CUDAStreamInfer) StreamForwardToken(pos int, embHidden []float32, nLayers int,
	layerPtrs []unsafe.Pointer, layerSizes []int32, logitsOut []float32) int {
	return -1
}

func (s *CUDAStreamInfer) EnableTrace(nLayers int) {}
func (s *CUDAStreamInfer) EnableNeedle(nLayers int, lr, beta1, wd float32) {}
