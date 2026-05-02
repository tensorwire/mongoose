//go:build !linux || !cgo

package mongoose

type GraphParams struct {
	Pos    int32
	SeqLen int32
	_      [6]int32
}

type GraphCaptureConfig struct {
	LayerStart    int
	LayerEnd      int
	StreamCount   int
	PipelineBreak int
}

type CUDAGraph struct {
	captured bool
	config   GraphCaptureConfig
}

func LoadGraphKernel() bool { return false }
func HasGraphKernel() bool  { return false }
func (f *CUDAFusedInference) TestMinimalCapture() bool { return false }
func (f *CUDAFusedInference) ValidateGraphCapture(hiddenIn []float32, layerStart, layerEnd int) bool { return false }
func (f *CUDAFusedInference) ValidateMultiStreamGraph(hiddenIn []float32, layerStart, layerEnd, pipelineBreak int) bool { return false }
func (f *CUDAFusedInference) CaptureGraph(cfg GraphCaptureConfig, pos int) *CUDAGraph {
	return nil
}
func (f *CUDAFusedInference) UploadHidden(data []float32) {}
func (f *CUDAFusedInference) ReadHidden() []float32       { return nil }
func (g *CUDAGraph) Launch(pos int)                                    {}
func (f *CUDAFusedInference) UpdateAndLaunch(g *CUDAGraph, pos int) bool { return false }
func (g *CUDAGraph) Destroy()                                          {}
