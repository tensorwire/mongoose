//go:build !linux || !cgo

package mongoose

type MLPTrainGraph struct{}

func NewMLPTrainGraph(eng *CUDA, mlp *MLP, batchSize int) *MLPTrainGraph { return nil }
func (g *MLPTrainGraph) Capture()                                        {}
func (g *MLPTrainGraph) CaptureWithParams(lr, bc1, bc2 float32)          {}
func (g *MLPTrainGraph) UploadBatch(features, targets []float32)         {}
func (g *MLPTrainGraph) RunStep(lr float32) float32                      { return 0 }
func (g *MLPTrainGraph) LaunchGraph(lr float32) float32                  { return 0 }
func (g *MLPTrainGraph) DownloadWeights()                                {}
func (g *MLPTrainGraph) Destroy()                                        {}
