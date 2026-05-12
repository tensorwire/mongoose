//go:build !(linux && cgo)

package mongoose

type MLPFused struct{}

func NewMLPFused(eng *CUDA, mlp *MLP, batchSize int) *MLPFused { return nil }
func (f *MLPFused) UploadBatch(features, targets []float32)    {}
func (f *MLPFused) TrainStep(lr float32) float32                { return 0 }
func (f *MLPFused) DownloadWeights()                            {}
func (f *MLPFused) Destroy()                                    {}
