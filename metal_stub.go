//go:build !darwin || !cgo

package mongoose

type Metal struct{}

func NewMetal() *Metal                                         { return nil }
func (m *Metal) Name() string                                  { return "metal/unavailable" }
func (m *Metal) MatMul(a, b []float32, m2, k, n int) []float32 { return nil }
func (m *Metal) RMSNorm(x, weight []float32, eps float32)      {}
func (m *Metal) SoftMax(x []float32, n int)                    {}
func (m *Metal) ReLU(x []float32)                              {}
func (m *Metal) VRAM() uint64                                  { return 0 }
func (m *Metal) Benchmark() float64                            { return 0 }
func (m *Metal) Close()                                        {}
func (m *Metal) BeginBatch()                                   {}
func (m *Metal) EndBatch()                                     {}
func (m *Metal) Sync()                                         {}
func MtlComputeReady() bool                                    { return false }

func (m *Metal) MatMulTransBInto(out, A, B []float32, m2, k, n int) {}
func (m *Metal) MatMulInto(out, A, B []float32, m2, k, n int)       {}
func (m *Metal) MatMulAddInto(G, A, B []float32, m2, k, n int)      {}
func (m *Metal) MatMulTransA(A, B []float32, m2, k, n int) []float32 { return nil }
func (m *Metal) GER(G, x, y []float32, m2, n int, alpha float32)    {}
func (m *Metal) Nrm2(x []float32) float32                           { return 0 }
func (m *Metal) Scal(x []float32, alpha float32)                    {}
func (m *Metal) AdamWStep(D, G, M, V []float32, n int, lr, beta1, beta2, bc1, bc2, eps, wd float32) {}

func (m *Metal) BuildFullGraph(dim, kvDim, headDim, nHeads, nKVHeads, ffnDim, vocabSize, nLayers, seqLen int, ropeTheta float64, mode int) int { return -1 }
func (m *Metal) GraphTrainStepAdam(tokens, targets []int32, lr float32) float32 { return -1 }
func (m *Metal) GraphNumWeights() int                          { return 0 }
func (m *Metal) GraphSetVariable(varIdx int, data []float32) int { return -1 }

func (m *Metal) GraphFullBuilt() bool                          { return false }
func (m *Metal) GraphNumDiffable() int                         { return 0 }
func (m *Metal) GraphReadVariable(varIdx int, dst []float32) int { return -1 }
func (m *Metal) GraphApplyWeights(varIdx int, data []float32) int { return -1 }

type MetalSubprocess struct{}

func NewMetalSubprocess() *MetalSubprocess                            { return nil }
func (m *MetalSubprocess) Name() string                               { return "metal-subprocess/unavailable" }
func (m *MetalSubprocess) MatMul(a, b []float32, m2, k, n int) []float32 { return nil }
func (m *MetalSubprocess) RMSNorm(x, weight []float32, eps float32)   {}
func (m *MetalSubprocess) SoftMax(x []float32, n int)                 {}
func (m *MetalSubprocess) ReLU(x []float32)                           {}
func (m *MetalSubprocess) VRAM() uint64                               { return 0 }
func (m *MetalSubprocess) Benchmark() float64                         { return 0 }
func (m *MetalSubprocess) Close()                                     {}
