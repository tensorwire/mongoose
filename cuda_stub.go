//go:build !linux || !cgo

package mongoose

type CUDA struct{}

func NewCUDA() *CUDA                                          { return nil }
func (c *CUDA) Name() string                                  { return "cuda/unavailable" }
func (c *CUDA) MatMul(a, b []float32, m, k, n int) []float32  { return nil }
func (c *CUDA) RMSNorm(x, weight []float32, eps float32)      {}
func (c *CUDA) SoftMax(x []float32, n int)                    {}
func (c *CUDA) ReLU(x []float32)                              {}
func (c *CUDA) VRAM() uint64                                  { return 0 }
func (c *CUDA) Benchmark() float64                            { return 0 }
func (c *CUDA) Sync()                                         {}

type CUDATimer struct{}

func NewCUDATimer() *CUDATimer       { return &CUDATimer{} }
func (t *CUDATimer) Start()          {}
func (t *CUDATimer) StopMs() float32 { return 0 }
func (t *CUDATimer) Destroy()        {}

func (c *CUDA) MatMulTransBInto(out, A, B []float32, m, k, n int) {}
func (c *CUDA) MatMulInto(out, A, B []float32, m, k, n int)       {}
func (c *CUDA) MatMulAddInto(G, A, B []float32, m, k, n int)      {}
func (c *CUDA) MatMulTransA(A, B []float32, m, k, n int) []float32 { return nil }
func (c *CUDA) GER(G, x, y []float32, m, n int, alpha float32)    {}
func (c *CUDA) Nrm2(x []float32) float32                          { return 0 }
func (c *CUDA) Scal(x []float32, alpha float32)                   {}
func (c *CUDA) AdamWStep(D, G, M, V []float32, n int, lr, beta1, beta2, bc1, bc2, eps, wd float32) {}

func (c *CUDA) BuildFullGraph(dim, kvDim, headDim, nHeads, nKVHeads, ffnDim, vocabSize, nLayers, seqLen int, ropeTheta float64, mode int) int { return 0 }
func (c *CUDA) GraphTrainStepAdam(tokens, targets []int32, lr float32) float32 { return -1 }
func (c *CUDA) GraphNumWeights() int                    { return 0 }
func (c *CUDA) GraphSetVariable(varIdx int, data []float32) int { return 0 }

func PoolStats() (int64, int64, int64, int64) { return 0, 0, 0, 0 }
