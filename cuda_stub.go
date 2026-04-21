//go:build !linux || !cgo

package mongoose

import "unsafe"

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

func (c *CUDA) MatMulL3(wT *Tensor, pinnedX unsafe.Pointer, pinnedOut unsafe.Pointer, m, k, n int) {}
func (c *CUDA) MatMulTransposeBTInto(out, a, b *Tensor, m, k, n int) {}
func (c *CUDA) MatMulTInto(out, a, b *Tensor, m, k, n int) {}
func (c *CUDA) MatMulTransposeATInto(out, a, b *Tensor, m, k, n int) {}

// TensorEngine stubs
func (c *CUDA) FromHost(data []float32, shape []int) *Tensor    { return nil }
func (c *CUDA) Zeros(shape []int) *Tensor                       { return nil }
func (c *CUDA) ToHost(t *Tensor) []float32                      { return nil }
func (c *CUDA) Release(t *Tensor)                               {}
func (c *CUDA) MatMulT(a, b *Tensor, m, k, n int) *Tensor       { return nil }
func (c *CUDA) MatMulTransposeAT(a, b *Tensor, m, k, n int) *Tensor { return nil }
func (c *CUDA) MatMulTransposeBT(a, b *Tensor, m, k, n int) *Tensor { return nil }
func (c *CUDA) AddT(a, b *Tensor) *Tensor                       { return nil }
func (c *CUDA) AddInPlace(a, b *Tensor)                         {}
func (c *CUDA) ScaleT(a *Tensor, s float32) *Tensor             { return nil }
func (c *CUDA) ReLUT(a *Tensor) *Tensor                         { return nil }
func (c *CUDA) ReLUBackwardT(dOut, fwdInput *Tensor) *Tensor    { return nil }
func (c *CUDA) TransposeT(a *Tensor, rows, cols int) *Tensor    { return nil }
func (c *CUDA) CopyT(src *Tensor) *Tensor                       { return nil }
func (c *CUDA) CopyInto(dst, src *Tensor)                       {}
func (c *CUDA) UploadInto(dst *Tensor, data []float32)           {}
func (c *CUDA) UploadSlice(dst *Tensor, offsetFloats int, data []float32) {}
func (c *CUDA) ZerosBF16(shape []int) *Tensor                   { return nil }

// INT8 stubs
type QuantizedTensor struct {
	DataInt8 []int8
	Scales   []float32
	Shape    []int
	Rows     int
	Cols     int
}

type Int8Tensor struct {
	DataPtr  unsafe.Pointer
	ScalePtr unsafe.Pointer
	Rows     int
	Cols     int
}

func (q *Int8Tensor) VRAMBytes() int                             { return 0 }
func (c *CUDA) FromHostInt8(qt *QuantizedTensor) *Int8Tensor     { return nil }
func (c *CUDA) ReleaseInt8(q *Int8Tensor)                        {}
func (c *CUDA) DequantToFP16(q *Int8Tensor, fp16Buf unsafe.Pointer) {}
func (c *CUDA) DequantToFP32(q *Int8Tensor, fp32Buf unsafe.Pointer) {}
func (c *CUDA) AllocGPU(bytes int) unsafe.Pointer                   { return nil }
func (c *CUDA) UploadRaw(dst *Tensor, data []int8)                  {}
func (c *CUDA) UploadRawBytes(dst *Tensor, data unsafe.Pointer, nBytes int) {}
func (c *CUDA) ZerosL3(shape []int) *Tensor                         { return nil }
func (c *CUDA) AllocFP16Buffer(nElements int) unsafe.Pointer     { return nil }
func (c *CUDA) MatMulFP16TransposeBT(a, b *Tensor, m, k, n int) *Tensor { return nil }
func (c *CUDA) AllocFP16Tensor(nElements int, shape []int) *Tensor { return nil }
func (c *CUDA) FreeFP16Tensor(t *Tensor)                        {}
func (c *CUDA) FromHostFP16(data []float32, shape []int) *Tensor { return nil }

func PoolStats() (int64, int64, int64, int64) { return 0, 0, 0, 0 }
