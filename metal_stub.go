//go:build !darwin || !cgo

package mongoose

import "unsafe"

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

func (m *Metal) FromHost(data []float32, shape []int) *Tensor    { return nil }
func (m *Metal) Zeros(shape []int) *Tensor                      { return nil }
func (m *Metal) ToHost(t *Tensor) []float32                     { return nil }
func (m *Metal) Release(t *Tensor)                              {}
func (m *Metal) MatMulT(a, b *Tensor, rows, k, n int) *Tensor   { return nil }
func (m *Metal) MatMulTransposeBT(a, b *Tensor, m2, k, n int) *Tensor { return nil }
func (m *Metal) MatMulTransposeAT(a, b *Tensor, m2, k, n int) *Tensor { return nil }
func (m *Metal) AddInPlace(a, b *Tensor)                        {}
func (m *Metal) AddT(a, b *Tensor) *Tensor                      { return nil }
func (m *Metal) ScaleT(a *Tensor, s float32) *Tensor            { return nil }
func (m *Metal) ReLUT(a *Tensor) *Tensor                        { return nil }
func (m *Metal) ReLUBackwardT(dOut, fwdInput *Tensor) *Tensor   { return nil }
func (m *Metal) TransposeT(a *Tensor, rows, cols int) *Tensor   { return nil }
func (m *Metal) CopyT(src *Tensor) *Tensor                      { return nil }

func (m *Metal) BuildInferGraph(dim, kvDim, headDim, nHeads, nKVHeads, ffnDim, vocabSize, nLayers int, ropeTheta float64) int { return -1 }
func (m *Metal) InferNumWeights() int { return 0 }
func (m *Metal) InferSetWeight(idx int, data []float32) int { return -1 }
func (m *Metal) InferForwardA(hidden []float32, cosSlice, sinSlice []float32, qOut, kOut, vOut []float32, layer int) int { return -1 }
func (m *Metal) InferForwardB(hidden []float32, attnOut []float32, layer int) int { return -1 }
func (m *Metal) InferLogits(hidden []float32, logitsOut []float32) int { return -1 }

func (m *Metal) BuildFused(dim, kvDim, headDim, nHeads, nKVHeads, ffnDim, vocabSize, nLayers, maxSeq int, ropeTheta, rmsEps float64) int { return -1 }
func (m *Metal) FusedNumWeights() int { return 0 }
func (m *Metal) FusedSetWeight(idx int, data []float32) int { return -1 }
func (m *Metal) FusedStep(hidden []float32, cosSlice, sinSlice []float32, pos int, logitsOut []float32) int { return -1 }
func (m *Metal) FusedResetKV() {}
func (m *Metal) FusedResetKVSlot(slot int) {}
func (m *Metal) FusedNumSlots() int { return 0 }
func (m *Metal) FusedPartialStep(hiddenIn []float32, pos, layerStart, layerEnd int, hiddenOut, logitsOut []float32) int { return -1 }
func (m *Metal) FusedPartialStepSlot(slot int, hiddenIn []float32, pos, layerStart, layerEnd int, hiddenOut, logitsOut []float32) int { return -1 }
func (m *Metal) StreamBuild() int { return -1 }
func (m *Metal) StreamUploadLayer(set, layer int, norm1, wq, wk, wv, bq, bk, bv []float32, wo, norm2, gate, up, down []float32) {}
func (m *Metal) StreamStepLayer(set, layer, pos int) int { return -1 }
func (m *Metal) StreamStepFinal(pos int, logitsOut []float32) int { return -1 }
func (m *Metal) StreamSetHidden(hidden []float32) {}
func (m *Metal) BuildFusedInfer(dim, kvDim, headDim, nHeads, nKVHeads, ffnDim, vocabSize, nLayers, maxSeq int, ropeTheta float64) int { return -1 }
func (m *Metal) FusedInferNumWeights() int { return 0 }
func (m *Metal) FusedInferSetWeight(idx int, data []float32) int { return -1 }
func (m *Metal) FusedInferStep(hidden []float32, cosSlice, sinSlice []float32, pos int, logitsOut []float32) int { return -1 }
func (m *Metal) FusedInferReset() int { return -1 }

func (m *Metal) BuildFullGraph(dim, kvDim, headDim, nHeads, nKVHeads, ffnDim, vocabSize, nLayers, seqLen int, ropeTheta float64, mode int) int { return -1 }
func (m *Metal) GraphTrainStepAdam(tokens, targets []int32, lr float32) float32 { return -1 }
func (m *Metal) GraphNumWeights() int                          { return 0 }
func (m *Metal) GraphSetVariable(varIdx int, data []float32) int { return -1 }

func (m *Metal) GraphFullBuilt() bool                          { return false }
func (m *Metal) GraphNumDiffable() int                         { return 0 }
func (m *Metal) GraphReadVariable(varIdx int, dst []float32) int { return -1 }
func (m *Metal) GraphApplyWeights(varIdx int, data []float32) int { return -1 }
func (m *Metal) UploadInto(dst *Tensor, data []float32) {}

func (m *Metal) FusedBegin()             {}
func (m *Metal) FusedEnd()               {}
func (m *Metal) FusedBeginSlot(slot int) {}
func (m *Metal) FusedEndSlot(slot int)   {}
func (m *Metal) FusedSetSlot(slot int)   {}
func (m *Metal) FusedSyncAll()           {}
func (m *Metal) FusedGemmBT(a, b, c *Tensor, M, K, N int) {}
func (m *Metal) FusedGemmNN(a, b, c *Tensor, M, K, N int) {}
func (m *Metal) FusedGemmTN(a, b, c *Tensor, M, K, N int) {}
func (m *Metal) FusedGemmF32BT(a, b, c *Tensor, M, K, N int) {}
func (m *Metal) FusedGemmF32NN(a, b, c *Tensor, M, K, N int) {}
func (m *Metal) FusedGemmF32TN(a, b, c *Tensor, M, K, N int) {}
func (m *Metal) FusedRMSNorm(x, w, scale *Tensor, seqLen, dim int) {}
func (m *Metal) FusedRMSNormBwd(dOut, xIn, w, scale, dx *Tensor, seqLen, dim int) {}
func (m *Metal) FusedRoPE(x *Tensor, headDim, nHeads int, theta float32, stride, seqLen int) {}
func (m *Metal) FusedAttn(q, k, v, out, scores *Tensor, dim, kvDim, headDim, nHeads, nKVHeads, seqLen int) {}
func (m *Metal) FusedAttnBwd(dOut, q, k, v, scores, dQ, dK, dV *Tensor, dim, kvDim, headDim, nHeads, nKVHeads, seqLen, qLen int) {}
func (m *Metal) FusedSiLUGateMul(gate, up, out *Tensor, n int) {}
func (m *Metal) SiLUGateBackwardGPU(dOut, gatePre, upOut, gateAct, dGatePre, dUp *Tensor) {}
func (m *Metal) FusedAddInplace(a, b *Tensor, n int) {}
func (m *Metal) FusedCopy(dst, src *Tensor, n int) {}
func (m *Metal) CELoss(logits, targets, losses *Tensor, seqLen, vocabSize int) {}
func (m *Metal) AdamWT(param, grad, mS, vS *Tensor, lr, wd float32, step int) {}
func (m *Metal) DNARungGPU(d1, g1, m1, v1, d2, g2, m2, v2 *Tensor, bb1, gly1, hb1, hb2, gly2, bb2, bondStr, lr, beta1, beta2, bc1, bc2, eps, wd float32, n int) {}
func (m *Metal) GradNormSq(grad, out *Tensor, n int) {}

type WarmCache struct{}
func (m *Metal) NewWarmCache(nFloats int) *WarmCache { return nil }
func (wc *WarmCache) Slice(off, n int) []float32     { return nil }
func (wc *WarmCache) ByteOffset(floatIdx int) int    { return 0 }
func (wc *WarmCache) BufPtr() unsafe.Pointer         { return nil }
func (wc *WarmCache) Release()                       {}
func (m *Metal) SharedSlice(t *Tensor) []float32 { return nil }
func (m *Metal) DNARungWarm(d1, g1, d2, g2 *Tensor, wc *WarmCache, m1Off, v1Off, m2Off, v2Off int, bb1, gly1, hb1, hb2, gly2, bb2, bondStr, lr, beta1, beta2, bc1, bc2, eps, wd float32, n int) {}
func (m *Metal) AdamWWarm(param, grad *Tensor, wc *WarmCache, mOff, vOff int, lr, beta1, beta2, bc1, bc2, eps, wd float32, n int) {}

func MtlBufPtr(t *Tensor) unsafe.Pointer { return nil }

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
