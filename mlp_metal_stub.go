//go:build !darwin || !cgo

package mongoose

import "unsafe"

type MLPMetal struct{}

func NewMLPMetal(eng *Metal, mlp *MLP, batchSize int) *MLPMetal { return nil }
func (m *MLPMetal) UploadBatch(features, targets []float32)     {}
func (m *MLPMetal) TrainStep(lr float32) float32                 { return 0 }
func (m *MLPMetal) DownloadWeights()                             {}
func (m *MLPMetal) Destroy()                                     {}

func MtlGemmBT(a, b, c unsafe.Pointer, M, K, N int)  {}
func MtlGemmTN(a, b, c unsafe.Pointer, M, K, N int)  {}
func MtlGemmNN(a, b, c unsafe.Pointer, M, K, N int)  {}
func MtlBiasAdd(out, bias unsafe.Pointer, B, D int)   {}
func MtlBNForward(x, mean, vr, gamma, beta, runMean, runVar unsafe.Pointer, B, D int, momentum float32) {}
func MtlBNBackward(dOut, x, mean, vr, gamma, dGamma, dBeta unsafe.Pointer, B, D int) {}
func MtlBCE(logits, targets, grad, lossBuf, lossScalar unsafe.Pointer, B int) {}
func MtlAdamWStep(param, grad, m, v unsafe.Pointer, lr, beta1, beta2, bc1, bc2, eps, wd float32, n int) {}
func MtlDropoutFwd(x, mask unsafe.Pointer, n int, p float32, seed, counter uint32) {}
func MtlDropoutBwd(dx, mask unsafe.Pointer, n int) {}
func MtlBufAlloc(nFloats int) unsafe.Pointer       { return nil }
func MtlBufFromHost(data []float32) unsafe.Pointer  { return nil }
func MtlBufZeros(nFloats int) unsafe.Pointer        { return nil }
func MtlBufSharedSlice(ref unsafe.Pointer, n int) []float32 { return nil }
func MtlMLPInit() bool { return false }
