//go:build !linux || !cgo

package mongoose

import "unsafe"

func LoadKernels(paths ...string) bool { return false }
func KernelsLoaded() bool              { return false }
func KRMSNorm(xPtr, weightPtr unsafe.Pointer, seqLen, dim int) {}
func KRMSNormOut(inputPtr, outPtr, weightPtr unsafe.Pointer, seqLen, dim int) {}
func KReLU(xPtr unsafe.Pointer, n int) {}
func KReLUOut(inputPtr, outPtr unsafe.Pointer, n int) {}
func KReLUBackward(outPtr, dOutPtr, inputPtr unsafe.Pointer, n int) {}
func KAddInPlace(aPtr, bPtr unsafe.Pointer, n int) {}
func KScaleByWeight(xPtr, weightPtr unsafe.Pointer, seqLen, dim int) {}
func KEmbeddingGather(outPtr, tokEmbPtr, posEmbPtr, tokensPtr unsafe.Pointer, seqLen, dim int) {}
func KFusedAddRMSNorm(aPtr, bPtr, outPtr, weightPtr unsafe.Pointer, seqLen, dim int) {}
func KAddOut(aPtr, bPtr, outPtr unsafe.Pointer, n int) {}
func KCausalAttention(qPtr, kPtr, vPtr, outPtr unsafe.Pointer, seqLen, dim, numHeads int) {}
func KAdamW(paramPtr, gradPtr, mPtr, vPtr unsafe.Pointer, lr, wd float32, step int, n int) {}
func KCopy(dst, src unsafe.Pointer, bytes int) {}
func KZero(ptr unsafe.Pointer, bytes int) {}
func KSync() {}
func KReLUAndIndex(xPtr, activeIdxPtr, activeCountPtr unsafe.Pointer, n int) {}
func SparseKernelsLoaded() bool { return false }
func KSparseMatMul(outPtr, wtPtr, xPtr, activeIdxPtr unsafe.Pointer, activeCount, rows, cols int) {}
func KReLUAndIndexFP16(xPtr, activeIdxPtr, activeCountPtr unsafe.Pointer, n int) {}
func SparseFP16KernelsLoaded() bool { return false }
func KSparseMatMulFP16(outPtr, wtPtr, xPtr, activeIdxPtr unsafe.Pointer, activeCount, rows, cols int) {}
func TrainKernelsLoaded() bool { return false }
func KSiLUGateMul(gatePtr, upPtr, outPtr unsafe.Pointer, n int) {}
func KSiLUGateBackward(dOutPtr, gatePtr, upPtr, dGatePtr, dUpPtr unsafe.Pointer, n int) {}
func KRoPE(xPtr, cosPtr, sinPtr unsafe.Pointer, seqLen, dim, headDim, nHeads int) {}
func KRoPEBackward(dxPtr, cosPtr, sinPtr unsafe.Pointer, seqLen, dim, headDim, nHeads int) {}
func KScaleOut(xPtr, outPtr unsafe.Pointer, alpha float32, n int) {}
func KEmbedGather2(outPtr, embedPtr, tokensPtr unsafe.Pointer, seqLen, dim int) {}
func KRMSNormOutSave(inputPtr, outPtr, weightPtr, scalesPtr unsafe.Pointer, seqLen, dim int) {}
func KRMSNormBackward(dOutPtr, xInPtr, weightPtr, scalesPtr, dxPtr unsafe.Pointer, seqLen, dim int) {}
func KRMSNormWeightGrad(dOutPtr, normedPtr, weightPtr, dWPtr unsafe.Pointer, seqLen, dim int)      {}
func KGradSumSq(gradPtr, sumsqPtr unsafe.Pointer, n int)                                          {}
func KGradScale(gradPtr unsafe.Pointer, scale float32, n int)                                     {}
func KDecodeAttention(qPtr, kCachePtr, vCachePtr, outPtr unsafe.Pointer, cacheLen, dim, kvDim, numHeads, numKVHeads int) {}
func KCausalAttentionGQA(qPtr, kPtr, vPtr, outPtr unsafe.Pointer, seqLen, dim, kvDim, numHeads, numKVHeads int) {}
func AttnBackwardLoaded() bool { return false }
func KCausalAttentionBackward(qPtr, kPtr, vPtr, dOutPtr, dQPtr, dKPtr, dVPtr unsafe.Pointer, seqLen, dim, kvDim, numHeads, numKVHeads int) {}
func KCrossEntropy(hiddenPtr, embedWPtr unsafe.Pointer, D, vocabSize int, targetsPtr, lossesPtr, dHiddenPtr unsafe.Pointer, invN float32, nPos int) {}
func KSoftmaxCE(logitsPtr, targetsPtr, lossesPtr, gradPtr unsafe.Pointer, nPos, vocabSize int, invN float32) {}
func SoftmaxCELoaded() bool { return false }
func DequantKernelsLoaded() bool { return false }
func KDequantInt8ToFP16(dataPtr, scalesPtr, outPtr unsafe.Pointer, rows, cols int) {}
func KDequantInt8ToFP32(dataPtr, scalesPtr, outPtr unsafe.Pointer, rows, cols int) {}
func KDequantInt8DeltaToFP32(dataPtr, scalesPtr, deltaPtr, outPtr unsafe.Pointer, n, cols int) {}
func KFP32ToFP16(inPtr, outPtr unsafe.Pointer, n int) {}
func KFP16ToFP32(inPtr, outPtr unsafe.Pointer, n int) {}
func HelixDNALoaded() bool { return false }
func KHelixDNAStep(d1, d2, g1, g2, m1, m2, v1, v2 unsafe.Pointer, lr, beta1, beta2 float32, step int, eps, wd float32, bb1, gly1, hb1, hb2, gly2, bb2, bondStrength float32, n int) {}
func HelixNeedleLoaded() bool { return false }
func KHelixNeedle(dataPtr, scalesPtr, gradPtr, momPtr, velPtr, maskPtr unsafe.Pointer, lr, beta1, beta2 float32, step int, eps, wd float32, n, cols int) {}
func KHelixNeedlePaired(d1Ptr, d2Ptr, s1Ptr, s2Ptr, g1Ptr, g2Ptr, m1Ptr, m2Ptr, v1Ptr, v2Ptr, maskPtr unsafe.Pointer, lr, beta1, beta2 float32, step int, eps, wd float32, bb1, gly1, hb1, hb2, gly2, bb2, bondStrength float32, n, cols int) {}
func CUDACheck() string { return "" }
func KQ8Matvec(actPtr, weightPtr, scalesPtr, outPtr unsafe.Pointer, N, K int) {}
func KQ4Matvec(actPtr, weightPtr, scalesPtr, outPtr unsafe.Pointer, N, K int) {}
func KKVCacheWrite(cachePtr, srcPtr unsafe.Pointer, pos, kvDim int) {}
func HasQ8Matvec() bool { return false }
func HasQ4Matvec() bool { return false }
