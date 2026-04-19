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
func KFP32ToFP16(inPtr, outPtr unsafe.Pointer, n int) {}
func KFP16ToFP32(inPtr, outPtr unsafe.Pointer, n int) {}
