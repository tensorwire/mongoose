//go:build !linux || !cgo

package mongoose

func (f *CUDAFusedInference) BuildNeedleForwardConfig() *NeedleForwardConfig {
	return nil
}

func (f *CUDAFusedInference) BuildNeedleBatchConfig(seqLen int) *NeedleBatchConfig {
	return nil
}
