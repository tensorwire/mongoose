package mongoose

import "unsafe"

// NeedleForwardConfig holds all parameters for the fused needle forward step.
// One CGo call runs the full autoregressive sequence with DNA-paired pokes.
type NeedleForwardConfig struct {
	// Per-layer weight pointers (unsafe.Pointer to C arrays)
	WqPtrs, WkPtrs, WvPtrs, WoPtrs       unsafe.Pointer
	WgatePtrs, WupPtrs, WdownPtrs         unsafe.Pointer
	WqSPtrs, WkSPtrs, WvSPtrs, WoSPtrs   unsafe.Pointer
	WgateSPtrs, WupSPtrs, WdownSPtrs      unsafe.Pointer
	Norm1Ptrs, Norm2Ptrs                   unsafe.Pointer
	KCachePtrs, VCachePtrs                 unsafe.Pointer

	// Scratch buffers (GPU-resident, reused per token)
	HiddenPtr, PreHiddenPtr               unsafe.Pointer
	NormedPtr, QPtr, KPtr, VPtr           unsafe.Pointer
	AttnOutPtr, ProjPtr, Normed2Ptr       unsafe.Pointer
	GatePrePtr, UpOutPtr, FfnMidPtr       unsafe.Pointer
	Q8ScratchPtr                          unsafe.Pointer
	CosTabPtr, SinTabPtr                  unsafe.Pointer

	// Final layer
	FinalNormPtr, LmHeadPtr, LmHeadSPtr   unsafe.Pointer
	LogitsBufPtr                          unsafe.Pointer

	// Coherence output (GPU float[nLayers], read after sync)
	CohBufPtr                             unsafe.Pointer

	// Embedding table (GPU) + token IDs (GPU int32 array)
	EmbedTablePtr                         unsafe.Pointer
	TokenIDsPtr                           unsafe.Pointer

	// Dimensions
	Dim, KvDim, HeadDim, FfnDim           int
	NHeads, NKVHeads, HalfHead            int
	VocabSize, NLayers, SeqLen            int

	// Weight format: 0=Q8, 1=Q4, 2=FP16
	Format                                int
	// Needle mode: 0=none, 1=inline(finetune), 2=sparse(scratch)
	NeedleMode                            int

	// Needle hyperparams
	SignalScale                           float32
	NeedleLR, NeedleBeta1, NeedleWD      float32
	JitterAmp, DropoutP                   float32

	// DNA rung geometry (from Helix)
	Backbone1, Glyco1, Hbond1             float32
	Hbond2, Glyco2, Backbone2             float32

	// Needle state pointer arrays ([7*nLayers] each)
	NeedleMomsPtrs, NeedleDeltasPtrs      unsafe.Pointer
	NeedleMasksPtrs                       unsafe.Pointer // inline only
	NeedleCachesPtrs                      unsafe.Pointer // Q8 FP32 cache only
	NeedleHotIdxPtrs                      unsafe.Pointer // sparse only
	NHot                                  int
}

// Format constants matching CUDA defines
const (
	FormatQ8   = 0
	FormatQ4   = 1
	FormatFP16 = 2
)

// Needle mode constants
const (
	NeedleModeNone   = 0
	NeedleModeInline = 1 // finetune
	NeedleModeSparse = 2 // from scratch
)

// NeedleBatchConfig holds parameters for the batched needle training step.
// Full sequence [seqLen, dim] processed in parallel via batch matmuls.
type NeedleBatchConfig struct {
	// Per-layer weight + scale pointers (unsafe.Pointer to C arrays)
	WqPtrs, WkPtrs, WvPtrs, WoPtrs       unsafe.Pointer
	WqSPtrs, WkSPtrs, WvSPtrs, WoSPtrs   unsafe.Pointer
	WgatePtrs, WupPtrs, WdownPtrs         unsafe.Pointer
	WgateSPtrs, WupSPtrs, WdownSPtrs      unsafe.Pointer
	Norm1Ptrs, Norm2Ptrs                   unsafe.Pointer
	KCachePtrs, VCachePtrs                 unsafe.Pointer

	// Batch buffers [seqLen, dim] or [seqLen, ffnDim]
	HiddenPtr, PreHiddenPtr               unsafe.Pointer
	NormedPtr, QPtr, KPtr, VPtr           unsafe.Pointer
	AttnOutPtr, ProjPtr, Normed2Ptr       unsafe.Pointer
	GatePrePtr, UpOutPtr, FfnMidPtr       unsafe.Pointer
	ScratchWPtr                           unsafe.Pointer // [max(dim*dim, ffnDim*dim)] for dequant
	FP16ScratchPtr                        unsafe.Pointer
	CosTabPtr, SinTabPtr                  unsafe.Pointer

	// Final layer
	FinalNormPtr, LmHeadPtr, LmHeadSPtr   unsafe.Pointer
	LogitsBufPtr                          unsafe.Pointer // [seqLen, vocabSize]

	// Coherence
	CohBufPtr                             unsafe.Pointer // [nLayers]
	CohPerPosPtr                          unsafe.Pointer // [seqLen] scratch

	// Dimensions
	Dim, KvDim, HeadDim, FfnDim           int
	NHeads, NKVHeads, HalfHead            int
	VocabSize, NLayers, SeqLen            int

	// Format + mode
	Format, NeedleMode                    int

	// Needle params
	SignalScale                           float32
	NeedleLR, NeedleBeta1, NeedleWD      float32
	JitterAmp, DropoutP                   float32

	// DNA rung
	Backbone1, Glyco1, Hbond1             float32
	Hbond2, Glyco2, Backbone2             float32

	// Needle state arrays [7*nLayers]
	NeedleMomsPtrs, NeedleDeltasPtrs      unsafe.Pointer
	NeedleMasksPtrs                       unsafe.Pointer
	NeedleCachesPtrs                      unsafe.Pointer
	NeedleHotIdxPtrs                      unsafe.Pointer
	NHot                                  int
}
