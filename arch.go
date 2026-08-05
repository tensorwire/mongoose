package mongoose

// ArchParams carries the architecture-specific scalar multipliers that some
// model families apply on top of an otherwise Llama-shaped forward pass.
//
// This type was reconstructed after the 2026-08-05 crash from the C struct
// MongooseArchParams and the call sites in metal.go, both of which survived in
// the Go build cache. Field names, order, and types are fixed by those sites.
//
// The zero value means "Llama defaults": every multiplier is skipped, so
// ArchParams{} leaves the forward pass byte-identical to a build that never
// mentions arch scalars at all. That is what makes this safe to thread through
// unconditionally.
//
// Granite requires all four inference fields. Its tensor layout is identical to
// Llama's, so a model run without them produces fluent-looking nonsense rather
// than an error — there is no shape mismatch to catch the mistake.
type ArchParams struct {
	// EmbeddingMultiplier scales token embeddings immediately after lookup.
	// Granite-4.1: 12.0.
	EmbeddingMultiplier float32

	// ResidualMultiplier scales each block's output before it is added back
	// into the residual stream. Granite-4.1-3b: 0.22.
	ResidualMultiplier float32

	// AttentionScale replaces the usual 1/sqrt(headDim) applied to the QK
	// product.
	//
	// For Granite this is config.json's attention_multiplier, and it is
	// 1/headDim, NOT 1/sqrt(headDim) — the two agree only at headDim 1. Do not
	// "correct" this to a sqrt; it is the single easiest way to reintroduce
	// coherent-looking garbage.
	AttentionScale float32

	// LogitsScaling DIVIDES the final logits — it is not a multiplier despite
	// the name upstream gives it. Granite-4.1-3b: 6.0 (dim 2560 / 320... read
	// it from config.json rather than deriving it).
	LogitsScaling float32

	// AdamBeta2 overrides the optimizer's second-moment decay during training.
	// Training-only: FusedSetArch ignores it, and only BuildFullGraphArch
	// forwards it to the graph.
	AdamBeta2 float32
}

// IsZero reports whether every field is unset, i.e. whether these params are
// equivalent to Llama defaults. Callers use it to skip the arch plumbing
// entirely rather than pushing a struct of zeros through to the GPU.
func (a ArchParams) IsZero() bool {
	return a.EmbeddingMultiplier == 0 &&
		a.ResidualMultiplier == 0 &&
		a.AttentionScale == 0 &&
		a.LogitsScaling == 0 &&
		a.AdamBeta2 == 0
}
