package mongoose

// TrainEngine provides the BLAS and optimizer primitives needed for training.
// Accelerate (macOS AMX), CUDA (cuBLAS), and CPU all implement this.
// The training loop calls these methods — never vendor-specific APIs directly.
type TrainEngine interface {
	Engine

	MatMulTransBInto(C, A, B []float32, m, k, n int)
	MatMulInto(C, A, B []float32, m, k, n int)
	MatMulAddInto(G, A, B []float32, m, k, n int)
	Nrm2(x []float32) float32
	Scal(x []float32, alpha float32)
	MatMulTransA(A, B []float32, m, k, n int) []float32
	GER(G, x, y []float32, m, n int, alpha float32)
	AdamWStep(D, G, M, V []float32, n int,
		lr, beta1, beta2, bc1, bc2, eps, wd float32)
}

// GraphTrainEngine provides GPU-graph-based training.
// One fused dispatch for forward + backward + optimizer.
// MPSGraph on Metal, fused kernel chain on CUDA.
type GraphTrainEngine interface {
	BuildFullGraph(dim, kvDim, headDim, nHeads, nKVHeads, ffnDim,
		vocabSize, nLayers, seqLen int, ropeTheta float64, mode int) int
	GraphTrainStepAdam(tokens, targets []int32, lr float32) float32
	GraphNumWeights() int
	GraphSetVariable(varIdx int, data []float32) int
	Sync()
}

// SupportsGraphTrain returns true if the engine supports graph-based training.
func SupportsGraphTrain(eng Engine) bool {
	_, ok := eng.(GraphTrainEngine)
	return ok
}

// AsGraphTrainEngine returns the GraphTrainEngine or nil.
func AsGraphTrainEngine(eng Engine) GraphTrainEngine {
	g, _ := eng.(GraphTrainEngine)
	return g
}

// AsTrainEngine returns the TrainEngine or nil.
func AsTrainEngine(eng Engine) TrainEngine {
	t, _ := eng.(TrainEngine)
	return t
}
