package mongoose

import (
	"math"
	"math/rand"
)

// MLP is a multi-layer perceptron with BatchNorm, activation, and Dropout.
// Supports both CPU and GPU (via TensorEngine) execution.
//
// Training (numerically stable):
//
//	mlp := NewMLP([]int{nFeatures, 512, 256, 128, 1}, MLPConfig{
//	    Activation: "relu", Dropout: 0.2, BatchNorm: true, Sigmoid: true,
//	})
//	mlp.ToGPU(engine)  // optional: accelerate with GPU
//	opt := NewMLPOptimizer(0.0003, 0.0005, 350)
//	for epoch := range 350 {
//	    logits := mlp.ForwardLogits(batch, batchSize, true)
//	    loss, grad := mlp.BCEWithLogitsLoss(logits, targets, posWeight)
//	    mlp.BackwardFromLogits(grad, batchSize)
//	    params, grads := mlp.Params()
//	    opt.Step(params, grads, epoch)
//	    if mlp.eng != nil { mlp.SyncWeightsToGPU() }
//	}
type MLP struct {
	Layers          []MLPLayer
	Config          MLPConfig
	rng             *rand.Rand
	eng             TensorEngine // nil = CPU mode
	gpuDropoutCounter uint64
}

// MLPLayer holds weights, biases, and optional BatchNorm params for one layer.
type MLPLayer struct {
	// Weights [outDim, inDim] and Bias [outDim]
	W    []float32
	B    []float32
	InDim, OutDim int

	// Gradients (same shapes)
	DW []float32
	DB []float32

	// BatchNorm params (nil if disabled)
	BNGamma   []float32 // scale [outDim]
	BNBeta    []float32 // shift [outDim]
	BNMean    []float32 // running mean [outDim]
	BNVar     []float32 // running variance [outDim]
	BNDGamma  []float32
	BNDBeta   []float32

	// Saved activations for backward pass
	preAct  []float32 // before activation [batchSize, outDim]
	postBN  []float32 // after batchnorm [batchSize, outDim]
	postAct []float32 // after activation [batchSize, outDim]
	mask    []float32 // dropout mask [batchSize, outDim]
	input   []float32 // input to this layer [batchSize, inDim]

	// GPU tensors (populated by ToGPU)
	gW, gB                     *Tensor
	gDW, gDB                   *Tensor
	gBNGamma, gBNBeta          *Tensor
	gSaveMean, gSaveInvStd     *Tensor
	gRunMean, gRunVar          *Tensor
	gMask                      *Tensor
}

// MLPConfig controls the MLP architecture.
type MLPConfig struct {
	Activation    string  // "relu" (default), "silu", "tanh", "none"
	Dropout       float32 // dropout probability (0 = disabled)
	BatchNorm     bool    // enable BatchNorm between layers
	Sigmoid       bool    // apply sigmoid to final output
	LabelSmooth   float32 // label smoothing for BCE loss (0 = disabled)
	WeightDecay   float32 // L2 regularization
	BNMomentum    float32 // BatchNorm running stat momentum (default 0.1)
}

// NewMLP creates a multi-layer perceptron with the given layer dimensions.
// dims[0] is input, dims[len-1] is output, everything in between is hidden.
func NewMLP(dims []int, cfg MLPConfig) *MLP {
	if cfg.Activation == "" {
		cfg.Activation = "relu"
	}
	if cfg.BNMomentum == 0 {
		cfg.BNMomentum = 0.1
	}

	rng := rand.New(rand.NewSource(42))
	layers := make([]MLPLayer, len(dims)-1)

	for i := 0; i < len(dims)-1; i++ {
		inDim, outDim := dims[i], dims[i+1]
		l := MLPLayer{
			InDim:  inDim,
			OutDim: outDim,
			W:      make([]float32, outDim*inDim),
			B:      make([]float32, outDim),
			DW:     make([]float32, outDim*inDim),
			DB:     make([]float32, outDim),
		}

		// Kaiming init
		scale := float32(math.Sqrt(2.0 / float64(inDim)))
		for j := range l.W {
			l.W[j] = (rng.Float32()*2 - 1) * scale
		}

		// BatchNorm for hidden layers (not the final layer)
		if cfg.BatchNorm && i < len(dims)-2 {
			l.BNGamma = make([]float32, outDim)
			l.BNBeta = make([]float32, outDim)
			l.BNMean = make([]float32, outDim)
			l.BNVar = make([]float32, outDim)
			l.BNDGamma = make([]float32, outDim)
			l.BNDBeta = make([]float32, outDim)
			for j := range l.BNGamma {
				l.BNGamma[j] = 1.0
				l.BNVar[j] = 1.0
			}
		}

		layers[i] = l
	}

	return &MLP{Layers: layers, Config: cfg, rng: rng}
}

// ForwardLogits runs the MLP and returns raw logits (no sigmoid).
// Use with BCEWithLogitsLoss for numerically stable training.
func (m *MLP) ForwardLogits(input []float32, batchSize int, training bool) []float32 {
	return m.forward(input, batchSize, training, false)
}

// Forward runs the MLP on a batch of inputs.
// input shape: [batchSize, inputDim], returns [batchSize, outputDim].
// training=true enables dropout and uses batch stats for BatchNorm.
func (m *MLP) Forward(input []float32, batchSize int, training bool) []float32 {
	return m.forward(input, batchSize, training, true)
}

func (m *MLP) forward(input []float32, batchSize int, training, applySigmoid bool) []float32 {
	x := input
	inDim := m.Layers[0].InDim

	for i := range m.Layers {
		l := &m.Layers[i]
		outDim := l.OutDim
		isLast := i == len(m.Layers)-1

		// Save input for backward
		l.input = make([]float32, len(x))
		copy(l.input, x)

		// Linear: out = x @ W^T + b
		out := make([]float32, batchSize*outDim)
		for b := 0; b < batchSize; b++ {
			for o := 0; o < outDim; o++ {
				sum := l.B[o]
				for k := 0; k < inDim; k++ {
					sum += x[b*inDim+k] * l.W[o*inDim+k]
				}
				out[b*outDim+o] = sum
			}
		}
		l.preAct = make([]float32, len(out))
		copy(l.preAct, out)

		// BatchNorm (hidden layers only)
		if l.BNGamma != nil {
			out = m.batchNorm(l, out, batchSize, outDim, training)
		}
		l.postBN = make([]float32, len(out))
		copy(l.postBN, out)

		// Activation (hidden layers only)
		if !isLast {
			for j := range out {
				out[j] = m.activate(out[j])
			}
		}
		l.postAct = make([]float32, len(out))
		copy(l.postAct, out)

		// Dropout (hidden layers only, training only)
		if !isLast && training && m.Config.Dropout > 0 {
			l.mask = make([]float32, len(out))
			scale := 1.0 / (1.0 - m.Config.Dropout)
			for j := range out {
				if m.rng.Float32() < m.Config.Dropout {
					out[j] = 0
					l.mask[j] = 0
				} else {
					out[j] *= float32(scale)
					l.mask[j] = float32(scale)
				}
			}
		} else {
			l.mask = nil
		}

		// Sigmoid on final layer if configured and requested
		if isLast && m.Config.Sigmoid && applySigmoid {
			for j := range out {
				out[j] = 1.0 / (1.0 + float32(math.Exp(-float64(out[j]))))
			}
		}

		x = out
		inDim = outDim
	}

	return x
}

// BackwardFromLogits computes gradients when loss was computed on raw logits.
// dOut is the gradient from BCEWithLogitsLoss (sigmoid(logit) - target).
// Skips sigmoid derivative since the loss already accounts for it.
func (m *MLP) BackwardFromLogits(dOut []float32, batchSize int) {
	m.backward(dOut, batchSize, true)
}

// Backward computes gradients given the loss gradient w.r.t. the output.
// dOut shape: [batchSize, outputDim]. Populates DW, DB, BNDGamma, BNDBeta.
func (m *MLP) Backward(dOut []float32, batchSize int) {
	m.backward(dOut, batchSize, false)
}

func (m *MLP) backward(dOut []float32, batchSize int, fromLogits bool) {
	dx := dOut

	for i := len(m.Layers) - 1; i >= 0; i-- {
		l := &m.Layers[i]
		outDim := l.OutDim
		inDim := l.InDim
		isLast := i == len(m.Layers)-1

		// Sigmoid backward (skip when using BCEWithLogitsLoss — gradient already accounts for sigmoid)
		if isLast && m.Config.Sigmoid && !fromLogits {
			for j := range dx {
				s := l.postAct[j]
				dx[j] *= s * (1.0 - s)
			}
		}

		// Dropout backward
		if l.mask != nil {
			for j := range dx {
				dx[j] *= l.mask[j]
			}
		}

		// Activation backward (hidden layers)
		if !isLast {
			for j := range dx {
				dx[j] *= m.activateGrad(l.postBN[j])
			}
		}

		// BatchNorm backward
		if l.BNGamma != nil {
			dx = m.batchNormBackward(l, dx, batchSize, outDim)
		}

		// Linear backward: dW, dB, dInput
		// Zero grads
		for j := range l.DW { l.DW[j] = 0 }
		for j := range l.DB { l.DB[j] = 0 }

		invBatch := 1.0 / float32(batchSize)
		dInput := make([]float32, batchSize*inDim)

		for b := 0; b < batchSize; b++ {
			for o := 0; o < outDim; o++ {
				d := dx[b*outDim+o]
				l.DB[o] += d * invBatch
				for k := 0; k < inDim; k++ {
					l.DW[o*inDim+k] += d * l.input[b*inDim+k] * invBatch
					dInput[b*inDim+k] += d * l.W[o*inDim+k]
				}
			}
		}

		// Weight decay
		if m.Config.WeightDecay > 0 {
			wd := m.Config.WeightDecay
			for j := range l.DW {
				l.DW[j] += wd * l.W[j]
			}
		}

		dx = dInput
	}
}

// BCELoss computes binary cross-entropy loss with optional label smoothing.
// predictions and targets shape: [batchSize, 1]. Returns scalar loss and gradient.
// DEPRECATED: Use BCEWithLogitsLoss for numerically stable training on raw logits.
func (m *MLP) BCELoss(predictions, targets []float32) (float32, []float32) {
	n := len(predictions)
	smooth := m.Config.LabelSmooth
	grad := make([]float32, n)
	loss := float32(0)

	for i := range predictions {
		p := predictions[i]
		t := targets[i]

		if smooth > 0 {
			t = t*(1.0-smooth) + 0.5*smooth
		}

		if p < 1e-7 { p = 1e-7 }
		if p > 1.0-1e-7 { p = 1.0 - 1e-7 }

		loss += -t*float32(math.Log(float64(p))) - (1.0-t)*float32(math.Log(float64(1.0-p)))
		grad[i] = (p - t) / float32(n)
	}

	return loss / float32(n), grad
}

// BCEWithLogitsLoss computes numerically stable binary cross-entropy on raw logits.
// logits shape: [batchSize, 1] (raw output before sigmoid).
// Uses: loss = max(x,0) - x*t + log(1 + exp(-|x|)) which avoids log(0).
// Gradient: sigmoid(x) - t, averaged over batch.
// posWeight > 0 scales the positive class (for imbalanced datasets).
func (m *MLP) BCEWithLogitsLoss(logits, targets []float32, posWeight float32) (float32, []float32) {
	n := len(logits)
	grad := make([]float32, n)
	loss := float32(0)
	smooth := m.Config.LabelSmooth

	for i := range logits {
		x := float64(logits[i])
		t := float64(targets[i])

		if smooth > 0 {
			t = t*(1.0-float64(smooth)) + 0.5*float64(smooth)
		}

		// Numerically stable: max(x,0) - x*t + log(1 + exp(-|x|))
		absX := x
		if absX < 0 { absX = -absX }
		l := math.Max(x, 0) - x*t + math.Log(1.0+math.Exp(-absX))

		pw := float64(posWeight)
		if pw > 0 {
			l = t*pw*(math.Log(1.0+math.Exp(-absX))+math.Max(-x, 0)) + (1-t)*(math.Log(1.0+math.Exp(-absX))+math.Max(x, 0))
		}

		loss += float32(l)

		sig := 1.0 / (1.0 + math.Exp(-x))
		if pw > 0 {
			grad[i] = float32((sig*(pw*t+(1-t)) - t*pw) / float64(n))
		} else {
			grad[i] = float32((sig - t) / float64(n))
		}
	}

	return loss / float32(n), grad
}

// Params returns all trainable parameters as flat slices (for optimizers).
func (m *MLP) Params() (params, grads [][]float32) {
	for i := range m.Layers {
		l := &m.Layers[i]
		params = append(params, l.W, l.B)
		grads = append(grads, l.DW, l.DB)
		if l.BNGamma != nil {
			params = append(params, l.BNGamma, l.BNBeta)
			grads = append(grads, l.BNDGamma, l.BNDBeta)
		}
	}
	return
}

// ParamCount returns total trainable parameters.
func (m *MLP) ParamCount() int {
	n := 0
	for _, l := range m.Layers {
		n += len(l.W) + len(l.B)
		if l.BNGamma != nil {
			n += len(l.BNGamma) + len(l.BNBeta)
		}
	}
	return n
}

// ToGPU uploads MLP weights to GPU tensors for accelerated forward/backward.
func (m *MLP) ToGPU(eng TensorEngine) {
	m.eng = eng
	for i := range m.Layers {
		l := &m.Layers[i]
		l.gW = eng.FromHost(l.W, []int{l.OutDim, l.InDim})
		l.gB = eng.FromHost(l.B, []int{1, l.OutDim})
		l.gDW = eng.Zeros([]int{l.OutDim, l.InDim})
		l.gDB = eng.Zeros([]int{1, l.OutDim})
		if l.BNGamma != nil {
			l.gBNGamma = eng.FromHost(l.BNGamma, []int{1, l.OutDim})
			l.gBNBeta = eng.FromHost(l.BNBeta, []int{1, l.OutDim})
			l.gRunMean = eng.FromHost(l.BNMean, []int{1, l.OutDim})
			l.gRunVar = eng.FromHost(l.BNVar, []int{1, l.OutDim})
			l.gSaveMean = eng.Zeros([]int{1, l.OutDim})
			l.gSaveInvStd = eng.Zeros([]int{1, l.OutDim})
		}
	}
}

// ToCPU downloads GPU weights back to CPU slices (for export/serialization).
func (m *MLP) ToCPU() {
	if m.eng == nil { return }
	for i := range m.Layers {
		l := &m.Layers[i]
		if l.gW != nil { copy(l.W, m.eng.ToHost(l.gW)) }
		if l.gB != nil { copy(l.B, m.eng.ToHost(l.gB)) }
		if l.gBNGamma != nil { copy(l.BNGamma, m.eng.ToHost(l.gBNGamma)) }
		if l.gBNBeta != nil { copy(l.BNBeta, m.eng.ToHost(l.gBNBeta)) }
	}
}

// ForwardGPU runs the MLP forward pass entirely on GPU.
// input: [batchSize, inputDim] as *Tensor. Caller must Release the returned tensor.
// Uses GPU kernels for bias, BatchNorm, ReLU, and dropout — zero CPU round-trips.
// Falls back to CPU-transfer path if MLP kernels aren't loaded.
func (m *MLP) ForwardGPU(input *Tensor, batchSize int, training bool) *Tensor {
	eng := m.eng

	if !HasMLPKernels() {
		return m.forwardGPUFallback(input, batchSize, training)
	}

	// Save input for backward (download once)
	l0 := &m.Layers[0]
	l0.input = eng.ToHost(input)

	var prevOut *Tensor
	curInput := input
	m.gpuDropoutCounter++

	for i := range m.Layers {
		l := &m.Layers[i]
		isLast := i == len(m.Layers)-1
		n := batchSize * l.OutDim

		if i > 0 {
			l.input = eng.ToHost(curInput)
		}

		// Linear: out = x @ W^T (cuBLAS)
		out := eng.MatMulTransposeBT(curInput, l.gW, batchSize, l.InDim, l.OutDim)
		if prevOut != nil {
			eng.Release(prevOut)
		}

		// Bias add (GPU)
		KBiasAdd(out.DevicePtr(), l.gB.DevicePtr(), batchSize, l.OutDim)

		// Save pre-activation for backward
		l.preAct = eng.ToHost(out)

		// BatchNorm (GPU)
		if l.BNGamma != nil && l.gBNGamma != nil {
			if l.gSaveMean == nil {
				l.gSaveMean = eng.Zeros([]int{1, l.OutDim})
				l.gSaveInvStd = eng.Zeros([]int{1, l.OutDim})
				l.gRunMean = eng.FromHost(l.BNMean, []int{1, l.OutDim})
				l.gRunVar = eng.FromHost(l.BNVar, []int{1, l.OutDim})
			}
			bnOut := eng.Zeros([]int{batchSize, l.OutDim})
			KBatchNormFwd(out.DevicePtr(), bnOut.DevicePtr(),
				l.gBNGamma.DevicePtr(), l.gBNBeta.DevicePtr(),
				l.gRunMean.DevicePtr(), l.gRunVar.DevicePtr(),
				l.gSaveMean.DevicePtr(), l.gSaveInvStd.DevicePtr(),
				batchSize, l.OutDim, m.Config.BNMomentum, 1e-5)
			eng.Release(out)
			out = bnOut
		}
		l.postBN = eng.ToHost(out)

		// Activation (GPU, hidden layers only)
		if !isLast {
			KReLU(out.DevicePtr(), n)
		}
		l.postAct = eng.ToHost(out)

		// Dropout (GPU, hidden layers, training only)
		if !isLast && training && m.Config.Dropout > 0 {
			if l.gMask == nil {
				l.gMask = eng.Zeros([]int{batchSize, l.OutDim})
			}
			KDropoutFwd(out.DevicePtr(), l.gMask.DevicePtr(), 42, m.gpuDropoutCounter, m.Config.Dropout, n)
			l.mask = eng.ToHost(l.gMask)
		} else {
			l.mask = nil
		}

		prevOut = out
		curInput = out
	}

	return prevOut
}

// forwardGPUFallback is the CPU-transfer path used when MLP kernels aren't loaded.
func (m *MLP) forwardGPUFallback(input *Tensor, batchSize int, training bool) *Tensor {
	eng := m.eng
	inputHost := eng.ToHost(input)
	l0 := &m.Layers[0]
	l0.input = inputHost

	var prevOut *Tensor
	curInput := input

	for i := range m.Layers {
		l := &m.Layers[i]
		isLast := i == len(m.Layers)-1

		if i > 0 { l.input = eng.ToHost(curInput) }

		out := eng.MatMulTransposeBT(curInput, l.gW, batchSize, l.InDim, l.OutDim)
		if prevOut != nil { eng.Release(prevOut) }

		outHost := eng.ToHost(out)
		eng.Release(out)
		for b := 0; b < batchSize; b++ {
			for j := 0; j < l.OutDim; j++ { outHost[b*l.OutDim+j] += l.B[j] }
		}
		l.preAct = make([]float32, len(outHost))
		copy(l.preAct, outHost)
		if l.BNGamma != nil { outHost = m.batchNorm(l, outHost, batchSize, l.OutDim, training) }
		l.postBN = make([]float32, len(outHost))
		copy(l.postBN, outHost)
		if !isLast { for j := range outHost { outHost[j] = m.activate(outHost[j]) } }
		l.postAct = make([]float32, len(outHost))
		copy(l.postAct, outHost)
		if !isLast && training && m.Config.Dropout > 0 {
			l.mask = make([]float32, len(outHost))
			scale := 1.0 / (1.0 - m.Config.Dropout)
			for j := range outHost {
				if m.rng.Float32() < m.Config.Dropout { outHost[j] = 0; l.mask[j] = 0 } else { outHost[j] *= float32(scale); l.mask[j] = float32(scale) }
			}
		} else { l.mask = nil }
		curOut := eng.FromHost(outHost, []int{batchSize, l.OutDim})
		prevOut = curOut
		curInput = curOut
	}
	return prevOut
}

// BackwardGPU computes gradients on GPU. dOut is gradient w.r.t. output logits.
// fromLogits=true skips sigmoid derivative (use with BCEWithLogitsLoss).
func (m *MLP) BackwardGPU(dOut []float32, batchSize int, fromLogits bool) {
	eng := m.eng
	dx := dOut

	for i := len(m.Layers) - 1; i >= 0; i-- {
		l := &m.Layers[i]
		outDim := l.OutDim
		inDim := l.InDim
		isLast := i == len(m.Layers)-1

		// Sigmoid backward
		if isLast && m.Config.Sigmoid && !fromLogits {
			for j := range dx {
				s := l.postAct[j]
				dx[j] *= s * (1.0 - s)
			}
		}

		// Dropout backward
		if l.mask != nil {
			for j := range dx { dx[j] *= l.mask[j] }
		}

		// Activation backward
		if !isLast {
			for j := range dx { dx[j] *= m.activateGrad(l.postBN[j]) }
		}

		// BatchNorm backward (CPU)
		if l.BNGamma != nil {
			dx = m.batchNormBackward(l, dx, batchSize, outDim)
		}

		// Weight gradient: dW = dOut^T @ input, dB = sum(dOut) — on GPU
		dxT := eng.FromHost(dx, []int{batchSize, outDim})
		inputT := eng.FromHost(l.input, []int{batchSize, inDim})
		dWGPU := eng.MatMulTransposeAT(dxT, inputT, batchSize, outDim, inDim)

		invBatch := 1.0 / float32(batchSize)
		dwHost := eng.ToHost(dWGPU)
		for j := range dwHost { dwHost[j] *= invBatch }
		copy(l.DW, dwHost)
		eng.Release(dWGPU)

		// dB = mean of dOut over batch
		for j := range l.DB { l.DB[j] = 0 }
		for b := 0; b < batchSize; b++ {
			for o := 0; o < outDim; o++ {
				l.DB[o] += dx[b*outDim+o] * invBatch
			}
		}

		// Input gradient: dInput = dOut @ W — on GPU
		dInputGPU := eng.MatMulT(dxT, l.gW, batchSize, outDim, inDim)
		dInput := eng.ToHost(dInputGPU)
		eng.Release(dInputGPU)
		eng.Release(dxT)
		eng.Release(inputT)

		// Weight decay
		if m.Config.WeightDecay > 0 {
			wd := m.Config.WeightDecay
			for j := range l.DW { l.DW[j] += wd * l.W[j] }
		}

		dx = dInput
	}
}

// SyncWeightsToGPU uploads current CPU weights to GPU after optimizer step.
func (m *MLP) SyncWeightsToGPU() {
	if m.eng == nil { return }
	for i := range m.Layers {
		l := &m.Layers[i]
		if l.gW != nil {
			m.eng.Release(l.gW)
			l.gW = m.eng.FromHost(l.W, []int{l.OutDim, l.InDim})
		}
		if l.gB != nil {
			m.eng.Release(l.gB)
			l.gB = m.eng.FromHost(l.B, []int{1, l.OutDim})
		}
	}
}

// MLPOptimizer is an AdamW optimizer with cosine annealing for MLP training.
type MLPOptimizer struct {
	LR         float32
	Beta1      float32
	Beta2      float32
	Eps        float32
	WeightDecay float32
	MaxEpochs  int
	m          [][]float32 // first moment
	v          [][]float32 // second moment
	step       int
}

// NewMLPOptimizer creates an AdamW optimizer matching PyTorch defaults.
func NewMLPOptimizer(lr, weightDecay float32, maxEpochs int) *MLPOptimizer {
	return &MLPOptimizer{
		LR: lr, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8,
		WeightDecay: weightDecay, MaxEpochs: maxEpochs,
	}
}

// Step performs one AdamW update with cosine annealing LR.
func (opt *MLPOptimizer) Step(params, grads [][]float32, epoch int) {
	if opt.m == nil {
		opt.m = make([][]float32, len(params))
		opt.v = make([][]float32, len(params))
		for i, p := range params {
			opt.m[i] = make([]float32, len(p))
			opt.v[i] = make([]float32, len(p))
		}
	}
	opt.step++

	// Cosine annealing
	lr := opt.LR
	if opt.MaxEpochs > 0 {
		lr = opt.LR * 0.5 * (1.0 + float32(math.Cos(math.Pi*float64(epoch)/float64(opt.MaxEpochs))))
	}

	bc1 := 1.0 - float32(math.Pow(float64(opt.Beta1), float64(opt.step)))
	bc2 := 1.0 - float32(math.Pow(float64(opt.Beta2), float64(opt.step)))

	for i := range params {
		for j := range params[i] {
			g := grads[i][j]
			opt.m[i][j] = opt.Beta1*opt.m[i][j] + (1-opt.Beta1)*g
			opt.v[i][j] = opt.Beta2*opt.v[i][j] + (1-opt.Beta2)*g*g
			mHat := opt.m[i][j] / bc1
			vHat := opt.v[i][j] / bc2
			params[i][j] -= lr * (mHat/(float32(math.Sqrt(float64(vHat)))+opt.Eps) + opt.WeightDecay*params[i][j])
		}
	}
}

// --- internals ---

func (m *MLP) activate(x float32) float32 {
	switch m.Config.Activation {
	case "silu":
		return x / (1.0 + float32(math.Exp(-float64(x))))
	case "tanh":
		return float32(math.Tanh(float64(x)))
	case "none":
		return x
	default: // relu
		if x > 0 { return x }
		return 0
	}
}

func (m *MLP) activateGrad(x float32) float32 {
	switch m.Config.Activation {
	case "silu":
		s := 1.0 / (1.0 + float32(math.Exp(-float64(x))))
		return s + x*s*(1.0-s)
	case "tanh":
		t := float32(math.Tanh(float64(x)))
		return 1.0 - t*t
	case "none":
		return 1.0
	default: // relu
		if x > 0 { return 1.0 }
		return 0
	}
}

func (m *MLP) batchNorm(l *MLPLayer, x []float32, batchSize, dim int, training bool) []float32 {
	out := make([]float32, len(x))
	eps := float32(1e-5)
	mom := m.Config.BNMomentum

	if training {
		// Compute batch mean and variance
		mean := make([]float32, dim)
		variance := make([]float32, dim)
		invN := 1.0 / float32(batchSize)

		for b := 0; b < batchSize; b++ {
			for d := 0; d < dim; d++ {
				mean[d] += x[b*dim+d]
			}
		}
		for d := range mean { mean[d] *= invN }

		for b := 0; b < batchSize; b++ {
			for d := 0; d < dim; d++ {
				diff := x[b*dim+d] - mean[d]
				variance[d] += diff * diff
			}
		}
		for d := range variance { variance[d] *= invN }

		// Update running stats
		for d := 0; d < dim; d++ {
			l.BNMean[d] = (1.0-mom)*l.BNMean[d] + mom*mean[d]
			l.BNVar[d] = (1.0-mom)*l.BNVar[d] + mom*variance[d]
		}

		// Normalize
		for b := 0; b < batchSize; b++ {
			for d := 0; d < dim; d++ {
				xhat := (x[b*dim+d] - mean[d]) / float32(math.Sqrt(float64(variance[d]+eps)))
				out[b*dim+d] = l.BNGamma[d]*xhat + l.BNBeta[d]
			}
		}
	} else {
		// Use running stats
		for b := 0; b < batchSize; b++ {
			for d := 0; d < dim; d++ {
				xhat := (x[b*dim+d] - l.BNMean[d]) / float32(math.Sqrt(float64(l.BNVar[d]+eps)))
				out[b*dim+d] = l.BNGamma[d]*xhat + l.BNBeta[d]
			}
		}
	}

	return out
}

func (m *MLP) batchNormBackward(l *MLPLayer, dOut []float32, batchSize, dim int) []float32 {
	eps := float32(1e-5)
	invN := 1.0 / float32(batchSize)

	// Recompute batch stats from saved preAct
	mean := make([]float32, dim)
	variance := make([]float32, dim)
	for b := 0; b < batchSize; b++ {
		for d := 0; d < dim; d++ {
			mean[d] += l.preAct[b*dim+d]
		}
	}
	for d := range mean { mean[d] *= invN }
	for b := 0; b < batchSize; b++ {
		for d := 0; d < dim; d++ {
			diff := l.preAct[b*dim+d] - mean[d]
			variance[d] += diff * diff
		}
	}
	for d := range variance { variance[d] *= invN }

	// Compute xhat and gradients
	xhat := make([]float32, batchSize*dim)
	for b := 0; b < batchSize; b++ {
		for d := 0; d < dim; d++ {
			xhat[b*dim+d] = (l.preAct[b*dim+d] - mean[d]) / float32(math.Sqrt(float64(variance[d]+eps)))
		}
	}

	// dGamma, dBeta
	for j := range l.BNDGamma { l.BNDGamma[j] = 0 }
	for j := range l.BNDBeta { l.BNDBeta[j] = 0 }
	for b := 0; b < batchSize; b++ {
		for d := 0; d < dim; d++ {
			l.BNDGamma[d] += dOut[b*dim+d] * xhat[b*dim+d] * invN
			l.BNDBeta[d] += dOut[b*dim+d] * invN
		}
	}

	// dInput
	dx := make([]float32, batchSize*dim)
	for b := 0; b < batchSize; b++ {
		for d := 0; d < dim; d++ {
			invStd := 1.0 / float32(math.Sqrt(float64(variance[d]+eps)))
			dxhat := dOut[b*dim+d] * l.BNGamma[d]
			dx[b*dim+d] = invStd * (dxhat - invN*(l.BNDBeta[d]*float32(batchSize)+l.BNDGamma[d]*float32(batchSize)*xhat[b*dim+d]))
		}
	}

	return dx
}
