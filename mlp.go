package mongoose

import (
	"math"
	"math/rand"
)

// MLP is a generic multi-layer perceptron with configurable layer sizes,
// BatchNorm, ReLU activation, and Dropout between hidden layers.
// The final layer has no activation (raw logits) or optional sigmoid.
//
// Example: binary classification
//   mlp := NewMLP([]int{inputDim, 512, 256, 128, 1}, MLPConfig{
//       Activation: "relu",
//       Dropout:    0.2,
//       BatchNorm:  true,
//       Sigmoid:    true,
//   })
type MLP struct {
	Layers []MLPLayer
	Config MLPConfig
	rng    *rand.Rand
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

// Forward runs the MLP on a batch of inputs.
// input shape: [batchSize, inputDim], returns [batchSize, outputDim].
// training=true enables dropout and uses batch stats for BatchNorm.
func (m *MLP) Forward(input []float32, batchSize int, training bool) []float32 {
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

		// Sigmoid on final layer if configured
		if isLast && m.Config.Sigmoid {
			for j := range out {
				out[j] = 1.0 / (1.0 + float32(math.Exp(-float64(out[j]))))
			}
		}

		x = out
		inDim = outDim
	}

	return x
}

// Backward computes gradients given the loss gradient w.r.t. the output.
// dOut shape: [batchSize, outputDim]. Populates DW, DB, BNDGamma, BNDBeta.
func (m *MLP) Backward(dOut []float32, batchSize int) {
	dx := dOut

	for i := len(m.Layers) - 1; i >= 0; i-- {
		l := &m.Layers[i]
		outDim := l.OutDim
		inDim := l.InDim
		isLast := i == len(m.Layers)-1

		// Sigmoid backward
		if isLast && m.Config.Sigmoid {
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
func (m *MLP) BCELoss(predictions, targets []float32) (float32, []float32) {
	n := len(predictions)
	smooth := m.Config.LabelSmooth
	grad := make([]float32, n)
	loss := float32(0)

	for i := range predictions {
		p := predictions[i]
		t := targets[i]

		// Label smoothing
		if smooth > 0 {
			t = t*(1.0-smooth) + 0.5*smooth
		}

		// Clamp for numerical stability
		if p < 1e-7 { p = 1e-7 }
		if p > 1.0-1e-7 { p = 1.0 - 1e-7 }

		loss += -t*float32(math.Log(float64(p))) - (1.0-t)*float32(math.Log(float64(1.0-p)))
		grad[i] = (p - t) / float32(n)
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
