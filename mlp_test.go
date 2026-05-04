package mongoose

import (
	"math"
	"testing"
)

func TestMLPForwardShape(t *testing.T) {
	mlp := NewMLP([]int{10, 512, 256, 128, 1}, MLPConfig{
		Activation: "relu",
		Dropout:    0.2,
		BatchNorm:  true,
		Sigmoid:    true,
	})

	batchSize := 4
	input := make([]float32, batchSize*10)
	for i := range input { input[i] = float32(i) * 0.01 }

	out := mlp.Forward(input, batchSize, true)
	if len(out) != batchSize*1 {
		t.Fatalf("expected output size %d, got %d", batchSize, len(out))
	}
	for i, v := range out {
		if v < 0 || v > 1 {
			t.Errorf("sigmoid output[%d] = %f, expected [0,1]", i, v)
		}
	}
}

func TestMLPBackward(t *testing.T) {
	mlp := NewMLP([]int{4, 8, 1}, MLPConfig{
		Activation: "relu",
		Sigmoid:    true,
	})

	batchSize := 2
	input := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	targets := []float32{1, 0}

	out := mlp.Forward(input, batchSize, true)
	loss, grad := mlp.BCELoss(out, targets)
	mlp.Backward(grad, batchSize)

	if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
		t.Fatalf("loss is %f", loss)
	}

	// Check gradients are non-zero
	hasNonZero := false
	for _, l := range mlp.Layers {
		for _, g := range l.DW {
			if g != 0 { hasNonZero = true; break }
		}
	}
	if !hasNonZero {
		t.Fatal("all gradients are zero")
	}
}

func TestMLPParamCount(t *testing.T) {
	mlp := NewMLP([]int{10, 512, 256, 128, 1}, MLPConfig{BatchNorm: true})
	n := mlp.ParamCount()
	// 10*512+512 + 512*256+256 + 256*128+128 + 128*1+1 + 3*(gamma+beta)
	expected := 10*512 + 512 + 512*256 + 256 + 256*128 + 128 + 128*1 + 1 + (512+256+128)*2
	if n != expected {
		t.Errorf("param count %d, expected %d", n, expected)
	}
}

func TestMLPBCELoss(t *testing.T) {
	mlp := NewMLP([]int{1, 1}, MLPConfig{Sigmoid: true})
	pred := []float32{0.9, 0.1}
	target := []float32{1.0, 0.0}

	loss, grad := mlp.BCELoss(pred, target)
	if loss > 0.2 {
		t.Errorf("loss %.4f too high for correct predictions", loss)
	}
	if len(grad) != 2 {
		t.Fatalf("grad length %d, expected 2", len(grad))
	}
}

func TestMLPNoSigmoid(t *testing.T) {
	mlp := NewMLP([]int{4, 8, 2}, MLPConfig{Activation: "relu"})
	input := make([]float32, 2*4)
	for i := range input { input[i] = float32(i) }

	out := mlp.Forward(input, 2, false)
	if len(out) != 4 {
		t.Fatalf("expected 4 outputs, got %d", len(out))
	}
	// Without sigmoid, outputs can be any value
}
