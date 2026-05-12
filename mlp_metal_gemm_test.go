//go:build darwin && cgo

package mongoose

import (
	"math"
	"testing"
)

func TestMLPMetal_GemmBT_Direct(t *testing.T) {
	eng := NewMetal()
	if eng == nil { t.Skip("no Metal") }

	mlp := NewMLP([]int{3, 2}, MLPConfig{
		Activation: "none", // no ReLU to keep the matmul output intact
		BatchNorm:  false,
		Dropout:    0,
		Sigmoid:    false,
	})
	// Override weights: W = B = [[1,0,1],[0,1,0]]
	copy(mlp.Layers[0].W, []float32{1, 0, 1, 0, 1, 0})
	// Bias = 0 (already initialized to 0)

	mtlMLP := NewMLPMetal(eng, mlp, 2)
	if mtlMLP == nil { t.Fatal("MLPMetal creation failed") }

	// Input = A = [[1,2,3],[4,5,6]]
	// Targets don't matter for forward check
	mtlMLP.UploadBatch([]float32{1, 2, 3, 4, 5, 6}, []float32{0, 0})
	mtlMLP.TrainStep(0.0)

	// Check act[0] = the matmul+bias output (saved in preBN)
	preBN := (*[1 << 30]float32)(mtlBufSharedPtr(mtlMLP.preBN[0]))[:4]
	t.Logf("preBN (matmul output): [%.2f, %.2f, %.2f, %.2f]", preBN[0], preBN[1], preBN[2], preBN[3])
	t.Logf("Expected: [4.00, 2.00, 10.00, 5.00]")

	expected := []float32{4, 2, 10, 5}
	for i, exp := range expected {
		diff := math.Abs(float64(preBN[i] - exp))
		if diff > 0.01 {
			t.Errorf("preBN[%d] = %.4f, expected %.4f (diff=%.4e)", i, preBN[i], exp, diff)
		}
	}
}
