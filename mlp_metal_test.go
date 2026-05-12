//go:build darwin && cgo

package mongoose

import (
	"math"
	"testing"
)

func TestMLPMetal_ForwardMatchesCPU(t *testing.T) {
	eng := NewMetal()
	if eng == nil {
		t.Skip("Metal not available")
	}

	mlp := NewMLP([]int{4, 8, 1}, MLPConfig{
		Activation: "relu",
		BatchNorm:  true,
		Dropout:    0,
		Sigmoid:    true,
		BNMomentum: 0.1,
	})

	batchSize := 4
	input := []float32{
		0.1, 0.2, 0.3, 0.4,
		0.5, 0.6, 0.7, 0.8,
		-0.1, -0.2, 0.1, 0.5,
		0.3, -0.1, 0.4, 0.2,
	}
	targets := []float32{1, 0, 1, 0}

	cpuLogits := mlp.ForwardLogits(input, batchSize, true)
	cpuLoss, _ := mlp.BCEWithLogitsLoss(cpuLogits, targets, 0)

	mtl := NewMLPMetal(eng, mlp, batchSize)
	if mtl == nil {
		t.Skip("MLPMetal not available (metallib not found)")
	}
	defer mtl.Destroy()

	mtl.UploadBatch(input, targets)
	gpuLoss := mtl.TrainStep(0.0003)

	lossDiff := math.Abs(float64(gpuLoss - cpuLoss))
	t.Logf("CPU loss=%.6f  GPU loss=%.6f  diff=%.6e", cpuLoss, gpuLoss, lossDiff)
	if lossDiff > 0.05 {
		t.Errorf("loss diverged beyond tolerance: CPU=%.6f GPU=%.6f diff=%.6e", cpuLoss, gpuLoss, lossDiff)
	}
}

func TestMLPMetal_CPUBaseline(t *testing.T) {
	mlp := NewMLP([]int{2, 8, 1}, MLPConfig{
		Activation: "relu", BatchNorm: false, Dropout: 0, Sigmoid: true, WeightDecay: 0,
	})
	batchSize := 8
	input := []float32{0,0, 0.2,0.3, 0.1,0.4, 0.3,0.2, 0.8,0.9, 0.9,0.6, 0.7,0.8, 1.0,1.0}
	targets := []float32{0, 0, 0, 0, 1, 1, 1, 1}
	opt := NewMLPOptimizer(0.0003, 0, 2000)
	for step := 0; step < 2000; step++ {
		logits := mlp.ForwardLogits(input, batchSize, true)
		loss, grad := mlp.BCEWithLogitsLoss(logits, targets, 0)
		mlp.BackwardFromLogits(grad, batchSize)
		params, grads := mlp.Params()
		opt.Step(params, grads, 0)
		if step%500 == 0 { t.Logf("CPU step %d: loss=%.6f", step, loss) }
	}
	logits := mlp.ForwardLogits(input, batchSize, false)
	loss, _ := mlp.BCEWithLogitsLoss(logits, targets, 0)
	t.Logf("CPU final loss=%.6f", loss)
	if loss > 0.3 { t.Errorf("CPU didn't converge: loss=%.6f", loss) }
}

func TestMLPMetal_TrainingConverges(t *testing.T) {
	eng := NewMetal()
	if eng == nil {
		t.Skip("Metal not available")
	}

	batchSize := 8
	input := []float32{
		0.0, 0.0, 0.2, 0.3, 0.1, 0.4, 0.3, 0.2,
		0.8, 0.9, 0.9, 0.6, 0.7, 0.8, 1.0, 1.0,
	}
	targets := []float32{0, 0, 0, 0, 1, 1, 1, 1}

	// Run CPU first
	mlpCPU := NewMLP([]int{2, 8, 1}, MLPConfig{
		Activation: "relu", BatchNorm: false, Dropout: 0, Sigmoid: true, WeightDecay: 0,
	})
	// Clone weights for Metal
	mlpGPU := NewMLP([]int{2, 8, 1}, MLPConfig{
		Activation: "relu", BatchNorm: false, Dropout: 0, Sigmoid: true, WeightDecay: 0,
	})
	for i := range mlpGPU.Layers {
		copy(mlpGPU.Layers[i].W, mlpCPU.Layers[i].W)
		copy(mlpGPU.Layers[i].B, mlpCPU.Layers[i].B)
	}

	// CPU training
	opt := NewMLPOptimizer(0.0003, 0, 500)
	for step := 0; step < 500; step++ {
		logits := mlpCPU.ForwardLogits(input, batchSize, true)
		loss, grad := mlpCPU.BCEWithLogitsLoss(logits, targets, 0)
		mlpCPU.BackwardFromLogits(grad, batchSize)
		params, grads := mlpCPU.Params()
		opt.Step(params, grads, 0)
		if step%100 == 0 {
			t.Logf("CPU step %d: loss=%.6f", step, loss)
		}
	}

	// Check CPU forward on mlpGPU BEFORE Metal training
	cpuLogits0 := mlpGPU.ForwardLogits(input, batchSize, false)
	cpuLoss0, _ := mlpGPU.BCEWithLogitsLoss(cpuLogits0, targets, 0)
	t.Logf("Pre-Metal CPU check: loss=%.6f logits=%.4f,%.4f,%.4f,%.4f",
		cpuLoss0, cpuLogits0[0], cpuLogits0[1], cpuLogits0[2], cpuLogits0[3])

	// GPU training with identical initial weights
	mtl := NewMLPMetal(eng, mlpGPU, batchSize)
	if mtl == nil {
		t.Skip("MLPMetal not available")
	}
	defer mtl.Destroy()

	mtl.UploadBatch(input, targets)
	for step := 0; step < 500; step++ {
		loss := mtl.TrainStep(0.0003)
		if step%100 == 0 {
			t.Logf("GPU step %d: loss=%.6f", step, loss)
		}
	}

	// Both should have converged
	cpuLogits := mlpCPU.ForwardLogits(input, batchSize, false)
	cpuLoss, _ := mlpCPU.BCEWithLogitsLoss(cpuLogits, targets, 0)

	mtl.DownloadWeights()
	gpuLogits := mlpGPU.ForwardLogits(input, batchSize, false)
	gpuLoss, _ := mlpGPU.BCEWithLogitsLoss(gpuLogits, targets, 0)

	t.Logf("GPU W[0][0]=%.6f (initial was %.6f)", mlpGPU.Layers[0].W[0], mlpCPU.Layers[0].W[0])
	t.Logf("Final CPU loss=%.6f  GPU loss=%.6f", cpuLoss, gpuLoss)
	if gpuLoss > 0.6 {
		t.Errorf("GPU didn't converge: loss=%.6f (CPU=%.6f)", gpuLoss, cpuLoss)
	}
}
