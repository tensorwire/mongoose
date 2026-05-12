//go:build darwin && cgo

package mongoose

import (
	"math"
	"testing"
)

func initMetal(t *testing.T) {
	t.Helper()
	eng := NewMetal()
	if eng == nil {
		t.Skip("Metal not available")
	}
	if !MtlMLPInit() {
		t.Skip("mlp_train.metallib not loaded")
	}
}

func assertClose(t *testing.T, name string, got, want float32, tol float64) {
	t.Helper()
	diff := math.Abs(float64(got - want))
	if diff > tol {
		t.Errorf("%s: got %.6f, want %.6f (diff=%.2e, tol=%.2e)", name, got, want, diff, tol)
	}
}

// ==================== GEMM Tests ====================

func TestMLPMetal_GemmTN_Direct(t *testing.T) {
	initMetal(t)

	// gemm_tn: C[K,N] = A[M,K]^T @ B[M,N]
	// A[3,2] = [[1,2],[3,4],[5,6]]   (M=3, K=2)
	// B[3,2] = [[7,8],[9,10],[11,12]] (M=3, N=2)
	// A^T[2,3] = [[1,3,5],[2,4,6]]
	// C = A^T @ B = [[1*7+3*9+5*11, 1*8+3*10+5*12], [2*7+4*9+6*11, 2*8+4*10+6*12]]
	//             = [[7+27+55, 8+30+60], [14+36+66, 16+40+72]]
	//             = [[89, 98], [116, 128]]

	a := MtlBufFromHost([]float32{1, 2, 3, 4, 5, 6})
	b := MtlBufFromHost([]float32{7, 8, 9, 10, 11, 12})
	c := MtlBufZeros(4)

	MtlGemmTN(a, b, c, 3, 2, 2)

	out := MtlBufSharedSlice(c, 4)
	t.Logf("gemm_tn output: [%.1f, %.1f, %.1f, %.1f]", out[0], out[1], out[2], out[3])

	expected := []float32{89, 98, 116, 128}
	for i, exp := range expected {
		assertClose(t, "C["+string(rune('0'+i))+"]", out[i], exp, 0.01)
	}
}

func TestMLPMetal_GemmNN_Direct(t *testing.T) {
	initMetal(t)

	// gemm_nn: C[M,N] = A[M,K] @ B[K,N]
	// A[2,3] = [[1,2,3],[4,5,6]]   (M=2, K=3)
	// B[3,2] = [[7,8],[9,10],[11,12]] (K=3, N=2)
	// C = A @ B = [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
	//           = [[7+18+33, 8+20+36], [28+45+66, 32+50+72]]
	//           = [[58, 64], [139, 154]]

	a := MtlBufFromHost([]float32{1, 2, 3, 4, 5, 6})
	b := MtlBufFromHost([]float32{7, 8, 9, 10, 11, 12})
	c := MtlBufZeros(4)

	MtlGemmNN(a, b, c, 2, 3, 2)

	out := MtlBufSharedSlice(c, 4)
	t.Logf("gemm_nn output: [%.1f, %.1f, %.1f, %.1f]", out[0], out[1], out[2], out[3])

	expected := []float32{58, 64, 139, 154}
	for i, exp := range expected {
		assertClose(t, "C["+string(rune('0'+i))+"]", out[i], exp, 0.01)
	}
}

// ==================== Bias Add ====================

func TestMLPMetal_BiasAdd_Direct(t *testing.T) {
	initMetal(t)

	// out[b*D+d] += bias[d]
	// B=2, D=3
	// input: [[1,2,3],[4,5,6]]
	// bias:  [10, 20, 30]
	// expected: [[11,22,33],[14,25,36]]

	out := MtlBufFromHost([]float32{1, 2, 3, 4, 5, 6})
	bias := MtlBufFromHost([]float32{10, 20, 30})

	MtlBiasAdd(out, bias, 2, 3)

	result := MtlBufSharedSlice(out, 6)
	expected := []float32{11, 22, 33, 14, 25, 36}
	for i, exp := range expected {
		assertClose(t, "out["+string(rune('0'+i))+"]", result[i], exp, 0.01)
	}
}

// ==================== BatchNorm Forward ====================

func TestMLPMetal_BNForward_Direct(t *testing.T) {
	initMetal(t)

	B, D := 4, 2
	input := []float32{
		1.0, 2.0,
		3.0, 4.0,
		5.0, 6.0,
		7.0, 8.0,
	}
	gamma := []float32{1.0, 1.0}
	beta := []float32{0.0, 0.0}

	// CPU reference
	cpuMLP := NewMLP([]int{2, 2, 1}, MLPConfig{BatchNorm: true, Activation: "relu", BNMomentum: 0.1})
	copy(cpuMLP.Layers[0].BNGamma, gamma)
	copy(cpuMLP.Layers[0].BNBeta, beta)
	for i := range cpuMLP.Layers[0].BNMean {
		cpuMLP.Layers[0].BNMean[i] = 0
		cpuMLP.Layers[0].BNVar[i] = 1
	}
	cpuOut := cpuMLP.batchNorm(&cpuMLP.Layers[0], append([]float32{}, input...), B, D, true)

	// GPU
	x := MtlBufFromHost(input)
	meanBuf := MtlBufZeros(D)
	varBuf := MtlBufZeros(D)
	gammaBuf := MtlBufFromHost(gamma)
	betaBuf := MtlBufFromHost(beta)
	runMean := MtlBufZeros(D)
	runVar := MtlBufFromHost([]float32{1.0, 1.0})

	MtlBNForward(x, meanBuf, varBuf, gammaBuf, betaBuf, runMean, runVar, B, D, 0.1)

	gpuOut := MtlBufSharedSlice(x, B*D)
	gpuMean := MtlBufSharedSlice(meanBuf, D)
	gpuVar := MtlBufSharedSlice(varBuf, D)
	gpuRunMean := MtlBufSharedSlice(runMean, D)
	gpuRunVar := MtlBufSharedSlice(runVar, D)

	t.Logf("GPU mean: [%.4f, %.4f]", gpuMean[0], gpuMean[1])
	t.Logf("GPU var:  [%.4f, %.4f]", gpuVar[0], gpuVar[1])
	t.Logf("GPU runMean: [%.4f, %.4f]", gpuRunMean[0], gpuRunMean[1])
	t.Logf("GPU runVar:  [%.4f, %.4f]", gpuRunVar[0], gpuRunVar[1])

	// mean should be [4.0, 5.0], var should be [5.0, 5.0]
	assertClose(t, "mean[0]", gpuMean[0], 4.0, 1e-5)
	assertClose(t, "mean[1]", gpuMean[1], 5.0, 1e-5)
	assertClose(t, "var[0]", gpuVar[0], 5.0, 1e-5)
	assertClose(t, "var[1]", gpuVar[1], 5.0, 1e-5)

	// Compare normalized output element-by-element
	for i := 0; i < B*D; i++ {
		assertClose(t, "bn_out["+itoa(i)+"]", gpuOut[i], cpuOut[i], 1e-5)
	}

	// Running stats: runMean = 0.9*0 + 0.1*4 = 0.4, 0.5
	assertClose(t, "runMean[0]", gpuRunMean[0], 0.4, 1e-5)
	assertClose(t, "runMean[1]", gpuRunMean[1], 0.5, 1e-5)
	// runVar = 0.9*1 + 0.1*5 = 1.4
	assertClose(t, "runVar[0]", gpuRunVar[0], 1.4, 1e-5)
	assertClose(t, "runVar[1]", gpuRunVar[1], 1.4, 1e-5)
}

// ==================== BCE Loss ====================

func TestMLPMetal_BCELoss_Direct(t *testing.T) {
	initMetal(t)

	B := 4
	logits := []float32{-1.0, 0.5, 2.0, -0.3}
	targets := []float32{0.0, 1.0, 1.0, 0.0}

	// CPU reference
	mlp := NewMLP([]int{1, 1}, MLPConfig{})
	cpuLoss, cpuGrad := mlp.BCEWithLogitsLoss(logits, targets, 0)

	// GPU
	logitsBuf := MtlBufFromHost(logits)
	targetsBuf := MtlBufFromHost(targets)
	gradBuf := MtlBufZeros(B)
	lossBuf := MtlBufZeros(B)
	lossScalar := MtlBufZeros(1)

	MtlBCE(logitsBuf, targetsBuf, gradBuf, lossBuf, lossScalar, B)

	gpuLoss := MtlBufSharedSlice(lossScalar, 1)[0]
	gpuGrad := MtlBufSharedSlice(gradBuf, B)

	t.Logf("CPU loss=%.6f  GPU loss=%.6f", cpuLoss, gpuLoss)
	assertClose(t, "loss", gpuLoss, cpuLoss, 1e-5)

	for i := 0; i < B; i++ {
		t.Logf("grad[%d]: CPU=%.6f GPU=%.6f", i, cpuGrad[i], gpuGrad[i])
		assertClose(t, "grad["+itoa(i)+"]", gpuGrad[i], cpuGrad[i], 1e-5)
	}
}

// ==================== AdamW ====================

func TestMLPMetal_AdamW_Direct(t *testing.T) {
	initMetal(t)

	n := 4
	param := []float32{1.0, -0.5, 0.3, 0.8}
	grad := []float32{0.1, -0.2, 0.05, 0.15}
	m := []float32{0, 0, 0, 0}
	v := []float32{0, 0, 0, 0}

	lr := float32(0.001)
	beta1 := float32(0.9)
	beta2 := float32(0.999)
	eps := float32(1e-8)
	wd := float32(0.01)
	step := 1
	bc1 := float32(1.0 - math.Pow(float64(beta1), float64(step)))
	bc2 := float32(1.0 - math.Pow(float64(beta2), float64(step)))

	// CPU reference
	cpuParam := append([]float32{}, param...)
	cpuM := append([]float32{}, m...)
	cpuV := append([]float32{}, v...)
	for j := 0; j < n; j++ {
		g := grad[j]
		cpuM[j] = beta1*cpuM[j] + (1-beta1)*g
		cpuV[j] = beta2*cpuV[j] + (1-beta2)*g*g
		mHat := cpuM[j] / bc1
		vHat := cpuV[j] / bc2
		cpuParam[j] -= lr * (mHat/(float32(math.Sqrt(float64(vHat)))+eps) + wd*cpuParam[j])
	}

	// GPU
	paramBuf := MtlBufFromHost(param)
	gradBuf := MtlBufFromHost(grad)
	mBuf := MtlBufFromHost(m)
	vBuf := MtlBufFromHost(v)

	MtlAdamWStep(paramBuf, gradBuf, mBuf, vBuf, lr, beta1, beta2, bc1, bc2, eps, wd, n)

	gpuParam := MtlBufSharedSlice(paramBuf, n)
	gpuM := MtlBufSharedSlice(mBuf, n)
	gpuV := MtlBufSharedSlice(vBuf, n)

	for j := 0; j < n; j++ {
		t.Logf("param[%d]: CPU=%.8f GPU=%.8f", j, cpuParam[j], gpuParam[j])
		assertClose(t, "param["+itoa(j)+"]", gpuParam[j], cpuParam[j], 1e-6)
		assertClose(t, "m["+itoa(j)+"]", gpuM[j], cpuM[j], 1e-6)
		assertClose(t, "v["+itoa(j)+"]", gpuV[j], cpuV[j], 1e-6)
	}
}

// ==================== Dropout ====================

func TestMLPMetal_Dropout_Direct(t *testing.T) {
	initMetal(t)

	n := 10000
	p := float32(0.3)

	input := make([]float32, n)
	for i := range input {
		input[i] = 1.0
	}

	x := MtlBufFromHost(input)
	mask := MtlBufZeros(n)

	MtlDropoutFwd(x, mask, n, p, 42, 1)

	gpuX := MtlBufSharedSlice(x, n)
	gpuMask := MtlBufSharedSlice(mask, n)

	scale := float32(1.0 / (1.0 - p))
	var zeroCount int
	for i := 0; i < n; i++ {
		m := gpuMask[i]
		if m == 0 {
			if gpuX[i] != 0 {
				t.Errorf("mask[%d]=0 but x[%d]=%.4f (expected 0)", i, i, gpuX[i])
			}
			zeroCount++
		} else {
			assertClose(t, "mask_scale", m, scale, 1e-5)
			assertClose(t, "x_scaled", gpuX[i], scale, 1e-5)
		}
	}

	dropRate := float64(zeroCount) / float64(n)
	t.Logf("dropout rate: %.3f (expected ~%.3f), zeros=%d/%d", dropRate, p, zeroCount, n)
	if math.Abs(dropRate-float64(p)) > 0.05 {
		t.Errorf("dropout rate %.3f too far from expected %.3f", dropRate, p)
	}

	// Test backward: dx *= mask
	dx := MtlBufFromHost(make([]float32, n))
	dxSlice := MtlBufSharedSlice(dx, n)
	for i := range dxSlice {
		dxSlice[i] = 2.0
	}

	MtlDropoutBwd(dx, mask, n)

	gpuDx := MtlBufSharedSlice(dx, n)
	for i := 0; i < min(20, n); i++ {
		expected := float32(2.0) * gpuMask[i]
		assertClose(t, "dx_bwd["+itoa(i)+"]", gpuDx[i], expected, 1e-5)
	}
}

// ==================== BatchNorm Backward ====================

func TestMLPMetal_BNBackward_Direct(t *testing.T) {
	initMetal(t)

	B, D := 4, 3
	// Pre-BN input (the saved x before normalization)
	preBN := []float32{
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
		7.0, 8.0, 9.0,
		10.0, 11.0, 12.0,
	}
	gamma := []float32{1.0, 1.0, 1.0}
	dOut := []float32{
		0.1, -0.2, 0.3,
		-0.1, 0.4, -0.5,
		0.2, -0.3, 0.6,
		-0.2, 0.1, -0.4,
	}

	// CPU reference: compute batch stats, then BN backward
	cpuMLP := NewMLP([]int{D, D, 1}, MLPConfig{BatchNorm: true, Activation: "relu", BNMomentum: 0.1})
	l := &cpuMLP.Layers[0]
	copy(l.BNGamma, gamma)
	for i := range l.BNBeta {
		l.BNBeta[i] = 0
	}
	l.preAct = append([]float32{}, preBN...)
	cpuDx := cpuMLP.batchNormBackward(l, append([]float32{}, dOut...), B, D)
	cpuDGamma := append([]float32{}, l.BNDGamma...)
	cpuDBeta := append([]float32{}, l.BNDBeta...)

	// Compute mean/var on CPU for GPU input (GPU bn_backward expects pre-computed mean/var)
	mean := make([]float32, D)
	variance := make([]float32, D)
	invN := 1.0 / float32(B)
	for b := 0; b < B; b++ {
		for d := 0; d < D; d++ {
			mean[d] += preBN[b*D+d]
		}
	}
	for d := range mean {
		mean[d] *= invN
	}
	for b := 0; b < B; b++ {
		for d := 0; d < D; d++ {
			diff := preBN[b*D+d] - mean[d]
			variance[d] += diff * diff
		}
	}
	for d := range variance {
		variance[d] *= invN
	}

	// GPU
	dOutBuf := MtlBufFromHost(append([]float32{}, dOut...))
	xBuf := MtlBufFromHost(preBN)
	meanBuf := MtlBufFromHost(mean)
	varBuf := MtlBufFromHost(variance)
	gammaBuf := MtlBufFromHost(gamma)
	dGammaBuf := MtlBufZeros(D)
	dBetaBuf := MtlBufZeros(D)

	MtlBNBackward(dOutBuf, xBuf, meanBuf, varBuf, gammaBuf, dGammaBuf, dBetaBuf, B, D)

	gpuDx := MtlBufSharedSlice(dOutBuf, B*D)
	gpuDGamma := MtlBufSharedSlice(dGammaBuf, D)
	gpuDBeta := MtlBufSharedSlice(dBetaBuf, D)

	t.Logf("CPU dGamma: [%.6f, %.6f, %.6f]", cpuDGamma[0], cpuDGamma[1], cpuDGamma[2])
	t.Logf("GPU dGamma: [%.6f, %.6f, %.6f]", gpuDGamma[0], gpuDGamma[1], gpuDGamma[2])
	t.Logf("CPU dBeta:  [%.6f, %.6f, %.6f]", cpuDBeta[0], cpuDBeta[1], cpuDBeta[2])
	t.Logf("GPU dBeta:  [%.6f, %.6f, %.6f]", gpuDBeta[0], gpuDBeta[1], gpuDBeta[2])

	for d := 0; d < D; d++ {
		assertClose(t, "dGamma["+itoa(d)+"]", gpuDGamma[d], cpuDGamma[d], 1e-4)
		assertClose(t, "dBeta["+itoa(d)+"]", gpuDBeta[d], cpuDBeta[d], 1e-4)
	}

	for i := 0; i < B*D; i++ {
		assertClose(t, "dx["+itoa(i)+"]", gpuDx[i], cpuDx[i], 1e-4)
	}
}

// ==================== Full Forward Element-by-Element ====================

func TestMLPMetal_ForwardElementwise(t *testing.T) {
	initMetal(t)

	mlp := NewMLP([]int{4, 8, 4, 1}, MLPConfig{
		Activation: "relu",
		BatchNorm:  false,
		Dropout:    0,
		Sigmoid:    true,
		BNMomentum: 0.1,
	})

	B := 4
	input := []float32{
		0.1, 0.2, 0.3, 0.4,
		0.5, 0.6, 0.7, 0.8,
		-0.1, -0.2, 0.1, 0.5,
		0.3, -0.1, 0.4, 0.2,
	}
	targets := []float32{1, 0, 1, 0}

	// CPU forward
	cpuLogits := mlp.ForwardLogits(input, B, true)
	cpuLoss, _ := mlp.BCEWithLogitsLoss(cpuLogits, targets, 0)

	// GPU: clone weights into Metal
	mlpGPU := NewMLP([]int{4, 8, 4, 1}, MLPConfig{
		Activation: "relu",
		BatchNorm:  false,
		Dropout:    0,
		Sigmoid:    true,
		BNMomentum: 0.1,
	})
	for i := range mlpGPU.Layers {
		copy(mlpGPU.Layers[i].W, mlp.Layers[i].W)
		copy(mlpGPU.Layers[i].B, mlp.Layers[i].B)
	}

	mtl := NewMLPMetal(NewMetal(), mlpGPU, B)
	if mtl == nil {
		t.Fatal("NewMLPMetal failed")
	}

	mtl.UploadBatch(input, targets)
	gpuLoss := mtl.TrainStep(0)

	t.Logf("CPU loss=%.8f  GPU loss=%.8f  diff=%.2e", cpuLoss, gpuLoss, math.Abs(float64(cpuLoss-gpuLoss)))
	assertClose(t, "forward_loss", gpuLoss, cpuLoss, 1e-4)

	// Check per-layer activations via preBN buffers
	for li := 0; li < mtl.nLayers; li++ {
		oD := mlp.Layers[li].OutDim
		gpuAct := MtlBufSharedSlice(mtl.preBN[li], B*oD)
		cpuAct := mlp.Layers[li].preAct

		maxDiff := float64(0)
		for j := 0; j < B*oD; j++ {
			d := math.Abs(float64(gpuAct[j] - cpuAct[j]))
			if d > maxDiff {
				maxDiff = d
			}
		}
		t.Logf("Layer %d preAct max diff: %.2e (shape [%d,%d])", li, maxDiff, B, oD)
		if maxDiff > 1e-4 {
			t.Errorf("Layer %d preAct diverged: max diff %.2e", li, maxDiff)
			for j := 0; j < min(8, B*oD); j++ {
				t.Logf("  [%d] CPU=%.6f GPU=%.6f", j, cpuAct[j], gpuAct[j])
			}
		}
	}
}

// ==================== Training with BN ====================

func TestMLPMetal_TrainingWithBN(t *testing.T) {
	initMetal(t)

	B := 8
	input := []float32{
		0.0, 0.0, 0.2, 0.3, 0.1, 0.4, 0.3, 0.2,
		0.8, 0.9, 0.9, 0.6, 0.7, 0.8, 1.0, 1.0,
	}
	targets := []float32{0, 0, 0, 0, 1, 1, 1, 1}

	mlp := NewMLP([]int{2, 8, 1}, MLPConfig{
		Activation: "relu",
		BatchNorm:  true,
		Dropout:    0,
		Sigmoid:    true,
		WeightDecay: 0,
		BNMomentum: 0.1,
	})

	mtl := NewMLPMetal(NewMetal(), mlp, B)
	if mtl == nil {
		t.Fatal("NewMLPMetal failed")
	}

	mtl.UploadBatch(input, targets)
	var lastLoss float32
	for step := 0; step < 500; step++ {
		lastLoss = mtl.TrainStep(0.001)
		if step%100 == 0 {
			t.Logf("BN step %d: loss=%.6f", step, lastLoss)
		}
	}

	t.Logf("BN final loss=%.6f", lastLoss)
	if lastLoss > 0.65 {
		t.Errorf("BN training didn't converge: final loss=%.6f", lastLoss)
	}
}

// ==================== Training with Dropout ====================

func TestMLPMetal_TrainingWithDropout(t *testing.T) {
	initMetal(t)

	B := 8
	input := []float32{
		0.0, 0.0, 0.2, 0.3, 0.1, 0.4, 0.3, 0.2,
		0.8, 0.9, 0.9, 0.6, 0.7, 0.8, 1.0, 1.0,
	}
	targets := []float32{0, 0, 0, 0, 1, 1, 1, 1}

	mlp := NewMLP([]int{2, 8, 1}, MLPConfig{
		Activation:  "relu",
		BatchNorm:   false,
		Dropout:     0.2,
		Sigmoid:     true,
		WeightDecay: 0,
	})

	mtl := NewMLPMetal(NewMetal(), mlp, B)
	if mtl == nil {
		t.Fatal("NewMLPMetal failed")
	}

	mtl.UploadBatch(input, targets)
	var lastLoss float32
	for step := 0; step < 1000; step++ {
		lastLoss = mtl.TrainStep(0.001)
		if step%200 == 0 {
			t.Logf("Dropout step %d: loss=%.6f", step, lastLoss)
		}
	}

	t.Logf("Dropout final loss=%.6f", lastLoss)
	if lastLoss > 0.65 {
		t.Errorf("Dropout training didn't converge: final loss=%.6f", lastLoss)
	}
}

// ==================== Real-world Dimensions ====================

func TestMLPMetal_RealDimensions(t *testing.T) {
	initMetal(t)

	nFeatures := 502
	B := 64
	mlp := NewMLP([]int{nFeatures, 512, 256, 128, 1}, MLPConfig{
		Activation:  "relu",
		BatchNorm:   true,
		Dropout:     0.2,
		Sigmoid:     true,
		WeightDecay: 0.0005,
		BNMomentum:  0.1,
	})

	mtl := NewMLPMetal(NewMetal(), mlp, B)
	if mtl == nil {
		t.Fatal("NewMLPMetal failed")
	}

	// Generate random input
	input := make([]float32, B*nFeatures)
	targets := make([]float32, B)
	for i := range input {
		input[i] = float32(i%100) / 100.0
	}
	for i := range targets {
		if i < B/2 {
			targets[i] = 0
		} else {
			targets[i] = 1
		}
	}

	mtl.UploadBatch(input, targets)

	var firstLoss, lastLoss float32
	for step := 0; step < 100; step++ {
		lastLoss = mtl.TrainStep(0.0003)
		if step == 0 {
			firstLoss = lastLoss
		}
		if step%20 == 0 {
			t.Logf("Real-dim step %d: loss=%.6f", step, lastLoss)
		}
	}

	t.Logf("Real-dim: first_loss=%.6f last_loss=%.6f", firstLoss, lastLoss)
	if math.IsNaN(float64(lastLoss)) || math.IsInf(float64(lastLoss), 0) {
		t.Errorf("loss is NaN/Inf at real dimensions")
	}
	if lastLoss >= firstLoss {
		t.Errorf("loss didn't decrease: first=%.6f last=%.6f", firstLoss, lastLoss)
	}

	// Verify weight download works at these dimensions
	mtl.DownloadWeights()
	for li, l := range mlp.Layers {
		for j := 0; j < min(4, len(l.W)); j++ {
			if math.IsNaN(float64(l.W[j])) {
				t.Errorf("Layer %d W[%d] is NaN after download", li, j)
			}
		}
	}
}

func itoa(i int) string {
	if i < 0 {
		return "-" + itoa(-i)
	}
	if i < 10 {
		return string(rune('0' + i))
	}
	return itoa(i/10) + string(rune('0'+i%10))
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
