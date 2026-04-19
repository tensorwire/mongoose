package mongoose

import (
	"math"
	"testing"
)

var _ Engine = (*CPU)(nil)

func TestCPUName(t *testing.T) {
	c := &CPU{}
	if c.Name() != "cpu" {
		t.Errorf("Name() = %q, want cpu", c.Name())
	}
}

func TestCPUMatMul_2x3_3x2(t *testing.T) {
	c := &CPU{}
	a := []float32{1, 2, 3, 4, 5, 6}
	b := []float32{7, 8, 9, 10, 11, 12}
	out := c.MatMul(a, b, 2, 3, 2)

	want := []float32{58, 64, 139, 154}
	for i, w := range want {
		if math.Abs(float64(out[i]-w)) > 0.001 {
			t.Errorf("out[%d] = %f, want %f", i, out[i], w)
		}
	}
}

func TestCPUMatMul_Identity(t *testing.T) {
	c := &CPU{}
	eye := []float32{1, 0, 0, 0, 1, 0, 0, 0, 1}
	x := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}
	out := c.MatMul(x, eye, 3, 3, 3)

	for i, want := range x {
		if math.Abs(float64(out[i]-want)) > 0.001 {
			t.Errorf("identity: out[%d] = %f, want %f", i, out[i], want)
		}
	}
}

func TestCPUReLU(t *testing.T) {
	c := &CPU{}
	x := []float32{-3, -1, 0, 1, 3}
	c.ReLU(x)

	want := []float32{0, 0, 0, 1, 3}
	for i, w := range want {
		if x[i] != w {
			t.Errorf("ReLU[%d] = %f, want %f", i, x[i], w)
		}
	}
}

func TestCPUSoftMax(t *testing.T) {
	c := &CPU{}
	x := []float32{1, 2, 3}
	c.SoftMax(x, 3)

	var sum float32
	for _, v := range x {
		sum += v
		if v < 0 || v > 1 {
			t.Errorf("softmax value %f out of [0,1]", v)
		}
	}
	if math.Abs(float64(sum-1.0)) > 0.001 {
		t.Errorf("softmax sum = %f, want 1.0", sum)
	}

	if x[2] <= x[1] || x[1] <= x[0] {
		t.Errorf("softmax order wrong: %v", x)
	}
}

func TestCPURMSNorm(t *testing.T) {
	c := &CPU{}
	x := []float32{1, 2, 3, 4}
	weight := []float32{1, 1, 1, 1}
	c.RMSNorm(x, weight, 1e-5)

	var ssq float32
	for _, v := range x {
		ssq += v * v
	}
	rms := float32(math.Sqrt(float64(ssq / float32(len(x)))))
	if math.Abs(float64(rms-1.0)) > 0.1 {
		t.Errorf("RMSNorm output RMS = %f, want ~1.0", rms)
	}
}

func TestCPUBenchmark(t *testing.T) {
	c := &CPU{}
	gflops := c.Benchmark()
	if gflops <= 0 {
		t.Errorf("Benchmark() = %f, want > 0", gflops)
	}
	t.Logf("CPU benchmark: %.2f GFLOPS", gflops)
}

func TestCPUVRAM(t *testing.T) {
	c := &CPU{}
	if c.VRAM() != 0 {
		t.Errorf("CPU VRAM = %d, want 0", c.VRAM())
	}
}
