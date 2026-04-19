//go:build !darwin || !cgo

package mongoose

// Accelerate stub for non-macOS platforms.
type Accelerate struct{}

func (a *Accelerate) Name() string { return "accelerate/unavailable" }
func (a *Accelerate) MatMul(aData, bData []float32, m, k, n int) []float32 { return nil }
func (a *Accelerate) RMSNorm(x, weight []float32, eps float32) {}
func (a *Accelerate) SoftMax(x []float32, n int) {}
func (a *Accelerate) ReLU(x []float32) {}
func (a *Accelerate) VRAM() uint64 { return 0 }
func (a *Accelerate) Benchmark() float64 { return 0 }

// BLAS extension stubs
func (a *Accelerate) MatMulTransB(aData, bData []float32, m, k, n int) []float32 { return nil }
func (a *Accelerate) MatMulTransBInto(out, aData, bData []float32, m, k, n int) {}
func (a *Accelerate) MatMulTransA(aData, bData []float32, m, k, n int) []float32 { return nil }
func (a *Accelerate) MatMulTransAInto(out, aData, bData []float32, m, k, n int) {}
func (a *Accelerate) MatMulInto(out, aData, bData []float32, m, k, n int) {}
func (a *Accelerate) MatMulAddInto(out, aData, bData []float32, m, k, n int) {}
func (a *Accelerate) GER(G []float32, x []float32, y []float32, m, n int, alpha float32) {}
func (a *Accelerate) Nrm2(x []float32) float32 { return 0 }
func (a *Accelerate) Scal(x []float32, alpha float32) {}
func (a *Accelerate) Dot(x, y []float32) float32 { return 0 }
func (a *Accelerate) Axpy(x, y []float32, alpha float32) {}
func (a *Accelerate) AdamWStep(D, G, M, V []float32, n int, lr, beta1, beta2, bc1, bc2, eps, wd float32) {}
