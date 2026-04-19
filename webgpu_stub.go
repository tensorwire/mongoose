//go:build cgo

package mongoose

// WebGPU stub for CGo builds (WebGPU uses goffi which requires CGO_ENABLED=0).
// On CGo platforms, we use Accelerate (macOS) or CUDA (Linux) instead.
type WebGPU struct{}

func NewWebGPU() *WebGPU          { return nil }
func NewWebGPULowPower() *WebGPU       { return nil }
func (g *WebGPU) AdapterName() string  { return "" }
func (g *WebGPU) Release()  {}
func (g *WebGPU) Name() string { return "webgpu/unavailable" }
func (g *WebGPU) MatMul(a, b []float32, m, k, n int) []float32 { return nil }
func (g *WebGPU) RMSNorm(x, weight []float32, eps float32) {}
func (g *WebGPU) SoftMax(x []float32, n int) {}
func (g *WebGPU) ReLU(x []float32) {}
func (g *WebGPU) VRAM() uint64 { return 0 }
func (g *WebGPU) Benchmark() float64 { return 0 }
