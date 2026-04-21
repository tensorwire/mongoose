package mongoose

import "unsafe"

// Tensor represents a multi-dimensional array that lives on a compute device.
// Data stays on-device (GPU) until explicitly moved to host.
// This is the key abstraction that eliminates PCIe round-trips between ops.
//
// Usage:
//   a := eng.FromHost(data, []int{m, k})  // upload once
//   b := eng.FromHost(weights, []int{k, n})
//   c := eng.MatMulT(a, b)                // stays on GPU
//   d := eng.ReLUT(c)                     // stays on GPU
//   result := d.ToHost()                  // download once
type Tensor struct {
	Shape  []int       // dimensions
	Size   int         // total elements
	host   []float32   // CPU data (may be nil if GPU-only)
	device DevicePtr   // GPU pointer (nil if CPU-only)
	eng    TensorEngine // owning engine (for ToHost/Release)
}

// DevicePtr is an opaque GPU memory pointer.
type DevicePtr interface{}

// TensorEngine extends Engine with GPU-resident tensor operations.
// Backends that support GPU tensors implement this interface.
// Backends that don't (CPU, Accelerate) fall back to host-side ops.
type TensorEngine interface {
	Engine

	// Tensor lifecycle
	FromHost(data []float32, shape []int) *Tensor   // upload to GPU
	Zeros(shape []int) *Tensor                       // allocate zeros on GPU
	ToHost(t *Tensor) []float32                      // download from GPU
	Release(t *Tensor)                               // free GPU memory

	// GPU-resident operations (no host round-trip)
	MatMulT(a, b *Tensor, m, k, n int) *Tensor                // C = A @ B
	MatMulTransposeAT(a, b *Tensor, m, k, n int) *Tensor      // C = A^T @ B (gradient: dW = act^T @ dOut)
	MatMulTransposeBT(a, b *Tensor, m, k, n int) *Tensor      // C = A @ B^T (gradient: dAct = dOut @ W^T)
	AddT(a, b *Tensor) *Tensor                                 // element-wise add
	AddInPlace(a, b *Tensor)                                   // a += b (no alloc)
	ScaleT(a *Tensor, s float32) *Tensor                       // element-wise scale
	ReLUT(a *Tensor) *Tensor                                   // ReLU on GPU
	ReLUBackwardT(dOut, fwdInput *Tensor) *Tensor              // dOut * (fwdInput > 0)
	TransposeT(a *Tensor, rows, cols int) *Tensor              // matrix transpose
	CopyT(src *Tensor) *Tensor                                 // deep copy on GPU
}

// NewTensor creates a CPU-only tensor.
func NewTensor(data []float32, shape []int) *Tensor {
	size := 1
	for _, s := range shape {
		size *= s
	}
	return &Tensor{
		Shape: shape,
		Size:  size,
		host:  data,
	}
}

// RawDevicePtr is implemented by device pointer types to expose the raw GPU pointer.
type RawDevicePtr interface {
	RawPtr() unsafe.Pointer
}

type rawPtr struct{ p unsafe.Pointer }

func (r *rawPtr) RawPtr() unsafe.Pointer { return r.p }

// TensorFromDevicePtr wraps a raw GPU pointer as a Tensor.
// Used for multi-GPU allocations where the pointer lives on a remote device.
func TensorFromDevicePtr(ptr unsafe.Pointer, nElements int) *Tensor {
	return &Tensor{Shape: []int{nElements}, Size: nElements, device: &rawPtr{ptr}}
}

// DevicePtr returns the raw GPU pointer for passing to CUDA kernels.
// Returns nil if the tensor is CPU-only.
func (t *Tensor) DevicePtr() unsafe.Pointer {
	if t.device == nil {
		return nil
	}
	if raw, ok := t.device.(RawDevicePtr); ok {
		return raw.RawPtr()
	}
	return nil
}

// ToHost returns the CPU data, downloading from GPU if needed.
func (t *Tensor) ToHost() []float32 {
	if t.host != nil {
		return t.host
	}
	if t.eng != nil {
		t.host = t.eng.ToHost(t)
	}
	return t.host
}

// IsTensorEngine returns true if the engine supports GPU-resident tensors.
func IsTensorEngine(eng Engine) bool {
	_, ok := eng.(TensorEngine)
	return ok
}

// AsTensorEngine casts an Engine to TensorEngine, or nil.
func AsTensorEngine(eng Engine) TensorEngine {
	te, _ := eng.(TensorEngine)
	return te
}

// ResidentWeightEngine provides fast matmul with pre-uploaded weight tensors.
// The weight stays on GPU; only the small activation vector round-trips per call.
type ResidentWeightEngine interface {
	MatVecResidentW(W *Tensor, x []float32, rows, cols int) []float32
}

// AsResidentWeightEngine casts an Engine to ResidentWeightEngine, or nil.
func AsResidentWeightEngine(eng Engine) ResidentWeightEngine {
	rw, _ := eng.(ResidentWeightEngine)
	return rw
}
