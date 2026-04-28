# CLAUDE.md — Mongoose

## What This Is

Cross-platform GPU compute library for Go. One `Engine` interface, five backends: NVIDIA (CUDA/cuBLAS), Apple (Metal/MPS/Accelerate), WebGPU (Vulkan/DX12), Intel (Xe/Level Zero), and CPU (pure Go). Fused graph dispatch for training, fused compute shaders for inference. Faster than PyTorch on verified benchmarks.

## Build

```bash
CGO_ENABLED=1 go build ./...   # CUDA on Linux, Metal+Accelerate on macOS
CGO_ENABLED=0 go build ./...   # WebGPU/Vulkan, pure Go — no CGo required
```

CUDA kernels (fused Q8/Q4 matvec, RMSNorm, RoPE, attention, SiLU):
```bash
cd kernels
nvcc -shared -o libmongoose_kernels.so mongoose.cu -Xcompiler -fPIC -gencode arch=compute_90,code=compute_90
```

Metal 4 kernels (matmul2d TensorOp — pre-compiled .metallib):
```bash
cd kernels
xcrun metal -std=metal4.0 -O2 -c gemm_metal4.metal -o gemm_metal4.air
xcrun metallib -o gemm_metal4.metallib gemm_metal4.air
```

## Architecture

### Interfaces

- `Engine` — MatMul, RMSNorm, SoftMax, ReLU, VRAM, Benchmark
- `TensorEngine` — GPU-resident tensor ops: FromHost, Zeros, ToHost, Release, MatMulT, MatMulTransposeBT, AddInPlace
- `TrainEngine` — BLAS primitives: MatMulTransBInto, MatMulInto, Nrm2, Scal, GER, AdamWStep
- `GraphTrainEngine` — Fused graph dispatch: BuildFullGraph + GraphTrainStepAdam

### Backends

| Backend | File | Build Tag | What |
|---------|------|-----------|------|
| CPU | `cpu.go` | always | Pure Go fallback |
| Metal | `metal.go` | `darwin && cgo` | Apple MPS + MPSGraph + custom compute shaders |
| Accelerate | `accelerate.go` | `darwin && cgo` | Apple AMX via Accelerate.framework |
| CUDA | `cuda.go` | `linux && cgo` | NVIDIA cuBLAS + custom kernels |
| WebGPU | `webgpu.go` | `!cgo` | gogpu/wgpu (pure Go) |
| Xe | `xe.go` | `linux && cgo` | Intel Level Zero |

Every backend has a `*_stub.go` companion that compiles on all platforms.

### Custom Kernels

**CUDA** (`kernels/mongoose.cu`): RMSNorm, RoPE, GQA decode attention, SiLU gate mul, fused Q8 matvec, fused Q4 matvec, KV cache write, AdamW, embedding gather, cross-entropy loss, helix DNA optimizer, INT8 dequant.

**Metal** (`metal_impl_darwin.m` kernel source): RMSNorm (in-place + out-of-place), RoPE rotate_half, GQA attention, SiLU gate mul, KV cache write, FP32/FP16 conversion, INT8 dequant, fused Q8 matvec, fused Q4 matvec, AdamW, add/scale/relu.

**Metal 4** (`kernels/gemm_metal4.metal`): matmul2d TensorOp kernels using cooperative tensors. FP16 and FP32 variants for BT, NN, TN transposes. Pre-compiled to `.metallib`.

### Metal Fused Inference

`metal_impl_darwin.m` provides a fused inference pipeline (`mtl_fused_build` / `mtl_fused_step`):
- All layer weights stored as persistent MTLBuffers (FP16, Q8, or Q4)
- One command buffer per token: all layers encoded in sequence
- Custom compute kernels for element-wise ops
- Metal 4 matmul2d or fused Q8/Q4 matvec for weight matmuls
- MPS MatrixMultiplication as fallback

### Key Design Decisions

- **Fused graph dispatch** for training — `BuildFullGraph()` compiles the entire model at init, `GraphTrainStepAdam()` fires forward + backward + optimizer as one dispatch
- **Fused compute shaders** for inference — one command buffer (Metal) or one stream (CUDA) per token, zero CPU round-trips
- **Buffer pool** reuses GPU allocations (Metal: map[int][]MTLBufferRef, capped at 8 per size)
- **Runtime kernel loading** — CUDA kernels loaded via dlopen from `libmongoose_kernels.so`
- **Metal 4 detection** — loads pre-compiled `.metallib` at runtime, falls back to MPS if unavailable

## Test

```bash
go test -v ./...              # all platform-neutral tests
go test -v -run CPU ./...     # CPU backend tests
```

## Related Packages

- `github.com/tensorwire/gguf` — GGUF + SafeTensors + NumPy I/O
- `github.com/tensorwire/tokenizer` — BPE tokenizer (GPT-2, SentencePiece)
- `github.com/tensorwire/helix` — DNA optimizer (forward-only training)
- `github.com/tensorwire/needle` — Fused INT8 dequant + Adam kernels
