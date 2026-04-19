# CLAUDE.md — Mongoose

## What This Is

Cross-platform GPU compute library for Go. One `Engine` interface, five backends: NVIDIA (CUDA/cuBLAS), Apple (Metal/MPS/Accelerate), WebGPU (Vulkan/DX12), Intel (Xe/Level Zero), and CPU (pure Go). Fused graph dispatch for training — one launch forward, one launch backward. 2-280% faster than PyTorch on verified benchmarks.

## Build

```bash
CGO_ENABLED=1 go build ./...   # CUDA on Linux, Metal+Accelerate on macOS
CGO_ENABLED=0 go build ./...   # WebGPU/Vulkan, pure Go — no CGo required
```

## Architecture

### Public API — no `unsafe`, no `*Tensor`, no `FromHost`/`ToHost`

All operations use `[]float32` slices. Data stays on the GPU internally; the Engine handles transfers.

- `Engine` — MatMul, RMSNorm, SoftMax, ReLU, VRAM, Benchmark
- `TrainEngine` — BLAS primitives: MatMulTransBInto, MatMulInto, MatMulAddInto, Nrm2, Scal, GER, AdamWStep
- `GraphTrainEngine` — Fused graph dispatch: BuildFullGraph + GraphTrainStepAdam (one dispatch = forward + backward + optimizer)

### Backends

| Backend | File | Build Tag | What |
|---------|------|-----------|------|
| CPU | `cpu.go` | always | Pure Go fallback |
| Metal | `metal.go` | `darwin && cgo` | Apple MPS + MPSGraph |
| Accelerate | `accelerate.go` | `darwin && cgo` | Apple AMX via Accelerate.framework |
| CUDA | `cuda.go` | `linux && cgo` | NVIDIA cuBLAS + custom kernels |
| WebGPU | `webgpu.go` | `!cgo` | gogpu/wgpu (pure Go) |
| Xe | `xe.go` | `linux && cgo` | Intel Level Zero |

Every backend has a `*_stub.go` companion that compiles on all platforms. The build never fails due to missing hardware.

### Key Design Decisions

- **No `unsafe.Pointer` in Go API** — all operations take `[]float32` slices
- **Fused graph dispatch** is the breakthrough — `BuildFullGraph()` compiles the entire model at init, `GraphTrainStepAdam()` fires it as a single GPU dispatch
- **Buffer pool** reuses GPU allocations (Metal: map[int][]MTLBufferRef, capped at 8 per size)
- **Metal uses MPSGraph** for fused training, MPS for inference matmul
- **CUDA uses cuBLAS** with `CUBLAS_COMPUTE_32F_FAST_TF32` and `CUBLAS_TF32_TENSOR_OP_MATH`
- **WebGPU requires `CGO_ENABLED=0`** (gogpu's goffi conflicts with CGo)

## Test

```bash
go test -v ./...              # all platform-neutral tests
go test -v -run CPU ./...     # CPU backend tests
```

## Related Packages

- `github.com/open-ai-org/gguf` — GGUF + SafeTensors + NumPy I/O
- `github.com/open-ai-org/tokenizer` — BPE tokenizer (GPT-2, SentencePiece)
- `github.com/open-ai-org/helix` — DNA optimizer (forward-only training)
- `github.com/open-ai-org/needle` — Fused INT8 dequant + Adam kernels
