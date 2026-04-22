# mongoose

GPU compute for Go. Trains transformers without Python. One `Engine` interface, five backends — CUDA, Metal, Accelerate, WebGPU, CPU — selected at build time. Sparse by default.

## Training — Dual H100 SXM NVLink

Helix Dispatch: interleaved position parallelism. Both GPUs fire every GEMM simultaneously. No gradient sync. Per-GPU optimizer. Only K,V crosses the wire.

```
Byte-level transformer, 8 layers, seq_len=64, 2000 steps.

dim       mongoose    dense (DDP)    ratio
128       172.5       —              —
256       184.9       —              —
512        74.4       —              —
1024       66.0       —              —
2048       40.9       32.5           1.3x
4096       21.7       14.1           1.54x
```

At dim=4096 (1.2B params), mongoose trains 54% faster than the standard dense approach using the same hardware.

## How it works

**Conductor** observes which rows are active each step. Inactive rows get no gradient, no optimizer update, no weight writeback. The sparse TN GEMM kernel skips entire 32-row threadgroup tiles when the mask says zero.

**Helix** couples parameter pairs through DNA rung geometry — gate↔up (G≡C, 3 H-bonds), wq↔wo (A≡T, 2 H-bonds), wk↔wv (A≡T, 2 H-bonds). Three pairs per layer, no orphan weights. The coupled gradient cross-pollinates signal between functionally related weights.

**Helix Dispatch** splits sequence positions across GPUs. Even positions GPU 0, odd positions GPU 1. Both GPUs hold full weight replicas (read-only FP16). Each GPU runs its own Helix optimizer on its own FP32 master weights — two independent strands coupled through the conductor's shared hot-row mask. Scales to N GPUs: `pos % nGPUs == gpu`.

**Needle** (Metal path) trains INT8 weights directly. INT8 + per-row scale + FP32 delta residual. The weight is FP32 for nanoseconds — only in register, never in memory.

## Build

```bash
CGO_ENABLED=1 go build ./...   # CUDA on Linux, Metal on macOS
CGO_ENABLED=0 go build ./...   # WebGPU/Vulkan, pure Go
```

CUDA kernels (optional, enables fused training + inference):
```bash
cd kernels
nvcc -shared -o libmongoose_kernels.so mongoose.cu \
    -Xcompiler -fPIC -gencode arch=compute_90,code=compute_90
```

## Interfaces

```go
eng := mongoose.NewCUDA()  // or NewMetal(), &CPU{}, NewWebGPU()
te := mongoose.AsTensorEngine(eng)
t := te.FromHost(data, []int{rows, cols})
```

Four levels: `Engine` (host slices), `TensorEngine` (GPU tensors), `TrainEngine` (BLAS), `GraphTrainEngine` (fused dispatch).

## Backends

| Backend | Build tag | What |
|---------|-----------|------|
| CUDA | `linux && cgo` | cuBLAS + cublasLt + custom kernels |
| Metal | `darwin && cgo` | MPS + compute shaders + Metal 4 matmul2d |
| Accelerate | `darwin && cgo` | Apple AMX via cblas |
| WebGPU | `!cgo` | gogpu/wgpu (Vulkan/Metal/DX12) |
| CPU | always | Pure Go |

## Ecosystem

| Package | What |
|---------|------|
| [ai](https://github.com/open-ai-org/ai) | CLI — train, infer, chat, serve, quantize |
| [helix](https://github.com/open-ai-org/helix) | DNA optimizer |
| [needle](https://github.com/open-ai-org/needle) | Fused INT8 optimizer kernels |
| [gguf](https://github.com/open-ai-org/gguf) | GGUF + SafeTensors I/O |
| [tokenizer](https://github.com/open-ai-org/tokenizer) | BPE tokenizer |

## License

MIT
