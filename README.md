# mongoose

GPU compute for Go. Trains transformers without Python. One `Engine` interface, five backends — CUDA, Metal, Accelerate, WebGPU, CPU — selected at build time. Sparse by default.

## Training throughput

### Dual H100 SXM NVLink — Helix Dispatch

8 layers, seq_len=64, vocab=256. 2000 steps. Both GPUs fire every GEMM simultaneously via interleaved position parallelism. No gradient sync.

```
dim       params     mongoose    dense (DDP)    ratio
2048     302M         40.9       32.5           1.3x
4096    1209M         21.7       14.1           1.54x
```

### RTX 5090 — single GPU

4 layers, seq_len=64, vocab=256. 1000 steps. Pure Helix optimizer, FP16 tensor cores at dim>=512.

```
dim       params     steps/s
128       624K       569
512       4.9M       433
1024      16M        294
2048      57M         83
```

### Apple M4 Max — Metal + Needle INT8

4 layers, seq_len=64, vocab=256. 100 steps. Needle INT8 optimizer with conductor-driven sparsity. Single command buffer per step.

```
dim       params     steps/s
128       624K       134
256       2.4M       113
512       9.6M       104
1024      38M         49
2048      152M        32
4096      605M        11
```


## Fine-tuning — Qwen2.5-0.5B (RTX 5090)

2000 steps, seq=256, lr=1e-5. INT8 base weights, full backward pass.

| Method | Floor | Steps/s | Time | Optimizer VRAM | 14B estimate |
|--------|-------|---------|------|----------------|--------------|
| **Helix LoRA** (default) | **0.69** | **13.1** | **153s** | **85 MB** | **~2.4 GB** |
| AdamW full (--adamw) | 0.82 | 12.8 | 156s | 2.9 GB | ~72 GB |
| PyTorch QLoRA (baseline) | 0.64 | 10.0 | ~200s | ~500 MB | ~12 GB |

Helix LoRA: frozen INT8 base + rank-8 FP32 adapters, DNA-paired optimizer (gate+up G≡C, wk+wv A=T). AdamW full: dequant→KAdamW→requant on all weights every step. PyTorch: bitsandbytes INT8 + LoRA rank-8 + AdamW.

## Inference — Qwen2.5-0.5B (200 tokens, warm serve)

```
RTX 5090     234.0 tok/s    CUDA Q8 fused matvec
M1 Pro       121.7 tok/s    Metal fused Q8 inference
```

### TinyLlama-1.1B (50 tokens, warm serve)

```
M1 Pro        66.8 tok/s    Metal fused Q8 inference
```

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
| [ai](https://github.com/tensorwire/ai) | CLI — train, infer, chat, serve, quantize |
| [helix](https://github.com/tensorwire/helix) | DNA optimizer |
| [needle](https://github.com/tensorwire/needle) | Fused INT8 optimizer kernels |
| [gguf](https://github.com/tensorwire/gguf) | GGUF + SafeTensors I/O |
| [tokenizer](https://github.com/tensorwire/tokenizer) | BPE tokenizer |

## License

MIT
