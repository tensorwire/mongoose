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
128       624K       773
512       9.6M       455
1024      38M        191
2048      152M        74
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

### Convergence — RTX 5090, dim=512, 4 layers

```
step       loss     floor    notes
1          6.166    —        
50         3.876    2.573    immune checkpoint at step 39
100        2.588    2.374
200        2.184    1.892    
300        2.047    1.755
400        1.940    1.285    immune rewind at step 356
500        1.951    1.285    364.8 steps/s steady state
```

Loss drops 4.8x in 500 steps. The Helix immune system checkpoints at loss floors and reverts on rebound — visible in the floor column tracking the best-seen loss.

## Inference — Qwen2.5-0.5B (200 tokens)

```
RTX 5090     182.4 tok/s    Q8 fused matvec
M4 Max        54.3 tok/s    GPU-resident
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
| [ai](https://github.com/open-ai-org/ai) | CLI — train, infer, chat, serve, quantize |
| [helix](https://github.com/open-ai-org/helix) | DNA optimizer |
| [needle](https://github.com/open-ai-org/needle) | Fused INT8 optimizer kernels |
| [gguf](https://github.com/open-ai-org/gguf) | GGUF + SafeTensors I/O |
| [tokenizer](https://github.com/open-ai-org/tokenizer) | BPE tokenizer |

## License

MIT
