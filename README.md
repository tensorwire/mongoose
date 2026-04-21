# mongoose

GPU compute library for Go. Trains and infers transformers without Python. Five backends behind one `Engine` interface — CUDA, Metal, Accelerate, WebGPU, CPU — selected by build tags. The public API operates on `[]float32` slices and `*Tensor` pointers. No framework, no graph compiler, no JIT.

## Training throughput

Byte-level transformer, 4 layers, seq_len=64, vocab=256. 100 steps each. Measured April 2026.

```
                Metal (M4 Max 40GB)             CUDA (RTX 5090 32GB)
dim=128         134 steps/s                     700 steps/s
dim=256         113                             —
dim=512         104                             —
dim=1024         49                             —
dim=2048         32                             —
dim=4096         11 (605M params, 9.5s)         —
```

Metal path uses Needle INT8 optimizer with conductor-driven sparsity. CUDA path uses FP32 Helix DNA optimizer with AdamW for unpaired weights.

### vs PyTorch MPS (M4 Max, same model, same data)

```
dim       mongoose    PyTorch 2.8    ratio
128       134         62             2.2x
256       113         68             1.7x
512       104         64             1.6x
1024       49         60             0.8x
2048       32         24             1.3x
4096       11          6             1.7x
```

We lose at dim=1024 — PyTorch's MPSGraph fuses operations that our per-op dispatch path doesn't. Fused layer kernels exist for dim<512 but the tiled GEMM approach at larger dims needs the same treatment.

## What the training pipeline does

The Metal path runs the entire step — forward, GPU loss, backward, gradient clipping, optimizer, weight writeback — in a single command buffer with zero CPU synchronization.

**Needle** trains INT8 weights directly. Each weight is stored as INT8 + per-row FP32 scale + FP32 delta residual. The optimizer kernel dequants to FP32 in register, applies the Adam update with gradient clipping, requantizes, and writes the FP32 live weight for the next forward pass. The weight is FP32 for nanoseconds. FP16 momentum and velocity.

**Conductor** observes which embedding rows and projection output rows are active each step. Inactive rows get no gradient, no optimizer update, no dequant. At byte-level vocab=256 with seq=64, roughly 27 of 256 embed rows are hot. The sparse TN GEMM kernel skips entire 32-row threadgroup tiles in the backward pass when the mask is zero.

**Helix** couples parameter pairs through DNA rung geometry — gate↔up with G≡C bonding (3 hydrogen bonds), Q↔K with A≡T bonding (2 hydrogen bonds). The coupled gradient cross-pollinates signal between paired weights. Fibonacci stride adapts exploration rate to training signal conductivity. The immune system checkpoints at loss floors and reverts on rebound.

### CUDA status

The CUDA path uses FP32 weights with `KHelixDNAStep` for paired parameters and `KAdamW` for singles. The Needle INT8 CUDA kernels exist (`mongoose_helix_needle`, `mongoose_helix_needle_paired` in `mongoose.cu`) and the Go bindings are wired, but the full pipeline requires the Xe iGPU for INT8↔FP16 conversion through L3 cache to avoid bottlenecking the discrete GPU. Shipping CUDA Needle with an Intel iGPU dependency is too opinionated for v1 — it works on Arrow Lake systems where the Xe shares L3 with the CPU, but excludes AMD platforms and older Intel. The FP32 path runs clean at 700 steps/s on RTX 5090.

Checkpoints are cross-compatible: safetensors format with Llama-compatible config.json. Train on Metal with Needle, resume on CUDA with FP32, or vice versa.

## Build

```bash
# macOS — Metal + Accelerate
CGO_ENABLED=1 go build ./...

# Linux — CUDA
CGO_ENABLED=1 go build ./...

# Any platform — WebGPU/Vulkan, pure Go
CGO_ENABLED=0 go build ./...
```

Optional kernel compilation:

```bash
# CUDA custom kernels
cd kernels
nvcc -shared -o libmongoose_kernels.so mongoose.cu \
    -Xcompiler -fPIC -gencode arch=compute_90,code=compute_90

# Metal 4 cooperative tensor GEMM
cd kernels
xcrun metal -std=metal4.0 -O2 -c gemm_metal4.metal -o gemm_metal4.air
xcrun metallib -o gemm_metal4.metallib gemm_metal4.air

# Fused training kernels (pre/post attention)
xcrun metal -std=metal3.0 -O2 -c fused_train.metal -o fused_train.air
xcrun metallib -o fused_train.metallib fused_train.air
```

## Architecture

### Interfaces

```go
eng := mongoose.NewMetal()  // or NewCUDA(), &CPU{}, NewWebGPU()
result := eng.MatMul(weights, input, rows, cols, 1)

te := mongoose.AsTensorEngine(eng)
t := te.FromHost(data, []int{rows, cols})  // GPU-resident
```

Four interface levels: `Engine` (host slices), `TensorEngine` (GPU-resident tensors), `TrainEngine` (BLAS primitives for CPU-side training), `GraphTrainEngine` (fused single-dispatch training).

### Backends

| Backend | Build tag | Dispatch |
|---------|-----------|----------|
| CUDA | `linux && cgo` | cuBLAS + custom kernels via dlopen |
| Metal | `darwin && cgo` | MPS + Metal compute shaders + Metal 4 matmul2d |
| Accelerate | `darwin && cgo` | Apple AMX via cblas |
| WebGPU | `!cgo` | gogpu/wgpu WGSL shaders (Vulkan/Metal/DX12) |
| CPU | always | Pure Go, auto-vectorizes on NEON/AVX |

Every backend has a `*_stub.go` companion. `go build ./...` compiles on any platform.

### Metal training dispatch

At dim < 512: fused kernels (`fused_pre_attn`, `fused_post_attn`) combine RMSNorm + GEMM + RoPE + SiLU into one dispatch per layer phase. At dim >= 512: tiled 32×32 GEMMs with shared memory tiling. Auto-selected based on model shape.

Forward dispatches encode into one Metal compute command buffer. The batch-encode path (Apple7/M1) calls all dispatch functions from a single C function — one CGo round-trip per step. The ICB path (Apple8+/M2+) pre-encodes into an indirect command buffer at init — the infrastructure is built but blocked on a command encoding issue at high buffer bind counts.

### CUDA arena allocator

One `cudaMalloc` at init for 80% of VRAM. Sub-allocation by best-fit with 256-byte alignment and block merging. Zero `cudaMalloc`/`cudaFree` during training.

### Warm cache

On Metal: one MTLBuffer in unified memory holds all optimizer state. CPU and GPU access the same physical pages. On CUDA: pinned host memory (L3 bridge) for rung coefficients and masks.

## Custom kernels

**CUDA** (`kernels/mongoose.cu`): RMSNorm, RoPE, GQA attention, SiLU, fused Q8/Q4 matvec, AdamW, helix DNA step, helix needle (single + paired), cross-entropy, INT8 dequant (+delta variant), embedding gather, KV cache write.

**Metal** (`metal_impl_darwin.m`): Same kernel set compiled from inline MSL source at init. All `constant&` parameters migrated to `device const*` for indirect command buffer compatibility. Pipeline states created with `supportIndirectCommandBuffers = YES`.

**Metal 4** (`kernels/gemm_metal4.metal`): Cooperative tensor matmul2d in FP16 and FP32 (BT, NN, TN). Pre-compiled to `.metallib`, loaded at runtime, falls back to tiled GEMM if unavailable.

**Fused training** (`kernels/fused_train.metal`): `fused_pre_attn` (RMSNorm → Q/K/V GEMM → RoPE) and `fused_post_attn` (WO GEMM → residual → RMSNorm → gate/up GEMM → SiLU → down GEMM → residual). One threadgroup per position, dim threads. Pre-compiled to `.metallib`.

## Ecosystem

| Package | What |
|---------|------|
| [ai](https://github.com/open-ai-org/ai) | CLI — train, infer, chat, serve, quantize, resume |
| [helix](https://github.com/open-ai-org/helix) | DNA optimizer — rung coupling, immune system, Fibonacci stride |
| [needle](https://github.com/open-ai-org/needle) | Fused INT8 dequant + Adam kernels (CUDA + Metal) |
| [gguf](https://github.com/open-ai-org/gguf) | GGUF + SafeTensors + NumPy I/O |
| [tokenizer](https://github.com/open-ai-org/tokenizer) | BPE tokenizer (GPT-2, SentencePiece) |

## License

MIT
