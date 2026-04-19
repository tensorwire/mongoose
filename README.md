# mongoose

Cross-platform GPU compute library for Go. One `Engine` interface, five backends.

```go
eng := mongoose.NewMetal()  // or NewCUDA(), &CPU{}, NewWebGPU()
result := eng.MatMul(weights, input, rows, cols, 1)
eng.RMSNorm(hidden, normWeight, 1e-6)
```

Everything operates on `[]float32`. No `unsafe.Pointer` in the public API.

## Benchmarks

All numbers verified on 2026-04-19. Same model (624K Llama), same data (TinyStories 48MB),
same optimizer (AdamW), same config (dim=128, heads=4, kv=2, layers=4, ffn=256, seq=64).

### Training — tesseract vs PyTorch

| GPU | PyTorch | tesseract | Speedup | Loss |
|-----|--------:|----------:|--------:|------|
| Apple M4 Max (40GB) | 110.7 steps/s | 167.4 steps/s | **1.51x** | 5.6 → 1.67 |
| NVIDIA RTX 5090 (32GB) | 165.8 steps/s | 201.6 steps/s | **1.22x** | 5.6 → 2.87 |
| Apple M1 Pro (16GB) | — | 147.6 steps/s | — | 5.5 → 2.04 |
| CPU (pure Go) | — | 20.5 steps/s | — | 5.6 → 4.58 |

<!-- H100 SXM numbers pending — RunPod verification in progress -->

### MatMul Throughput (Engine.MatMul, 2048x2048)

| GPU | GFLOPS |
|-----|-------:|
| Apple M4 Max | 3,783 |
| NVIDIA RTX 5090 | 3,079 |
| Apple M1 Pro | 2,538 |
| CPU (pure Go) | 1.6 |

These are host-copy path numbers (upload, compute, download per call).
GPU-resident and fused graph dispatch paths are significantly faster.

## Backends

| Backend | Platform | Build | What |
|---------|----------|-------|------|
| Metal MPS | macOS | `CGO_ENABLED=1` | Apple GPU via MPSGraph — fused training dispatch |
| CUDA cuBLAS | Linux | `CGO_ENABLED=1` | NVIDIA GPU — TF32 tensor cores |
| Accelerate | macOS | `CGO_ENABLED=1` | Apple AMX coprocessor |
| WebGPU | Any | `CGO_ENABLED=0` | Vulkan/DX12 via gogpu/wgpu — pure Go, no CGo |
| CPU | Any | Always | Pure Go fallback |

Every backend has a stub companion. The build never fails due to missing hardware.

## Install

```bash
go get github.com/open-ai-org/mongoose
```

## Build

```bash
CGO_ENABLED=1 go build ./...   # Metal+Accelerate on macOS, CUDA on Linux
CGO_ENABLED=0 go build ./...   # WebGPU/Vulkan, pure Go — runs anywhere
```

## Architecture

### Engine — inference and compute

```go
type Engine interface {
    Name() string
    MatMul(a, b []float32, m, k, n int) []float32
    RMSNorm(x, weight []float32, eps float32)
    SoftMax(x []float32, n int)
    ReLU(x []float32)
    VRAM() uint64
    Benchmark() float64
}
```

### TrainEngine — BLAS primitives

```go
type TrainEngine interface {
    Engine
    MatMulTransBInto(C, A, B []float32, m, k, n int)
    MatMulInto(C, A, B []float32, m, k, n int)
    Nrm2(x []float32) float32
    AdamWStep(D, G, M, V []float32, n int, lr, beta1, beta2, bc1, bc2, eps, wd float32)
    // ... more BLAS ops
}
```

### GraphTrainEngine — fused GPU dispatch

```go
type GraphTrainEngine interface {
    BuildFullGraph(dim, kvDim, headDim, nHeads, nKVHeads, ffnDim,
        vocabSize, nLayers, seqLen int, ropeTheta float64, mode int) int
    GraphTrainStepAdam(tokens, targets []int32, lr float32) float32
    GraphNumWeights() int
    GraphSetVariable(varIdx int, data []float32) int
    Sync()
}
```

One dispatch = forward + backward + optimizer. This is the path that beats PyTorch.
MPSGraph on Metal. CUDA graph support planned.

### L3 Bridge — zero-copy GPU access

```go
bridge := cuda.AllocL3Bridge(4 * 1024 * 1024)  // 4MB pinned
data := bridge.Float32(0, 1024)                  // CPU slice
ptr := bridge.DevicePtr(0)                       // GPU reads same memory
```

Pinned host memory via `cudaHostAlloc`. CPU writes, GPU reads through L3 cache.
No memcpy. No DMA transfer. Clean ownership: CPU between dispatches, GPU during.

## Ecosystem

| Package | What |
|---------|------|
| [mongoose](https://github.com/open-ai-org/mongoose) | GPU compute engine |
| [tesseract](https://github.com/open-ai-org/tesseract) | CLI — train, infer, quantize, serve |
| [gguf](https://github.com/open-ai-org/gguf) | GGUF + SafeTensors + NumPy I/O |
| [tokenizer](https://github.com/open-ai-org/tokenizer) | BPE tokenizer (GPT-2, SentencePiece) |
| [helix](https://github.com/open-ai-org/helix) | DNA optimizer — forward-only training |
| [needle](https://github.com/open-ai-org/needle) | Fused INT8 dequant + Adam CUDA kernels |

## License

MIT
