# mongoose

Cross-platform GPU compute library for Go. Five backends behind one interface, with optional Intel Xe support. Trains and infers LLMs without Python, without PyTorch, without a framework. The entire pipeline — matrix multiplication, layer normalization, attention, optimizer step — runs through `[]float32` slices and a single `Engine` interface. No `unsafe.Pointer` in the public API.

Mongoose is not a wrapper around PyTorch. It dispatches directly to vendor compute APIs: cuBLAS on NVIDIA, Metal Performance Shaders and Metal 4 cooperative tensor operations on Apple Silicon, Level Zero on Intel Xe, WebGPU/Vulkan via gogpu for everything else, and a pure Go CPU fallback that auto-vectorizes on ARM NEON and x86 AVX. Every backend has a build-tag-guarded stub so the library compiles on any platform regardless of which GPUs are present.

```go
eng := mongoose.NewMetal()  // or NewCUDA(), &CPU{}, NewWebGPU(), NewXe()
result := eng.MatMul(weights, input, rows, cols, 1)
eng.RMSNorm(hidden, normWeight, 1e-6)
```

## Benchmarks

All numbers verified on 2026-04-20. Same data (TinyStories 48MB), same optimizer, seq=64.

![Training Throughput](docs/bench-training-steps.svg)

![ai vs PyTorch](docs/bench-vs-pytorch.svg)

![Inference](docs/bench-inference.svg)

![Wall Time](docs/bench-training-walltime.svg)

### Training Sweep — ai (500 steps, TinyStories 48MB, seq=64)

**NVIDIA RTX 5090 (32GB) — CUDA kernels + helix DNA optimizer:**

| Dim | Heads | Layers | Params | Steps/s | Loss | Wall (s) |
|-----|-------|--------|--------|---------|------|----------|
| 128 | 4 | 4 | 624K | **591** | 2.82 | 2.4 |
| 256 | 4 | 6 | 3.6M | **282** | 2.80 | 3.3 |
| 384 | 6 | 8 | 10.7M | **170** | 1.92 | 4.5 |
| 512 | 8 | 8 | 19M | **135** | 1.75 | 5.4 |
| 768 | 12 | 10 | 53M | **78** | 1.54 | 8.7 |
| 1024 | 16 | 12 | 114M | **41** | 1.51 | 15.3 |
| 1536 | 16 | 14 | 298M | **24** | 1.57 | 39.5 |
| 2048 | 16 | 16 | 605M | **10** | 1.92 | 129.2 |
| 3072 | 24 | 20 | 1.7B | **5.3** | 2.15 | 385.7 |
| 4096 | 32 | 24 | 3.6B | OOM | — | — |

**Apple M4 Max (40GB) — Metal 4 cooperative tensor GEMM + helix warm cache:**

| Dim | Heads | Layers | Params | Steps/s | Loss | Wall (s) |
|-----|-------|--------|--------|---------|------|----------|
| 128 | 4 | 4 | 624K | **185** | 1.77 | 4.3 |
| 256 | 4 | 6 | 3.6M | **156** | 1.60 | 5.2 |
| 384 | 6 | 8 | 10.7M | **111** | 1.71 | 7.2 |
| 512 | 8 | 8 | 19M | **98** | 1.73 | 7.9 |
| 768 | 12 | 10 | 53M | **50** | 1.70 | 13.9 |
| 1024 | 16 | 12 | 114M | **29** | 1.76 | 22.7 |
| 2048 | 16 | 16 | 605M | **7.3** | 2.00 | 81.0 |
| 3072 | 24 | 20 | 1.7B | 2.0 | 5.48 | 276.7 |
| 4096 | 32 | 24 | 3.6B | OOM | — | — |

### Training — ai vs PyTorch (dim=128, 624K params)

| GPU | PyTorch | ai | Speedup | Loss |
|-----|--------:|---:|--------:|------|
| NVIDIA RTX 5090 (32GB) | 165.8 steps/s | 591 steps/s | **3.6x** | 5.6 → 2.82 |
| Apple M4 Max (40GB) | 110.7 steps/s | 185 steps/s | **1.7x** | 5.6 → 1.77 |
| Apple M1 Pro (16GB) | — | 57 tok/s (infer) | — | — |

### Inference — Qwen2.5-0.5B

| GPU | PyTorch | ai | Speedup | Path |
|-----|--------:|---:|--------:|------|
| NVIDIA RTX 5070 Ti (16GB) | — | **99 tok/s** | — | CUDA Q8 fused matvec |
| Apple M1 Pro (16GB) | 3.3 tok/s | **57 tok/s** | **17x** | Metal Q8 fused matvec |

Automatic quantization: Q8 for models <4B params, Q4 for 7B+. Fused dequant-matvec kernels eliminate intermediate buffers entirely.

### MatMul Throughput (Engine.MatMul, 2048x2048)

| GPU | GFLOPS |
|-----|-------:|
| Apple M4 Max | 3,783 |
| NVIDIA RTX 5090 | 3,079 |
| Apple M1 Pro | 2,538 |
| CPU (pure Go) | 1.6 |

Host-copy path numbers. GPU-resident tensor operations and fused graph dispatch paths are significantly faster because they eliminate PCIe/unified-memory round-trips between operations.

## Architecture

Mongoose separates concerns into four layers: a hardware-agnostic interface, backend implementations that talk directly to vendor APIs, a calibration-based scheduler that routes work across heterogeneous GPUs, and training infrastructure (gradient accumulation, learning rate schedules, sparse embedding tracking) that sits on top.

### Layer 1: Engine Interface

Every backend implements the same `Engine` interface. Code that consumes an `Engine` never knows which GPU it's running on.

```go
type Engine interface {
    Name() string                                    // "cpu", "metal/Apple M4 Max", "cuda/RTX 5090"
    MatMul(a, b []float32, m, k, n int) []float32   // C[m,n] = A[m,k] @ B[k,n]
    RMSNorm(x, weight []float32, eps float32)        // in-place root mean square normalization
    SoftMax(x []float32, n int)                      // in-place softmax over n elements
    ReLU(x []float32)                                // in-place rectified linear unit
    VRAM() uint64                                     // available GPU memory in bytes (0 for CPU)
    Benchmark() float64                               // measured GFLOPS on a standard workload
}
```

For GPU-resident computation that avoids host round-trips:

```go
type TensorEngine interface {
    Engine
    FromHost(data []float32, shape []int) *Tensor    // upload once
    Zeros(shape []int) *Tensor                        // allocate on device
    ToHost(t *Tensor) []float32                       // download once
    Release(t *Tensor)                                // free device memory
    MatMulT(a, b *Tensor, m, k, n int) *Tensor       // C = A @ B, stays on GPU
    MatMulTransposeBT(a, b *Tensor, m, k, n int) *Tensor  // C = A @ B^T
    AddInPlace(a, b *Tensor)                          // a += b on device
    // ... ScaleT, ReLUT, ReLUBackwardT, TransposeT, CopyT
}
```

For training with BLAS-level primitives:

```go
type TrainEngine interface {
    Engine
    MatMulTransBInto(C, A, B []float32, m, k, n int) // C = A @ B^T, accumulate into buffer
    MatMulInto(C, A, B []float32, m, k, n int)       // C = A @ B, accumulate into buffer
    Nrm2(x []float32) float32                         // L2 norm (cblas_snrm2 on AMX)
    Scal(x []float32, alpha float32)                  // x *= alpha (cblas_sscal on AMX)
    GER(G, x, y []float32, m, n int, alpha float32)  // G += alpha * x @ y^T (rank-1 update)
    AdamWStep(D, G, M, V []float32, n int,
        lr, beta1, beta2, bc1, bc2, eps, wd float32)  // fused Adam + weight decay
}
```

For fused graph dispatch (one GPU submission = forward + backward + optimizer):

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

`BuildFullGraph` compiles the entire transformer at initialization. `GraphTrainStepAdam` fires one dispatch that executes the complete training step. This is the path that beats PyTorch — it eliminates all CPU-GPU synchronization points between operations.

### Layer 2: Backends

| Backend | Platform | Build Tag | Vendor API | What It Does |
|---------|----------|-----------|------------|--------------|
| **CUDA** | Linux | `linux && cgo` | cuBLAS, custom CUDA kernels | TF32 tensor core matmul via `cublasGemmEx`. Custom kernels loaded at runtime via `dlopen` for RMSNorm, RoPE, GQA attention, SiLU, fused Q8/Q4 dequant-matvec, cross-entropy, AdamW, and helix DNA step. GPU arena allocator: one `cudaMalloc` at init for 80% of VRAM, then best-fit sub-allocation with block merging — no `cudaMalloc`/`cudaFree` during training. |
| **Metal** | macOS | `darwin && cgo` | MPS, MPSGraph, Metal 4 `matmul2d` | Two dispatch paths. Metal 4 path (macOS 26+): pre-compiled `.metallib` with cooperative tensor GEMM kernels (`gemm4_bt`, `gemm4_nn`, `gemm4_tn` in FP16 and FP32) using `matmul2d` TensorOp with `execution_simdgroups<4>`. Detected and loaded at runtime — falls back to MPS tiled GEMM if unavailable. Custom Metal compute shaders compiled from inline source at init: RMSNorm (in-place + out-of-place + backward), RoPE (rotate_half, forward + conjugate backward), GQA causal attention (+ backward), SiLU gate mul (+ backward), fused Q8/Q4 dequant-matvec, cross-entropy loss, AdamW, DNA rung paired update, helix needle (INT8 + FP16 delta fold-back), add/scale/relu/zero. Buffer pool reuses MTLBuffer allocations capped at 8 per size class. `FusedBegin`/`FusedEnd` encode entire forward or backward pass into a single command buffer. `StorageModeShared` on all buffers — Apple Silicon unified memory means CPU and GPU access the same physical pages with zero copy. |
| **Accelerate** | macOS | `darwin && cgo` | Apple Accelerate framework | Routes to the AMX coprocessor on Apple Silicon for BLAS operations: `cblas_sgemm` for matmul, `cblas_snrm2`/`cblas_sscal`/`cblas_saxpy` for vector ops. Implements `TrainEngine` including `GER` (rank-1 outer product), `AdamWStep`, and all accumulating matmul variants. The AMX coprocessor runs in parallel with the GPU — helix uses it for gradient clipping while the CPU computes rung geometry. |
| **Intel Xe** *(optional)* | Linux | `linux && cgo && xe` | Level Zero | Requires explicit `-tags xe` build flag. Two modes. **Direct mode**: Level Zero initialization in-process, immediate command list dispatch, shared memory via `zeMemAllocShared` (unified — both CPU and GPU access at 64-byte alignment). **Daemon mode**: a standalone C process (`xe-daemon`) owns the Level Zero context to work around Intel IGC JIT compiler crashes in Go's address space. The daemon communicates over a Unix domain socket and shares a 256 MB `memfd`-backed arena with the Go process — split into a Go-write region (128 MB) and an Xe-write region (128 MB) separated by a 4 KB guard page (`mprotect PROT_NONE`) that segfaults on cross-boundary access. SPIR-V compute kernels (cross-entropy with fused softmax + gradient, RMSNorm, SiLU gate mul, element-wise add) are loaded at runtime from compiled `.spv` files via `zeModuleCreate`. Dispatch uses `zeKernelSetGroupSize` + `zeCommandListAppendLaunchKernel` on the immediate command list. |
| **WebGPU** | Any | `!cgo` | gogpu/wgpu (Vulkan, Metal, DX12) | Pure Go, zero CGo. Uses the gogpu WGSL shader compiler to run an 8x8 workgroup tiled matmul on whatever GPU the OS exposes through Vulkan (Linux, Android, RPi), Metal (macOS), or DX12 (Windows). Single binary runs on all platforms. Buffer pool reuses large allocations. Two power modes: `NewWebGPU()` (discrete GPU, high performance) and `NewWebGPULowPower()` (integrated GPU). |
| **CPU** | Any | always | none | Pure Go triple-nested-loop matmul. Go compiler auto-vectorizes to ARM NEON on Apple Silicon, SSE/AVX on x86. No CGo dependency. Fallback for platforms without a GPU or when all GPU backends return nil. Implements `Engine` only (not `TrainEngine` or `TensorEngine`). |

Every backend has a `*_stub.go` companion file with the negated build tag. Stubs implement all methods as no-ops returning nil/0/false. The library compiles cleanly on any platform — `go build ./...` never fails due to missing hardware.

### Layer 3: GPU-Agnostic Scheduler

The scheduler routes operations across heterogeneous GPUs using measured wall-clock times. No theoretical FLOP estimates. No vendor-specific heuristics. Every operation is timed on every available GPU during a calibration phase, and the assignment algorithm uses those measurements directly.

```go
sched := mongoose.NewScheduler(cuda, metal, cpu)

// Calibration: measure actual time per operation per GPU
sched.CalibrateMatMul(4096, 11008, 1)    // LLaMA-7B FFN shape
sched.CalibrateAll(mongoose.NormKey(4096), func(e Engine) {
    e.RMSNorm(x, w, 1e-6)
})

// Assignment: greedy finish-time minimization
ops := []OpKey{MatMulKey(n, dim, dim), NormKey(dim), MatMulKey(n, dim, ffn), ...}
assignments := sched.Assign(ops)  // each op → fastest GPU index
```

**Calibration** runs each operation 3 times on each GPU and records the average wall-clock microseconds. The `OpKey` identifies operations by type and dimensions (`"mm:64:4096:11008"`, `"norm:4096"`).

**Assignment** uses a greedy algorithm that minimizes finish time across a dependent operation sequence. For each operation, it finds the GPU where `accumulated_load + calibrated_time` is smallest. This naturally handles heterogeneous hardware: an H100 measures 113us on a large matmul while an Xe GPU measures 7100us — matmuls go to the H100. But RMSNorm measures 4us on H100 and 12us on Xe — the scheduler may route norms to Xe to free the H100 for matmuls when the load profile makes that beneficial.

#### WorkPool: Contention-Free Parallel Dispatch

For independent operations (batch of matmuls across multiple GPUs), the WorkPool pre-partitions work proportionally to measured GFLOPS before launching any goroutines.

```go
pool := mongoose.NewWorkPool()
pool.AddEngine("cuda", cuda, 3079.0)  // measured GFLOPS
pool.AddEngine("cpu", &CPU{}, 1.6)

results := pool.Run(items)  // 99.9% to CUDA, 0.1% to CPU
```

Each worker gets a contiguous slice of the work array. No locks in the hot path. No work-stealing. No contention. The fast GPU never blocks waiting for the slow GPU. Partition boundaries are computed once before goroutines launch.

#### HybridEngine: Multi-Backend Layer Assignment

For transformer inference and training across multiple GPUs, the `HybridEngine` assigns contiguous layer ranges to each backend proportional to its GFLOPS.

```go
hybrid := mongoose.NewHybridEngine()  // auto-discovers CUDA, WebGPU, CPU
hybrid.AssignLayers(32)                // distribute 32 transformer layers

eng := hybrid.BackendForLayer(17)      // which GPU handles layer 17?
```

Layer assignment ensures each backend gets at least one layer. Remainder layers go to the fastest backend. Activations pass between backends through host memory at layer boundaries.

### Layer 4: Training Infrastructure

#### Warm Cache (Unified Memory Optimizer State)

On Apple Silicon, a single `WarmCache` MTLBuffer holds all optimizer momentum and velocity state. Both CPU and GPU access the same physical memory — the helix optimizer writes m/v through `[]float32` slices on the CPU side, and GPU compute kernels read the same data through MTLBuffer byte offsets. No memcpy. No separate tensor allocations for optimizer state. The model weights are the only persistent memory footprint.

```go
cache := mtl.NewWarmCache(totalMVFloats)   // one MTLBuffer
mSlice := cache.Slice(mOffset, paramSize)  // CPU []float32 view
vSlice := cache.Slice(vOffset, paramSize)  // same physical memory the GPU reads
```

On CUDA, the equivalent is the L3 Bridge: pinned host memory allocated via `cudaHostAlloc` that the GPU reads through L3 cache coherency. CPU writes rung coefficients and hot row indices into the bridge; the GPU kernel reads them during dispatch.

```go
bridge := cuda.AllocL3Bridge(size)
data := bridge.Float32(0, 1024)       // CPU writes here
ptr := bridge.DevicePtr(0)            // GPU reads same physical pages
```

#### GPU Arena Allocator

CUDA training allocates one large arena (80% of free VRAM) at initialization via a single `cudaMalloc`. All subsequent allocations are sub-allocated from the arena using a best-fit algorithm with 256-byte alignment. Adjacent free blocks are merged on deallocation to prevent fragmentation. No `cudaMalloc`/`cudaFree` calls during training — just pointer arithmetic on the free list.

```go
arena := mongoose.NewGPUArena(0.8)          // 80% of free VRAM
ptr := arena.Alloc(dim * dim * 4)           // best-fit from free list
arena.Free(ptr, dim * dim * 4)              // merge with neighbors
stats := arena.Stats()                       // used, free, fragments, merges
```

#### Conductor (Sparse Embedding Tracking)

The Conductor tracks which embedding table rows are electrically active — an analogy to DNA gene expression where charge flows through stacked base pairs via pi-orbital overlap. Each training step, the conductor observes which token IDs appear in the batch. Over a configurable window (default 100 steps), it builds a charge map with exponential decay and identifies hot rows (charge above threshold). Dead rows are skipped during gradient computation.

```go
conductor := mongoose.NewConductor(vocabSize, 100)  // 100-step window
conductor.Observe(tokenIDs)                           // record gene expression
hotRows := conductor.HotRows()                        // sparse active indices
ratio := conductor.HotRatio()                         // typically 1-5% active
```

The conductor's hot row indices feed into needle's sparse dispatch kernel (`KNeedleSparse`), which fires threads only for active rows — 49 hot rows instead of 50K total, reducing optimizer compute by 99%+.

#### Gradient Accumulation

```go
accum := mongoose.NewGradAccumulator(params, accumSteps)
for microBatch := range data {
    accum.Accumulate(loss)
    if accum.Ready() {
        accum.Average()
        optimizer.Step(...)
        accum.Reset()
    }
}
```

Includes Chinchilla-derived helpers: `TokensPerBatch(paramCount)` recommends batch size based on model scale, `EstimateTrainingTokens(paramCount)` estimates total tokens for convergence (20x parameter count, clamped to literature minimums).

#### Learning Rate Schedules

Cosine decay with linear warmup — the standard schedule used by nanoGPT, Pythia, SmolLM, GPT-3, and Llama:

```go
sched := mongoose.NewCosineDecay(peakLR, minLR, totalSteps, warmupSteps)
lr := sched.LR(step)  // warmup → cosine curve → minLR floor
```

Default: `minLR = peakLR / 10`, warmup = 1% of total steps clamped to [100, 2000].

## Custom Kernels

### CUDA (`kernels/mongoose.cu`)

Compiled to shared library, loaded at runtime via `dlopen`:

```bash
cd kernels
nvcc -shared -o libmongoose_kernels.so mongoose.cu \
    -Xcompiler -fPIC -gencode arch=compute_90,code=compute_90
```

Kernel inventory: `mongoose_rmsnorm` (in-place + out-of-place + backward + weight gradient), `mongoose_relu` (forward + backward + indexed), `mongoose_silu_gate_mul` (+ backward), `mongoose_rope` (+ backward), `mongoose_embedding_gather`, `mongoose_causal_attention` (+ GQA + backward + decode), `mongoose_adamw`, `mongoose_helix_dna_step`, `mongoose_helix_needle` (+ paired), `mongoose_q8_matvec`, `mongoose_q4_matvec`, `mongoose_kv_cache_write`, `mongoose_cross_entropy`, `mongoose_softmax_ce`, `mongoose_dequant_int8_to_fp16`, `mongoose_dequant_int8_to_fp32`, `mongoose_copy`, `mongoose_zero`, `mongoose_sync`.

### Metal 4 (`kernels/gemm_metal4.metal`)

Pre-compiled to `.metallib`, loaded at runtime:

```bash
cd kernels
xcrun metal -std=metal4.0 -O2 -c gemm_metal4.metal -o gemm_metal4.air
xcrun metallib -o gemm_metal4.metallib gemm_metal4.air
```

Six GEMM variants using `matmul2d` TensorOp with cooperative tensor output and `execution_simdgroups<4>`: `gemm4_bt` (C = A @ B^T, FP16), `gemm4_nn` (C = A @ B, FP16), `gemm4_tn` (C = A^T @ B, FP16), `gemm4f_bt`, `gemm4f_nn`, `gemm4f_tn` (FP32 equivalents). Tile size: BM=64, BN=32. Threadgroup grid: `(ceil(N/32), ceil(M/64), 1)`, threads per group: `simdWidth * 4`.

Metal 4 detection is automatic — mongoose loads the `.metallib` from the binary's directory at init and sets `g_use_metal4_gemm = true`. If the library isn't found or Metal 4 isn't supported, all fused GEMM calls fall back to the MPS tiled kernel.

### Metal Compute Shaders (`metal_impl_darwin.m`)

Compiled from inline source strings at init via `newLibraryWithSource:` + `newComputePipelineStateWithFunction:`. Covers: RMSNorm (in-place, out-of-place, backward), RoPE (rotate_half), GQA causal attention (forward + backward), SiLU gate mul (forward + backward), KV cache write, FP32/FP16 conversion, INT8 dequant, fused Q8/Q4 dequant-matvec, AdamW, DNA rung paired update, helix needle (single + paired + inline), cross-entropy loss, gradient norm reduction, add/scale/relu/zero.

### Intel Xe SPIR-V (`kernels/spirv/`)

Compiled from OpenCL C to SPIR-V:

```bash
clang -cl-std=CL2.0 -target spir64 -O2 -o cross_entropy.bc -c -x cl cross_entropy.comp
llvm-spirv-18 cross_entropy.bc -o cross_entropy.spv
```

Fused cross-entropy kernel: per-position workgroup (256 threads) computes max-subtract → exp-sum → softmax → loss → gradient in three shared-memory reduction passes. Operates on the daemon's split arena — logits and targets in the Go-write region, losses and gradients in the Xe-write region.

## Build

```bash
# macOS: Metal + Accelerate backends
CGO_ENABLED=1 go build ./...

# Linux: CUDA backend
CGO_ENABLED=1 go build ./...

# Linux: CUDA + Intel Xe
CGO_ENABLED=1 go build -tags xe ./...

# Any platform: WebGPU/Vulkan, pure Go, no CGo
CGO_ENABLED=0 go build ./...
```

CUDA kernels (optional, enables fused Q8/Q4 inference + helix DNA training):
```bash
cd kernels
nvcc -shared -o libmongoose_kernels.so mongoose.cu \
    -Xcompiler -fPIC -gencode arch=compute_90,code=compute_90
```

Metal 4 kernels (optional, enables cooperative tensor GEMM on macOS 26+):
```bash
cd kernels
xcrun metal -std=metal4.0 -O2 -c gemm_metal4.metal -o gemm_metal4.air
xcrun metallib -o gemm_metal4.metallib gemm_metal4.air
```

Xe daemon (optional, for Intel GPU on Linux):
```bash
cd xe-daemon
make && sudo make install
```

## Test

```bash
go test -v ./...              # platform-neutral tests
go test -v -run CPU ./...     # CPU backend only
go test -v -run Scheduler ./...  # scheduler + workpool
```

## Ecosystem

| Package | What |
|---------|------|
| [mongoose](https://github.com/open-ai-org/mongoose) | GPU compute engine — this library |
| [ai](https://github.com/open-ai-org/ai) | CLI for training, inference, quantization, and serving LLMs |
| [helix](https://github.com/open-ai-org/helix) | DNA-inspired gradient descent optimizer with immune system and forward-only training |
| [needle](https://github.com/open-ai-org/needle) | Fused INT8 dequant + Adam CUDA/Metal kernels with FP16 delta fold-back |
| [gguf](https://github.com/open-ai-org/gguf) | GGUF, SafeTensors, and NumPy I/O — zero external dependencies |
| [tokenizer](https://github.com/open-ai-org/tokenizer) | BPE tokenizer (GPT-2, SentencePiece) — zero external dependencies |

## License

MIT
