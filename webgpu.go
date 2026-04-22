//go:build !cgo

package mongoose

import (
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"runtime"
	"time"

	"github.com/gogpu/gputypes"
	"github.com/gogpu/wgpu"
	_ "github.com/gogpu/wgpu/hal/allbackends" // Register Metal, Vulkan, DX12 HAL backends
)

// matmulShaderWGSL is the WGSL compute shader for matrix multiplication.
// C = A @ B where A is [M,K], B is [K,N], C is [M,N].
// Dimensions are passed via a uniform buffer.
const matmulShaderWGSL = `
struct Dims {
    M: u32,
    N: u32,
    K: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

@compute @workgroup_size(8, 8)
fn compute_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;

    if (row >= dims.M || col >= dims.N) {
        return;
    }

    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < dims.K; k = k + 1u) {
        sum = sum + A[row * dims.K + k] * B[k * dims.N + col];
    }

    C[row * dims.N + col] = sum;
}
`

// WebGPU implements Engine using the gogpu/wgpu pure-Go WebGPU implementation.
// One binary, every GPU: Metal (macOS), Vulkan (Linux/RPi), DX12 (Windows).
// No CGo. No C compiler. Pure Go.
type WebGPU struct {
	instance *wgpu.Instance
	adapter  *wgpu.Adapter
	device   *wgpu.Device
	pipeline *wgpu.ComputePipeline
	bgl      *wgpu.BindGroupLayout

	adapterName string
	backend     string

	maxBufSize     uint64
	maxStorageBind uint64

	// Buffer pool to avoid rapid create/destroy cycles (Metal stability)
	uniformBuf *wgpu.Buffer
	uniformSize uint64
	aBuf, bBuf, cBuf, stagingBuf *wgpu.Buffer
	aSize, bSize, cSize uint64
}

// NewWebGPU initializes the WebGPU compute backend (highest performance GPU).
// Returns nil if no GPU is available (falls back to CPU).
func NewWebGPU() *WebGPU {
	return newWebGPUWithPref(gputypes.PowerPreferenceHighPerformance)
}

// NewWebGPULowPower initializes WebGPU targeting the low-power/integrated GPU.
// Use this to select an Intel iGPU or AMD APU when a discrete GPU is also present.
func NewWebGPULowPower() *WebGPU {
	return newWebGPUWithPref(gputypes.PowerPreferenceLowPower)
}

func newWebGPUWithPref(pref gputypes.PowerPreference) *WebGPU {
	instance, err := wgpu.CreateInstance(&wgpu.InstanceDescriptor{
		Backends: gputypes.BackendsPrimary, // Metal on macOS, Vulkan on Linux, DX12 on Windows
	})
	if err != nil {
		log.Printf("WARN compute => WebGPU instance creation failed: %v", err)
		return nil
	}

	adapter, err := instance.RequestAdapter(&wgpu.RequestAdapterOptions{
		PowerPreference: pref,
	})
	if err != nil {
		log.Printf("WARN compute => WebGPU adapter request failed: %v", err)
		instance.Release()
		return nil
	}

	device, err := adapter.RequestDevice(&wgpu.DeviceDescriptor{
		Label: "tensorwire",
	})
	if err != nil {
		log.Printf("WARN compute => WebGPU device request failed: %v", err)
		adapter.Release()
		instance.Release()
		return nil
	}

	// Create matmul shader
	shader, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "tensorwire-matmul",
		WGSL:  matmulShaderWGSL,
	})
	if err != nil {
		log.Printf("WARN compute => WebGPU shader compilation failed: %v", err)
		device.Release()
		adapter.Release()
		instance.Release()
		return nil
	}
	// Note: don't release shader until pipeline is created

	// Bind group layout: [uniform dims][storage A][storage B][storage C]
	bgl, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "matmul-bgl",
		Entries: []wgpu.BindGroupLayoutEntry{
			{
				Binding:    0,
				Visibility: wgpu.ShaderStageCompute,
				Buffer:     &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeUniform},
			},
			{
				Binding:    1,
				Visibility: wgpu.ShaderStageCompute,
				Buffer:     &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeReadOnlyStorage},
			},
			{
				Binding:    2,
				Visibility: wgpu.ShaderStageCompute,
				Buffer:     &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeReadOnlyStorage},
			},
			{
				Binding:    3,
				Visibility: wgpu.ShaderStageCompute,
				Buffer:     &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeStorage},
			},
		},
	})
	if err != nil {
		log.Printf("WARN compute => WebGPU bind group layout failed: %v", err)
		device.Release()
		adapter.Release()
		instance.Release()
		return nil
	}

	pipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "matmul-pipeline-layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{bgl},
	})
	if err != nil {
		log.Printf("WARN compute => WebGPU pipeline layout failed: %v", err)
		shader.Release()
		bgl.Release()
		device.Release()
		adapter.Release()
		instance.Release()
		return nil
	}

	pipeline, err := device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:      "matmul-pipeline",
		Layout:     pipelineLayout,
		Module:     shader,
		EntryPoint: "compute_main",
	})

	// Safe to release shader and pipeline layout now — pipeline holds references
	shader.Release()
	pipelineLayout.Release()

	if err != nil {
		log.Printf("WARN compute => WebGPU compute pipeline failed: %v (GPU may not support compute)", err)
		bgl.Release()
		device.Release()
		adapter.Release()
		instance.Release()
		return nil
	}

	limits := device.Limits()

	g := &WebGPU{
		instance:       instance,
		adapter:        adapter,
		device:         device,
		pipeline:       pipeline,
		bgl:            bgl,
		maxBufSize:     limits.MaxBufferSize,
		maxStorageBind: limits.MaxStorageBufferBindingSize,
	}

	info := adapter.Info()
	g.adapterName = info.Name
	g.backend = info.Backend.String()

	log.Printf("[compute] WebGPU initialized: %s (%s)", g.adapterName, g.backend)
	return g
}

func (g *WebGPU) Name() string {
	return fmt.Sprintf("webgpu/%s (%s)", g.backend, g.adapterName)
}

// AdapterName returns the hardware device name (e.g., "NVIDIA GeForce RTX 5090").
func (g *WebGPU) AdapterName() string {
	return g.adapterName
}

// MatMul computes C = A @ B on the GPU with automatic fallback.
// Tries full GPU dispatch, then tiled GPU dispatch for oversized matrices,
// then CPU as last resort.
var wgpuFallbackLogged bool

func (g *WebGPU) MatMul(a, b []float32, m, k, n int) []float32 {
	result, err := g.gpuMatMulSingle(a, b, m, k, n)
	if err == nil {
		return result
	}

	result, err = g.tiledMatMul(a, b, m, k, n)
	if err == nil {
		return result
	}

	if !wgpuFallbackLogged {
		log.Printf("[webgpu] GPU matmul unavailable, using CPU fallback")
		wgpuFallbackLogged = true
	}
	return cpuMatMul(a, b, m, k, n)
}

// gpuMatMulSingle attempts a single GPU matmul dispatch.
// Returns an error if any GPU resource creation fails.
func (g *WebGPU) gpuMatMulSingle(a, b []float32, m, k, n int) ([]float32, error) {
	aSize := uint64(len(a)) * 4
	bSize := uint64(len(b)) * 4
	cSize := uint64(m * n * 4)

	// Use 64MB as safe limit — wgpu staging belt crashes on large mapped writes
	// even within device-reported limits. 64MB is conservative but avoids the
	// memmove SIGSEGV in stagingBelt.allocateOversized.
	maxBuf := uint64(64 * 1024 * 1024)
	if g.maxStorageBind > 0 && g.maxStorageBind < maxBuf {
		maxBuf = g.maxStorageBind
	}
	if g.maxBufSize > 0 && g.maxBufSize < maxBuf {
		maxBuf = g.maxBufSize
	}
	if aSize > maxBuf || bSize > maxBuf || cSize > maxBuf {
		return nil, fmt.Errorf("buffer %d bytes exceeds safe limit %d bytes", max64(aSize, max64(bSize, cSize)), maxBuf)
	}

	q := g.device.Queue()

	aBytes := float32sToBytes(a)
	bBytes := float32sToBytes(b)

	var err error

	if g.uniformBuf == nil {
		g.uniformBuf, err = g.device.CreateBuffer(&wgpu.BufferDescriptor{
			Label: "dims", Size: 16,
			Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
		})
		if err != nil {
			return nil, fmt.Errorf("uniform buffer: %w", err)
		}
		g.uniformSize = 16
	}
	if g.aBuf == nil || g.aSize < aSize {
		if g.aBuf != nil { g.aBuf.Release() }
		g.aBuf, err = g.device.CreateBuffer(&wgpu.BufferDescriptor{
			Label: "A", Size: aSize,
			Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
		})
		if err != nil {
			g.aBuf = nil
			g.aSize = 0
			return nil, fmt.Errorf("A buffer (%d bytes): %w", aSize, err)
		}
		g.aSize = aSize
	}
	if g.bBuf == nil || g.bSize < bSize {
		if g.bBuf != nil { g.bBuf.Release() }
		g.bBuf, err = g.device.CreateBuffer(&wgpu.BufferDescriptor{
			Label: "B", Size: bSize,
			Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
		})
		if err != nil {
			g.bBuf = nil
			g.bSize = 0
			return nil, fmt.Errorf("B buffer (%d bytes): %w", bSize, err)
		}
		g.bSize = bSize
	}
	if g.cBuf == nil || g.cSize < cSize {
		if g.cBuf != nil { g.cBuf.Release() }
		if g.stagingBuf != nil { g.stagingBuf.Release() }
		g.cBuf, err = g.device.CreateBuffer(&wgpu.BufferDescriptor{
			Label: "C", Size: cSize,
			Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
		})
		if err != nil {
			g.cBuf = nil
			g.stagingBuf = nil
			g.cSize = 0
			return nil, fmt.Errorf("C buffer (%d bytes): %w", cSize, err)
		}
		g.stagingBuf, err = g.device.CreateBuffer(&wgpu.BufferDescriptor{
			Label: "C-staging", Size: cSize,
			Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
		})
		if err != nil {
			g.cBuf.Release()
			g.cBuf = nil
			g.stagingBuf = nil
			g.cSize = 0
			return nil, fmt.Errorf("staging buffer (%d bytes): %w", cSize, err)
		}
		g.cSize = cSize
	}

	dimsBuf := make([]byte, 16)
	binary.LittleEndian.PutUint32(dimsBuf[0:], uint32(m))
	binary.LittleEndian.PutUint32(dimsBuf[4:], uint32(n))
	binary.LittleEndian.PutUint32(dimsBuf[8:], uint32(k))

	q.WriteBuffer(g.uniformBuf, 0, dimsBuf)
	q.WriteBuffer(g.aBuf, 0, aBytes)
	q.WriteBuffer(g.bBuf, 0, bBytes)

	bg, err := g.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "matmul-bg",
		Layout: g.bgl,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: g.uniformBuf, Offset: 0, Size: 16},
			{Binding: 1, Buffer: g.aBuf, Offset: 0, Size: aSize},
			{Binding: 2, Buffer: g.bBuf, Offset: 0, Size: bSize},
			{Binding: 3, Buffer: g.cBuf, Offset: 0, Size: cSize},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("bind group: %w", err)
	}
	defer bg.Release()

	encoder, err := g.device.CreateCommandEncoder(&wgpu.CommandEncoderDescriptor{Label: "matmul"})
	if err != nil {
		return nil, fmt.Errorf("command encoder: %w", err)
	}
	pass, err := encoder.BeginComputePass(&wgpu.ComputePassDescriptor{Label: "matmul"})
	if err != nil {
		return nil, fmt.Errorf("compute pass: %w", err)
	}
	pass.SetPipeline(g.pipeline)
	pass.SetBindGroup(0, bg, nil)

	groupsX := (uint32(n) + 7) / 8
	groupsY := (uint32(m) + 7) / 8
	pass.Dispatch(groupsX, groupsY, 1)
	pass.End()

	encoder.CopyBufferToBuffer(g.cBuf, 0, g.stagingBuf, 0, cSize)

	cmdBuf, err := encoder.Finish()
	if err != nil {
		return nil, fmt.Errorf("encoder finish: %w", err)
	}
	subIdx, err := q.Submit(cmdBuf)
	if err != nil {
		return nil, fmt.Errorf("queue submit: %w", err)
	}

	for q.Poll() < subIdx {
	}

	cBytes := make([]byte, cSize)
	q.ReadBuffer(g.stagingBuf, 0, cBytes)

	return bytesToFloat32s(cBytes), nil
}

// tiledMatMul splits C = A @ B into tiles that fit in GPU buffers.
// Tiles along both M (rows of A) and N (columns of B) as needed.
func (g *WebGPU) tiledMatMul(a, b []float32, m, k, n int) ([]float32, error) {
	limit := g.maxStorageBind
	if g.maxBufSize < limit {
		limit = g.maxBufSize
	}
	if limit == 0 {
		return nil, fmt.Errorf("device reports zero buffer limit")
	}

	maxElems := int(limit / 4)
	maxTileM := maxElems / k
	if maxTileM < 1 {
		return nil, fmt.Errorf("single row exceeds limit: k=%d limit=%d", k, limit)
	}
	if maxTileM > m {
		maxTileM = m
	}

	maxTileN := maxElems / k
	if maxTileN < 1 {
		return nil, fmt.Errorf("single col exceeds limit: k=%d limit=%d", k, limit)
	}
	if maxTileN > n {
		maxTileN = n
	}

	out := make([]float32, m*n)

	for rowStart := 0; rowStart < m; rowStart += maxTileM {
		tileM := maxTileM
		if rowStart+tileM > m {
			tileM = m - rowStart
		}

		aTile := a[rowStart*k : (rowStart+tileM)*k]

		for colStart := 0; colStart < n; colStart += maxTileN {
			tileN := maxTileN
			if colStart+tileN > n {
				tileN = n - colStart
			}

			var bTile []float32
			if tileN == n {
				bTile = b
			} else {
				bTile = make([]float32, k*tileN)
				for r := 0; r < k; r++ {
					copy(bTile[r*tileN:r*tileN+tileN], b[r*n+colStart:r*n+colStart+tileN])
				}
			}

			cTile, err := g.gpuMatMulSingle(aTile, bTile, tileM, k, tileN)
			if err != nil {
				return nil, fmt.Errorf("tile [%d:%d, %d:%d]: %w", rowStart, rowStart+tileM, colStart, colStart+tileN, err)
			}

			for r := 0; r < tileM; r++ {
				copy(out[(rowStart+r)*n+colStart:(rowStart+r)*n+colStart+tileN], cTile[r*tileN:r*tileN+tileN])
			}
		}
	}

	return out, nil
}

func max64(a, b uint64) uint64 {
	if a > b { return a }
	return b
}

// cpuMatMul computes C = A @ B in pure Go. Last-resort fallback.
func cpuMatMul(a, b []float32, m, k, n int) []float32 {
	out := make([]float32, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float32
			for l := 0; l < k; l++ {
				sum += a[i*k+l] * b[l*n+j]
			}
			out[i*n+j] = sum
		}
	}
	return out
}

// RMSNorm on CPU (small vector, not worth GPU dispatch overhead).
func (g *WebGPU) RMSNorm(x, weight []float32, eps float32) {
	n := len(x)
	var ss float32
	for i := 0; i < n; i++ {
		ss += x[i] * x[i]
	}
	ss = ss/float32(n) + eps
	ss = float32(1.0 / math.Sqrt(float64(ss)))
	for i := 0; i < n; i++ {
		x[i] = x[i] * ss * weight[i]
	}
}

// SoftMax on CPU.
func (g *WebGPU) SoftMax(x []float32, n int) {
	maxVal := x[0]
	for i := 1; i < n; i++ {
		if x[i] > maxVal {
			maxVal = x[i]
		}
	}
	var sum float32
	for i := 0; i < n; i++ {
		x[i] = float32(math.Exp(float64(x[i] - maxVal)))
		sum += x[i]
	}
	inv := float32(1.0) / sum
	for i := 0; i < n; i++ {
		x[i] *= inv
	}
}

// ReLU on CPU.
func (g *WebGPU) ReLU(x []float32) {
	for i := range x {
		if x[i] < 0 {
			x[i] = 0
		}
	}
}

// VRAM returns 0 for now (WebGPU doesn't expose VRAM directly).
func (g *WebGPU) VRAM() uint64 { return 0 }

// Benchmark runs a 512x512 matmul on the GPU.
func (g *WebGPU) Benchmark() float64 {
	const dim = 512
	a := make([]float32, dim*dim)
	b := make([]float32, dim*dim)
	for i := range a {
		a[i] = 0.001 * float32(i%1000)
		b[i] = 0.001 * float32(i%997)
	}

	runtime.GC()
	// Warmup
	g.MatMul(a, b, dim, dim, dim)

	start := time.Now()
	iterations := 50
	for i := 0; i < iterations; i++ {
		g.MatMul(a, b, dim, dim, dim)
	}
	elapsed := time.Since(start)

	flops := float64(2*dim*dim*dim*iterations) / elapsed.Seconds()
	return flops / 1e9
}

// Release frees GPU resources.
func (g *WebGPU) Release() {
	if g.pipeline != nil {
		g.pipeline.Release()
	}
	if g.bgl != nil {
		g.bgl.Release()
	}
	if g.device != nil {
		g.device.Release()
	}
	if g.adapter != nil {
		g.adapter.Release()
	}
	if g.instance != nil {
		g.instance.Release()
	}
}

// float32sToBytes converts a float32 slice to raw bytes.
func float32sToBytes(data []float32) []byte {
	buf := make([]byte, len(data)*4)
	for i, v := range data {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return buf
}

// bytesToFloat32s converts raw bytes to a float32 slice.
func bytesToFloat32s(data []byte) []float32 {
	out := make([]float32, len(data)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return out
}
