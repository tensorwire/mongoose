//go:build linux && cgo

package mongoose

/*
#include <cuda_runtime.h>
extern void* tw_get_kernel_lib();
#include <stdint.h>
#include <stdio.h>

// Graph capture lifecycle
static cudaGraph_t tw_graph_captured = NULL;

// Create a fresh stream for graph capture — avoids stale state on default stream
static cudaStream_t tw_graph_create_stream() {
    cudaStream_t s;
    cudaError_t err = cudaStreamCreate(&s);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GRAPH] cudaStreamCreate FAILED: %s\n", cudaGetErrorString(err));
        return NULL;
    }
    return s;
}

static void tw_graph_destroy_stream(cudaStream_t s) {
    if (s) cudaStreamDestroy(s);
}

// Abort any stale capture on a stream. After a failed capture, the stream
// may be stuck in capture mode. EndCapture with discard cleans it up.
static void tw_graph_abort_capture(cudaStream_t stream) {
    enum cudaStreamCaptureStatus status;
    cudaStreamGetCaptureInfo(stream, &status, NULL, NULL, NULL, NULL, NULL);
    if (status != cudaStreamCaptureStatusNone) {
        fprintf(stderr, "[GRAPH] aborting stale capture (status=%d)\n", (int)status);
        cudaGraph_t stale;
        cudaStreamEndCapture(stream, &stale);
        if (stale) cudaGraphDestroy(stale);
    }
}

// Minimal graph test: captures a cudaMemsetAsync on a fresh stream
static int tw_graph_test_api() {
    cudaStream_t stream = tw_graph_create_stream();
    if (!stream) return -1;

    float* tmp;
    cudaMalloc(&tmp, 256);
    cudaDeviceSynchronize();

    cudaGetLastError();
    cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GRAPH] API test: BeginCapture FAILED: %s\n", cudaGetErrorString(err));
        cudaFree(tmp);
        tw_graph_destroy_stream(stream);
        return -1;
    }
    cudaMemsetAsync(tmp, 0, 256, stream);
    cudaGraph_t graph;
    err = cudaStreamEndCapture(stream, &graph);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GRAPH] API test: EndCapture FAILED: %s\n", cudaGetErrorString(err));
        cudaFree(tmp);
        tw_graph_destroy_stream(stream);
        return -1;
    }
    cudaGraphExec_t exec;
    err = cudaGraphInstantiate(&exec, graph, 0);
    cudaGraphDestroy(graph);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GRAPH] API test: Instantiate FAILED: %s\n", cudaGetErrorString(err));
        cudaFree(tmp);
        tw_graph_destroy_stream(stream);
        return -1;
    }
    cudaGraphLaunch(exec, stream);
    cudaDeviceSynchronize();
    cudaGraphExecDestroy(exec);
    cudaFree(tmp);
    tw_graph_destroy_stream(stream);
    fprintf(stderr, "[GRAPH] API test: PASSED (cudaMemsetAsync on fresh stream)\n");
    return 0;
}

static int tw_graph_begin_capture(cudaStream_t stream) {
    tw_graph_abort_capture(stream);
    cudaGetLastError();
    cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GRAPH] begin capture FAILED: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

static cudaGraphExec_t tw_graph_end_capture(cudaStream_t stream) {
    cudaGraph_t graph;
    cudaError_t err = cudaStreamEndCapture(stream, &graph);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GRAPH] end capture FAILED: %s\n", cudaGetErrorString(err));
        return NULL;
    }

    cudaGraphExec_t exec;
    err = cudaGraphInstantiate(&exec, graph, 0);
    cudaGraphDestroy(graph);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GRAPH] instantiate FAILED: %s\n", cudaGetErrorString(err));
        return NULL;
    }
    return exec;
}

// Capture a new graph and update an existing exec in-place.
// Returns 0 on success, -1 on failure. ~0.1ms vs ~1ms for full re-instantiate.
static int tw_graph_update_exec(cudaGraphExec_t exec, cudaStream_t stream) {
    cudaGraph_t graph;
    cudaError_t err = cudaStreamEndCapture(stream, &graph);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GRAPH] update: EndCapture FAILED: %s\n", cudaGetErrorString(err));
        return -1;
    }
    cudaGraphExecUpdateResultInfo updateResult;
    err = cudaGraphExecUpdate(exec, graph, &updateResult);
    cudaGraphDestroy(graph);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GRAPH] cudaGraphExecUpdate FAILED: %s (result=%d)\n",
            cudaGetErrorString(err), (int)updateResult.result);
        return -1;
    }
    return 0;
}

static int tw_graph_launch(cudaGraphExec_t exec, cudaStream_t stream) {
    cudaError_t err = cudaGraphLaunch(exec, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GRAPH] launch FAILED: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

static void tw_graph_destroy(cudaGraphExec_t exec) {
    if (exec) cudaGraphExecDestroy(exec);
}

// Event helpers for multi-stream graph capture (Phase 2)
static cudaEvent_t tw_graph_create_event() {
    cudaEvent_t e;
    cudaError_t err = cudaEventCreate(&e);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GRAPH] cudaEventCreate FAILED: %s\n", cudaGetErrorString(err));
        return NULL;
    }
    return e;
}

static void tw_graph_destroy_event(cudaEvent_t e) {
    if (e) cudaEventDestroy(e);
}

static int tw_graph_event_record(cudaEvent_t e, cudaStream_t stream) {
    cudaError_t err = cudaEventRecord(e, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GRAPH] cudaEventRecord FAILED: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

static int tw_graph_stream_wait_event(cudaStream_t stream, cudaEvent_t e) {
    cudaError_t err = cudaStreamWaitEvent(stream, e, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GRAPH] cudaStreamWaitEvent FAILED: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

// Dynamic params: small device buffer for pos/seqLen
// The captured graph reads these from device memory instead of kernel args.
struct tw_graph_params {
    int pos;
    int seqLen;
    int reserved[6];
};

static struct tw_graph_params* tw_graph_alloc_params() {
    struct tw_graph_params* p;
    cudaMalloc(&p, sizeof(struct tw_graph_params));
    struct tw_graph_params init = {0, 1, {0}};
    cudaMemcpy(p, &init, sizeof(init), cudaMemcpyHostToDevice);
    return p;
}

static void tw_graph_update_params(struct tw_graph_params* p, int pos, int seqLen) {
    struct tw_graph_params h = {pos, seqLen, {0}};
    cudaMemcpyAsync(p, &h, sizeof(h), cudaMemcpyHostToDevice, 0);
}

static void tw_graph_update_params_on_stream(struct tw_graph_params* p, int pos, int seqLen, cudaStream_t stream) {
    struct tw_graph_params h = {pos, seqLen, {0}};
    cudaMemcpyAsync(p, &h, sizeof(h), cudaMemcpyHostToDevice, stream);
}

// Phase 3: graph-compatible fused dispatch (reads pos from device params)
#include <dlfcn.h>

// Graph kernel: same as fused but guaranteed capture-safe (no profile/sync)
typedef void (*fn_fused_graph)(
    const float**, const void**, const void**, const void**, const void**,
    const float**, const void**, const void**, const void**,
    float**, float**,
    float*, float*, float*, float*, float*,
    float*, float*, float*,
    float*, float*, float*,
    void*,
    const float*, const float*,
    int, int, int, int, int, int, int,
    int, int, int,
    cudaStream_t);

static fn_fused_graph k_fused_graph = NULL;

static int tw_graph_load_kernel() {
    k_fused_graph = (fn_fused_graph)dlsym(tw_get_kernel_lib(), "mongoose_partial_step_q4_fused_graph");
    if (!k_fused_graph) {
        fprintf(stderr, "[GRAPH] dlsym mongoose_partial_step_q4_fused_graph: %s (will fall back to non-graph dispatch)\n", dlerror());
        return -1;
    }
    fprintf(stderr, "[GRAPH] loaded graph kernel: mongoose_partial_step_q4_fused_graph\n");
    return 0;
}

static int tw_graph_has_kernel() { return k_fused_graph != NULL ? 1 : 0; }

static void tw_graph_fused_dispatch(
    void* norm1, void* wq, void* wk, void* wv, void* wo,
    void* norm2, void* wgate, void* wup, void* wdown,
    void* kCache, void* vCache,
    float* hidden, float* normed, float* Q, float* K, float* V,
    float* attnOut, float* proj, float* normed2,
    float* gatePre, float* upOut, float* ffnMid,
    void* q8Scratch,
    const float* cosTab, const float* sinTab,
    int dim, int kvDim, int headDim, int ffnDim,
    int nHeads, int nKVHeads, int halfHead,
    int pos, int layerStart, int layerEnd,
    cudaStream_t stream) {
    if (k_fused_graph)
        k_fused_graph(
            (const float**)norm1, (const void**)wq, (const void**)wk, (const void**)wv, (const void**)wo,
            (const float**)norm2, (const void**)wgate, (const void**)wup, (const void**)wdown,
            (float**)kCache, (float**)vCache,
            hidden, normed, Q, K, V, attnOut, proj, normed2,
            gatePre, upOut, ffnMid, q8Scratch, cosTab, sinTab,
            dim, kvDim, headDim, ffnDim, nHeads, nKVHeads, halfHead,
            pos, layerStart, layerEnd, stream);
}

*/
import "C"

import (
	"log"
	"runtime"
	"unsafe"
)

// LoadGraphKernel loads the graph-compatible fused dispatch from the .so.
// Must be called after KernelsLoad. Returns true if the graph kernel is available.
func LoadGraphKernel() bool {
	if C.tw_graph_has_kernel() != 0 {
		return true
	}
	C.tw_graph_load_kernel()
	return C.tw_graph_has_kernel() != 0
}

// HasGraphKernel returns true if the graph-compatible fused kernel is loaded.
func HasGraphKernel() bool {
	return C.tw_graph_has_kernel() != 0
}

// GraphParams mirrors the device-side tw_graph_params struct.
type GraphParams struct {
	Pos    int32
	SeqLen int32
	_      [6]int32
}

// GraphCaptureConfig describes what to capture in a CUDA graph.
type GraphCaptureConfig struct {
	LayerStart int
	LayerEnd   int
	StreamCount int // 1 = single-stream, 2 = multi-stream with pipeline break
	PipelineBreak int // layer index for Stage A/B split (0 = no split)
}

// CUDAGraph holds a captured CUDA graph for replaying a fixed kernel sequence.
type CUDAGraph struct {
	exec           C.cudaGraphExec_t
	params         *C.struct_tw_graph_params
	captureStream  C.cudaStream_t
	computeStream1 C.cudaStream_t // Phase 2: second stream for Stage B
	splitEvent     C.cudaEvent_t  // reused across UpdateAndLaunch for stable topology
	joinEvent      C.cudaEvent_t
	captured       bool
	config         GraphCaptureConfig
}

// CaptureGraph captures a forward pass as a CUDA graph based on the config.
// Phase 1: single-stream (config.StreamCount=1). Phase 2: multi-stream.
func (f *CUDAFusedInference) CaptureGraph(cfg GraphCaptureConfig, pos int) *CUDAGraph {
	if cfg.StreamCount == 0 { cfg.StreamCount = 1 }
	if cfg.StreamCount == 1 {
		return f.captureGraphSingleStream(pos, cfg)
	}
	if cfg.StreamCount == 2 {
		return f.captureGraphMultiStream(pos, cfg)
	}
	log.Printf("[GRAPH] unsupported StreamCount=%d (use 1 or 2)", cfg.StreamCount)
	return nil
}

func (f *CUDAFusedInference) captureGraphSingleStream(pos int, cfg GraphCaptureConfig) *CUDAGraph {
	layerStart, layerEnd := cfg.LayerStart, cfg.LayerEnd
	if !f.built {
		log.Printf("[GRAPH] cannot capture: not built")
		return nil
	}
	if !f.fusedReady {
		log.Printf("[GRAPH] fusedReady=false, building fused dispatch now (HasFusedDispatch=%v, nLayers=%d, norm1[0]=%v)",
			HasFusedDispatch(), f.nLayers, f.norm1[0] != nil)
		f.BuildFusedDispatch()
		if !f.fusedReady {
			log.Printf("[GRAPH] fused dispatch build failed — HasFusedDispatch=%v", HasFusedDispatch())
			return nil
		}
	}

	g := &CUDAGraph{
		config: cfg,
	}

	// Create fresh stream for capture — default stream has stale capture state
	g.captureStream = C.tw_graph_create_stream()
	if g.captureStream == nil {
		log.Printf("[GRAPH] failed to create capture stream")
		return nil
	}

	// Allocate params buffer BEFORE capture (allocations forbidden during capture)
	g.params = C.tw_graph_alloc_params()
	if g.params == nil {
		log.Printf("[GRAPH] params alloc failed")
		C.tw_graph_destroy_stream(g.captureStream)
		return nil
	}
	C.tw_graph_update_params(g.params, C.int(pos), C.int(pos+1))

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	C.cudaDeviceSynchronize()

	log.Printf("[GRAPH] starting capture on fresh stream: fusedReady=%v layerStart=%d layerEnd=%d pos=%d", f.fusedReady, layerStart, layerEnd, pos)
	if C.tw_graph_begin_capture(g.captureStream) != 0 {
		C.tw_graph_destroy_stream(g.captureStream)
		return nil
	}

	s := &f.scratch[0]
	hP := f.hidden[f.hiddenIdx].DevicePtr()
	f.dispatchGraphCapture(g, s, hP, layerStart, layerEnd, pos, g.captureStream)

	g.exec = C.tw_graph_end_capture(g.captureStream)
	if g.exec == nil {
		log.Printf("[GRAPH] capture failed")
		C.tw_graph_destroy_stream(g.captureStream)
		return nil
	}

	g.captured = true
	log.Printf("[GRAPH] captured: layers %d-%d, single-stream", layerStart, layerEnd-1)
	return g
}

func (f *CUDAFusedInference) captureGraphMultiStream(pos int, cfg GraphCaptureConfig) *CUDAGraph {
	if cfg.PipelineBreak <= cfg.LayerStart || cfg.PipelineBreak >= cfg.LayerEnd {
		log.Printf("[GRAPH] invalid PipelineBreak=%d (must be between LayerStart=%d and LayerEnd=%d)",
			cfg.PipelineBreak, cfg.LayerStart, cfg.LayerEnd)
		return nil
	}
	if !f.built {
		log.Printf("[GRAPH] cannot capture: not built")
		return nil
	}
	if !f.fusedReady {
		f.BuildFusedDispatch()
		if !f.fusedReady {
			log.Printf("[GRAPH] fused dispatch build failed")
			return nil
		}
	}

	f.SetVNodeCount(2)
	if f.scratch[1].Q == nil {
		log.Printf("[GRAPH] scratch[1] not allocated after SetVNodeCount(2)")
		return nil
	}

	g := &CUDAGraph{config: cfg}

	g.captureStream = C.tw_graph_create_stream()
	if g.captureStream == nil {
		log.Printf("[GRAPH] failed to create capture stream")
		return nil
	}
	g.computeStream1 = C.tw_graph_create_stream()
	if g.computeStream1 == nil {
		log.Printf("[GRAPH] failed to create compute stream 1")
		C.tw_graph_destroy_stream(g.captureStream)
		return nil
	}

	g.params = C.tw_graph_alloc_params()
	if g.params == nil {
		log.Printf("[GRAPH] params alloc failed")
		C.tw_graph_destroy_stream(g.captureStream)
		C.tw_graph_destroy_stream(g.computeStream1)
		return nil
	}
	C.tw_graph_update_params(g.params, C.int(pos), C.int(pos+1))

	g.splitEvent = C.tw_graph_create_event()
	g.joinEvent = C.tw_graph_create_event()
	if g.splitEvent == nil || g.joinEvent == nil {
		log.Printf("[GRAPH] event creation failed")
		C.tw_graph_destroy_stream(g.captureStream)
		C.tw_graph_destroy_stream(g.computeStream1)
		C.tw_graph_destroy_event(g.splitEvent)
		C.tw_graph_destroy_event(g.joinEvent)
		return nil
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	C.cudaDeviceSynchronize()

	log.Printf("[GRAPH] starting multi-stream capture: layers %d-%d, break=%d, pos=%d",
		cfg.LayerStart, cfg.LayerEnd-1, cfg.PipelineBreak, pos)

	if C.tw_graph_begin_capture(g.captureStream) != 0 {
		C.tw_graph_destroy_stream(g.captureStream)
		C.tw_graph_destroy_stream(g.computeStream1)
		C.tw_graph_destroy_event(g.splitEvent)
		C.tw_graph_destroy_event(g.joinEvent)
		return nil
	}

	hP := f.hidden[f.hiddenIdx].DevicePtr()
	sA := &f.scratch[0]
	sB := &f.scratch[1]

	// Stage A on captureStream (scratch[0])
	f.dispatchGraphCapture(g, sA, hP, cfg.LayerStart, cfg.PipelineBreak, pos, g.captureStream)

	// Fork: captureStream → computeStream1
	C.tw_graph_event_record(g.splitEvent, g.captureStream)
	C.tw_graph_stream_wait_event(g.computeStream1, g.splitEvent)

	// Stage B on computeStream1 (scratch[1])
	f.dispatchGraphCapture(g, sB, hP, cfg.PipelineBreak, cfg.LayerEnd, pos, g.computeStream1)

	// Join: computeStream1 → captureStream
	C.tw_graph_event_record(g.joinEvent, g.computeStream1)
	C.tw_graph_stream_wait_event(g.captureStream, g.joinEvent)

	g.exec = C.tw_graph_end_capture(g.captureStream)

	if g.exec == nil {
		log.Printf("[GRAPH] multi-stream capture failed")
		C.tw_graph_destroy_stream(g.captureStream)
		C.tw_graph_destroy_stream(g.computeStream1)
		return nil
	}

	g.captured = true
	log.Printf("[GRAPH] captured: layers %d-%d, multi-stream (break=%d)", cfg.LayerStart, cfg.LayerEnd-1, cfg.PipelineBreak)
	return g
}

// TestMinimalCapture verifies graph capture works on this CUDA runtime
// using a fresh stream and a simple cudaMemsetAsync operation.
// dispatchGraphCapture dispatches a fused forward pass during graph capture.
// Uses the graph kernel (capture-safe, no profile/sync) if available,
// otherwise falls back to the legacy kernel with profile=0.
func (f *CUDAFusedInference) dispatchGraphCapture(g *CUDAGraph, s *vnodeScratch, hP unsafe.Pointer, layerStart, layerEnd, pos int, stream C.cudaStream_t) {
	if HasGraphKernel() {
		C.tw_graph_fused_dispatch(
			unsafe.Pointer(&f.norm1Ptrs[0]), unsafe.Pointer(&f.wqPtrs[0]),
			unsafe.Pointer(&f.wkPtrs[0]), unsafe.Pointer(&f.wvPtrs[0]), unsafe.Pointer(&f.woPtrs[0]),
			unsafe.Pointer(&f.norm2Ptrs[0]), unsafe.Pointer(&f.wgatePtrs[0]),
			unsafe.Pointer(&f.wupPtrs[0]), unsafe.Pointer(&f.wdownPtrs[0]),
			unsafe.Pointer(&f.kCachePtrs[0]), unsafe.Pointer(&f.vCachePtrs[0]),
			(*C.float)(hP), (*C.float)(s.normed.DevicePtr()), (*C.float)(s.Q.DevicePtr()), (*C.float)(s.K.DevicePtr()), (*C.float)(s.V.DevicePtr()),
			(*C.float)(s.attnOut.DevicePtr()), (*C.float)(s.proj.DevicePtr()), (*C.float)(s.normed2.DevicePtr()),
			(*C.float)(s.gatePre.DevicePtr()), (*C.float)(s.upOut.DevicePtr()), (*C.float)(s.ffnMid.DevicePtr()),
			s.q8Scratch,
			(*C.float)(f.cosTab.DevicePtr()), (*C.float)(f.sinTab.DevicePtr()),
			C.int(f.dim), C.int(f.kvDim), C.int(f.headDim), C.int(f.ffnDim),
			C.int(f.nHeads), C.int(f.nKVHeads), C.int(f.halfHead),
			C.int(pos), C.int(layerStart), C.int(layerEnd),
			stream)
	} else {
		KPartialStepQ4FusedOnStream(
			unsafe.Pointer(&f.norm1Ptrs[0]), unsafe.Pointer(&f.wqPtrs[0]),
			unsafe.Pointer(&f.wkPtrs[0]), unsafe.Pointer(&f.wvPtrs[0]), unsafe.Pointer(&f.woPtrs[0]),
			unsafe.Pointer(&f.norm2Ptrs[0]), unsafe.Pointer(&f.wgatePtrs[0]),
			unsafe.Pointer(&f.wupPtrs[0]), unsafe.Pointer(&f.wdownPtrs[0]),
			unsafe.Pointer(&f.kCachePtrs[0]), unsafe.Pointer(&f.vCachePtrs[0]),
			hP, s.normed.DevicePtr(), s.Q.DevicePtr(), s.K.DevicePtr(), s.V.DevicePtr(),
			s.attnOut.DevicePtr(), s.proj.DevicePtr(), s.normed2.DevicePtr(),
			s.gatePre.DevicePtr(), s.upOut.DevicePtr(), s.ffnMid.DevicePtr(),
			s.q8Scratch,
			f.cosTab.DevicePtr(), f.sinTab.DevicePtr(),
			f.dim, f.kvDim, f.headDim, f.ffnDim, f.nHeads, f.nKVHeads, f.halfHead,
			pos, layerStart, layerEnd, 0,
			unsafe.Pointer(stream))
	}
}

func (f *CUDAFusedInference) TestMinimalCapture() bool {
	if !f.built { return false }

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	C.cudaDeviceSynchronize()

	if C.tw_graph_test_api() != 0 {
		log.Printf("[GRAPH] CUDA graph API BROKEN on this runtime")
		return false
	}

	log.Printf("[GRAPH] minimal capture: PASSED (fresh stream)")
	return true
}

// ValidateGraphCapture runs a single-token correctness test: compares graph
// output against non-graph output at the same position. Returns true if match.
func (f *CUDAFusedInference) ValidateGraphCapture(hiddenIn []float32, layerStart, layerEnd int) bool {
	// Step 0: test minimal capture first
	if !f.TestMinimalCapture() {
		log.Printf("[GRAPH] minimal capture failed — graph API not working on this runtime")
		return false
	}

	pos := 0

	// Run non-graph path
	refOut := make([]float32, f.dim)
	f.ResetKV()
	f.PartialStepQ4(hiddenIn, pos, layerStart, layerEnd, refOut, nil)
	log.Printf("[GRAPH] reference: h[0:4]=[%.4f,%.4f,%.4f,%.4f]", refOut[0], refOut[1], refOut[2], refOut[3])

	// Reset KV and capture graph
	f.ResetKV()
	cfg := GraphCaptureConfig{LayerStart: layerStart, LayerEnd: layerEnd, StreamCount: 1}
	g := f.CaptureGraph(cfg, pos)
	if g == nil {
		log.Printf("[GRAPH] validate: capture failed")
		return false
	}
	defer g.Destroy()

	// Upload same input and launch graph
	f.eng.UploadInto(f.hidden[f.hiddenIdx], hiddenIn)
	g.Launch(pos)
	KSync()

	// Read graph output
	graphOut := make([]float32, f.dim)
	copy(graphOut, f.eng.ToHost(f.hidden[f.hiddenIdx]))
	log.Printf("[GRAPH] graph:     h[0:4]=[%.4f,%.4f,%.4f,%.4f]", graphOut[0], graphOut[1], graphOut[2], graphOut[3])

	// Compare
	maxDiff := float32(0)
	for i := range refOut {
		d := refOut[i] - graphOut[i]
		if d < 0 { d = -d }
		if d > maxDiff { maxDiff = d }
	}
	log.Printf("[GRAPH] validate: maxDiff=%.6f (dim=%d)", maxDiff, f.dim)
	return maxDiff < 0.001
}

// ValidateMultiStreamGraph captures a two-stage graph and compares output against
// single-stream reference. This is THE KEY TEST for whether CUDA graphs fix the
// Blackwell multi-stream garble.
func (f *CUDAFusedInference) ValidateMultiStreamGraph(hiddenIn []float32, layerStart, layerEnd, pipelineBreak int) bool {
	pos := 0

	// Single-stream reference (proven correct by Phase 1)
	refOut := make([]float32, f.dim)
	f.ResetKV()
	f.PartialStepQ4(hiddenIn, pos, layerStart, layerEnd, refOut, nil)
	log.Printf("[GRAPH] multi-stream reference: h[0:4]=[%.4f,%.4f,%.4f,%.4f]", refOut[0], refOut[1], refOut[2], refOut[3])

	// Capture multi-stream graph
	f.ResetKV()
	cfg := GraphCaptureConfig{
		LayerStart:    layerStart,
		LayerEnd:      layerEnd,
		StreamCount:   2,
		PipelineBreak: pipelineBreak,
	}
	g := f.CaptureGraph(cfg, pos)
	if g == nil {
		log.Printf("[GRAPH] multi-stream validate: capture failed")
		return false
	}
	defer g.Destroy()

	f.eng.UploadInto(f.hidden[f.hiddenIdx], hiddenIn)
	g.Launch(pos)
	KSync()

	graphOut := make([]float32, f.dim)
	copy(graphOut, f.eng.ToHost(f.hidden[f.hiddenIdx]))
	log.Printf("[GRAPH] multi-stream graph:     h[0:4]=[%.4f,%.4f,%.4f,%.4f]", graphOut[0], graphOut[1], graphOut[2], graphOut[3])

	maxDiff := float32(0)
	for i := range refOut {
		d := refOut[i] - graphOut[i]
		if d < 0 { d = -d }
		if d > maxDiff { maxDiff = d }
	}
	log.Printf("[GRAPH] multi-stream validate: maxDiff=%.6f (dim=%d, break=%d)", maxDiff, f.dim, pipelineBreak)
	if maxDiff < 0.001 {
		log.Printf("[GRAPH] *** MULTI-STREAM GRAPH MATCHES REFERENCE — BLACKWELL FIX CONFIRMED ***")
		return true
	}
	log.Printf("[GRAPH] *** MULTI-STREAM GRAPH GARBLED — maxDiff=%.6f ***", maxDiff)
	return false
}

// UploadHidden copies host hidden state to the current GPU hidden buffer.
func (f *CUDAFusedInference) UploadHidden(data []float32) {
	f.eng.UploadInto(f.hidden[f.hiddenIdx], data)
}

// ReadHidden returns the current GPU hidden buffer contents.
func (f *CUDAFusedInference) ReadHidden() []float32 {
	return f.eng.ToHost(f.hidden[f.hiddenIdx])
}

// Launch replays the captured graph with updated pos on the capture stream.
// The params H2D memcpy is on the capture stream before graph launch,
// guaranteeing the update completes before the first kernel reads it.
func (g *CUDAGraph) Launch(pos int) {
	if !g.captured {
		return
	}
	C.tw_graph_update_params_on_stream(g.params, C.int(pos), C.int(pos+1), g.captureStream)
	C.tw_graph_launch(g.exec, g.captureStream)
}

// UpdateAndLaunch re-captures with new pos and updates the existing exec in-place
// via cudaGraphExecUpdate (~0.1ms vs ~1ms full re-instantiate). Then launches.
func (f *CUDAFusedInference) UpdateAndLaunch(g *CUDAGraph, pos int) bool {
	if !g.captured {
		return false
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	C.cudaDeviceSynchronize()

	if C.tw_graph_begin_capture(g.captureStream) != 0 {
		return false
	}

	if g.config.StreamCount == 2 && g.config.PipelineBreak > 0 {
		sA := &f.scratch[0]
		sB := &f.scratch[1]
		hP := f.hidden[f.hiddenIdx].DevicePtr()

		f.dispatchGraphCapture(g, sA, hP, g.config.LayerStart, g.config.PipelineBreak, pos, g.captureStream)

		C.tw_graph_event_record(g.splitEvent, g.captureStream)
		C.tw_graph_stream_wait_event(g.computeStream1, g.splitEvent)

		f.dispatchGraphCapture(g, sB, hP, g.config.PipelineBreak, g.config.LayerEnd, pos, g.computeStream1)

		C.tw_graph_event_record(g.joinEvent, g.computeStream1)
		C.tw_graph_stream_wait_event(g.captureStream, g.joinEvent)
	} else {
		s := &f.scratch[0]
		hP := f.hidden[f.hiddenIdx].DevicePtr()
		f.dispatchGraphCapture(g, s, hP, g.config.LayerStart, g.config.LayerEnd, pos, g.captureStream)
	}

	if C.tw_graph_update_exec(g.exec, g.captureStream) != 0 {
		log.Printf("[GRAPH] UpdateAndLaunch: exec update failed at pos=%d, falling back to re-capture", pos)
		return false
	}

	C.tw_graph_launch(g.exec, g.captureStream)
	return true
}

// Destroy frees the graph resources, streams, and events.
func (g *CUDAGraph) Destroy() {
	if g.exec != nil {
		C.tw_graph_destroy(g.exec)
		g.exec = nil
	}
	if g.splitEvent != nil {
		C.tw_graph_destroy_event(g.splitEvent)
		g.splitEvent = nil
	}
	if g.joinEvent != nil {
		C.tw_graph_destroy_event(g.joinEvent)
		g.joinEvent = nil
	}
	if g.captureStream != nil {
		C.tw_graph_destroy_stream(g.captureStream)
		g.captureStream = nil
	}
	if g.computeStream1 != nil {
		C.tw_graph_destroy_stream(g.computeStream1)
		g.computeStream1 = nil
	}
	g.captured = false
}
