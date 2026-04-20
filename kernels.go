//go:build linux && cgo

package mongoose

/*
#include <dlfcn.h>
#include <cuda_runtime.h>
#include <stdlib.h>

// Function pointers loaded from libmongoose_kernels.so
typedef void (*fn_rmsnorm)(float*, const float*, int, int, cudaStream_t);
typedef void (*fn_rmsnorm_out)(const float*, float*, const float*, int, int, cudaStream_t);
typedef void (*fn_relu)(float*, int, cudaStream_t);
typedef void (*fn_relu_out)(const float*, float*, int, cudaStream_t);
typedef void (*fn_relu_backward)(float*, const float*, const float*, int, cudaStream_t);
typedef void (*fn_add_inplace)(float*, const float*, int, cudaStream_t);
typedef void (*fn_scale_by_weight)(float*, const float*, int, int, cudaStream_t);
typedef void (*fn_embedding_gather)(float*, const float*, const float*, const int*, int, int, cudaStream_t);
typedef void (*fn_fused_add_rmsnorm)(const float*, const float*, float*, const float*, int, int, cudaStream_t);
typedef void (*fn_add_out)(const float*, const float*, float*, int, cudaStream_t);
typedef void (*fn_causal_attention)(const float*, const float*, const float*, float*, int, int, int, cudaStream_t);
typedef void (*fn_adamw)(float*, const float*, float*, float*, float, float, float, float, float, float, int, cudaStream_t);
typedef void (*fn_copy)(void*, const void*, size_t, cudaStream_t);
typedef void (*fn_zero)(void*, size_t, cudaStream_t);
typedef void (*fn_sync)(cudaStream_t);
typedef void (*fn_relu_and_index)(float*, int*, int*, int, cudaStream_t);
typedef void (*fn_sparse_matmul)(float*, const float*, const float*, const int*, int, int, int, cudaStream_t);
typedef void (*fn_silu_gate_mul)(const float*, const float*, float*, int, cudaStream_t);
typedef void (*fn_silu_gate_backward)(const float*, const float*, const float*, float*, float*, int, cudaStream_t);
typedef void (*fn_rope)(float*, const float*, const float*, int, int, int, int, cudaStream_t);
typedef void (*fn_rope_backward)(float*, const float*, const float*, int, int, int, int, cudaStream_t);
typedef void (*fn_scale_out)(const float*, float*, float, int, cudaStream_t);
typedef void (*fn_embed_gather2)(float*, const float*, const int*, int, int, cudaStream_t);
typedef void (*fn_cross_entropy)(const float*, const float*, int, int, const int*, float*, float*, float, int, cudaStream_t);
typedef void (*fn_attn_backward)(const float*, const float*, const float*, const float*, float*, float*, float*, int, int, int, int, int, cudaStream_t);
typedef void (*fn_attn_gqa)(const float*, const float*, const float*, float*, int, int, int, int, int, cudaStream_t);
typedef void (*fn_decode_attn)(const float*, const float*, const float*, float*, int, int, int, int, int, cudaStream_t);
typedef void (*fn_rmsnorm_save)(const float*, float*, const float*, float*, int, int, cudaStream_t);
typedef void (*fn_rmsnorm_bwd)(const float*, const float*, const float*, const float*, float*, int, int, cudaStream_t);
typedef void (*fn_softmax_ce)(float*, const int*, float*, float*, int, int, float, cudaStream_t);
typedef void (*fn_dequant_int8_fp16)(const void*, const float*, void*, int, int, cudaStream_t);
typedef void (*fn_dequant_int8_fp32)(const void*, const float*, float*, int, int, cudaStream_t);
typedef void (*fn_fp32_to_fp16)(const float*, void*, int, cudaStream_t);
typedef void (*fn_fp16_to_fp32)(const void*, float*, int, cudaStream_t);

static void* kernel_lib = NULL;
static fn_rmsnorm           k_rmsnorm = NULL;
static fn_rmsnorm_out       k_rmsnorm_out = NULL;
static fn_relu              k_relu = NULL;
static fn_relu_out          k_relu_out = NULL;
static fn_relu_backward     k_relu_backward = NULL;
static fn_add_inplace       k_add_inplace = NULL;
static fn_scale_by_weight   k_scale_by_weight = NULL;
static fn_embedding_gather  k_embedding_gather = NULL;
static fn_fused_add_rmsnorm k_fused_add_rmsnorm = NULL;
static fn_add_out           k_add_out = NULL;
static fn_causal_attention  k_causal_attention = NULL;
static fn_adamw             k_adamw = NULL;
static fn_copy              k_copy = NULL;
static fn_zero              k_zero = NULL;
static fn_sync              k_sync = NULL;
static fn_relu_and_index    k_relu_and_index = NULL;
static fn_sparse_matmul     k_sparse_matmul = NULL;
static fn_relu_and_index    k_relu_and_index_fp16 = NULL;
static fn_sparse_matmul     k_sparse_matmul_fp16 = NULL;
static fn_silu_gate_mul     k_silu_gate_mul = NULL;
static fn_silu_gate_backward k_silu_gate_backward = NULL;
static fn_rope              k_rope = NULL;
static fn_rope_backward     k_rope_backward = NULL;
static fn_scale_out         k_scale_out = NULL;
static fn_embed_gather2     k_embed_gather2 = NULL;
static fn_cross_entropy     k_cross_entropy = NULL;
static fn_attn_backward     k_attn_backward = NULL;
static fn_attn_gqa          k_attn_gqa = NULL;
static fn_decode_attn       k_decode_attn = NULL;
static fn_rmsnorm_save      k_rmsnorm_save = NULL;
static fn_rmsnorm_bwd       k_rmsnorm_bwd = NULL;
static fn_softmax_ce        k_softmax_ce = NULL;
static fn_dequant_int8_fp16 k_dequant_int8_fp16 = NULL;
static fn_dequant_int8_fp32 k_dequant_int8_fp32 = NULL;
static fn_fp32_to_fp16     k_fp32_to_fp16 = NULL;
static fn_fp16_to_fp32     k_fp16_to_fp32 = NULL;

int tw_load_kernels(const char* path) {
    kernel_lib = dlopen(path, RTLD_NOW);
    if (!kernel_lib) return -1;

    k_rmsnorm          = (fn_rmsnorm)dlsym(kernel_lib, "mongoose_rmsnorm");
    k_rmsnorm_out      = (fn_rmsnorm_out)dlsym(kernel_lib, "mongoose_rmsnorm_out");
    k_relu             = (fn_relu)dlsym(kernel_lib, "mongoose_relu");
    k_relu_out         = (fn_relu_out)dlsym(kernel_lib, "mongoose_relu_out");
    k_relu_backward    = (fn_relu_backward)dlsym(kernel_lib, "mongoose_relu_backward");
    k_add_inplace      = (fn_add_inplace)dlsym(kernel_lib, "mongoose_add_inplace");
    k_scale_by_weight  = (fn_scale_by_weight)dlsym(kernel_lib, "mongoose_scale_by_weight");
    k_embedding_gather = (fn_embedding_gather)dlsym(kernel_lib, "mongoose_embedding_gather");
    k_fused_add_rmsnorm = (fn_fused_add_rmsnorm)dlsym(kernel_lib, "mongoose_fused_add_rmsnorm");
    k_add_out           = (fn_add_out)dlsym(kernel_lib, "mongoose_add_out");
    k_causal_attention = (fn_causal_attention)dlsym(kernel_lib, "mongoose_causal_attention");
    k_adamw            = (fn_adamw)dlsym(kernel_lib, "mongoose_adamw");
    k_copy             = (fn_copy)dlsym(kernel_lib, "mongoose_copy");
    k_zero             = (fn_zero)dlsym(kernel_lib, "mongoose_zero");
    k_sync             = (fn_sync)dlsym(kernel_lib, "mongoose_sync");
    k_relu_and_index   = (fn_relu_and_index)dlsym(kernel_lib, "mongoose_relu_and_index");
    k_sparse_matmul    = (fn_sparse_matmul)dlsym(kernel_lib, "mongoose_sparse_matmul");
    k_relu_and_index_fp16 = (fn_relu_and_index)dlsym(kernel_lib, "mongoose_relu_and_index_fp16");
    k_sparse_matmul_fp16  = (fn_sparse_matmul)dlsym(kernel_lib, "mongoose_sparse_matmul_fp16");
    k_silu_gate_mul       = (fn_silu_gate_mul)dlsym(kernel_lib, "mongoose_silu_gate_mul");
    k_silu_gate_backward  = (fn_silu_gate_backward)dlsym(kernel_lib, "mongoose_silu_gate_backward");
    k_rope                = (fn_rope)dlsym(kernel_lib, "mongoose_rope");
    k_rope_backward       = (fn_rope_backward)dlsym(kernel_lib, "mongoose_rope_backward");
    k_scale_out           = (fn_scale_out)dlsym(kernel_lib, "mongoose_scale");
    k_embed_gather2       = (fn_embed_gather2)dlsym(kernel_lib, "mongoose_embed_gather");
    k_cross_entropy       = (fn_cross_entropy)dlsym(kernel_lib, "mongoose_cross_entropy");
    k_attn_backward       = (fn_attn_backward)dlsym(kernel_lib, "mongoose_causal_attention_backward");
    k_attn_gqa            = (fn_attn_gqa)dlsym(kernel_lib, "mongoose_causal_attention_gqa");
    k_decode_attn         = (fn_decode_attn)dlsym(kernel_lib, "mongoose_decode_attention");
    k_rmsnorm_save        = (fn_rmsnorm_save)dlsym(kernel_lib, "mongoose_rmsnorm_out_save");
    k_rmsnorm_bwd         = (fn_rmsnorm_bwd)dlsym(kernel_lib, "mongoose_rmsnorm_backward");
    k_softmax_ce          = (fn_softmax_ce)dlsym(kernel_lib, "mongoose_softmax_ce");
    k_dequant_int8_fp16   = (fn_dequant_int8_fp16)dlsym(kernel_lib, "mongoose_dequant_int8_to_fp16");
    k_dequant_int8_fp32   = (fn_dequant_int8_fp32)dlsym(kernel_lib, "mongoose_dequant_int8_to_fp32");
    k_fp32_to_fp16        = (fn_fp32_to_fp16)dlsym(kernel_lib, "mongoose_fp32_to_fp16");
    k_fp16_to_fp32        = (fn_fp16_to_fp32)dlsym(kernel_lib, "mongoose_fp16_to_fp32");

    if (!k_rmsnorm || !k_relu || !k_add_inplace || !k_embedding_gather) return -2;
    return 0;
}

int tw_kernels_loaded() { return kernel_lib != NULL ? 1 : 0; }

// Wrappers that call through function pointers
void tw_k_rmsnorm(float* x, const float* w, int seqLen, int dim) {
    if (k_rmsnorm) k_rmsnorm(x, w, seqLen, dim, 0);
}
void tw_k_rmsnorm_out(const float* input, float* out, const float* w, int seqLen, int dim) {
    if (k_rmsnorm_out) k_rmsnorm_out(input, out, w, seqLen, dim, 0);
}
void tw_k_relu(float* x, int n) {
    if (k_relu) k_relu(x, n, 0);
}
void tw_k_relu_out(const float* input, float* out, int n) {
    if (k_relu_out) k_relu_out(input, out, n, 0);
}
void tw_k_relu_backward(float* out, const float* dOut, const float* input, int n) {
    if (k_relu_backward) k_relu_backward(out, dOut, input, n, 0);
}
void tw_k_add_inplace(float* a, const float* b, int n) {
    if (k_add_inplace) k_add_inplace(a, b, n, 0);
}
void tw_k_scale_by_weight(float* x, const float* w, int seqLen, int dim) {
    if (k_scale_by_weight) k_scale_by_weight(x, w, seqLen, dim, 0);
}
void tw_k_embedding_gather(float* out, const float* tokEmb, const float* posEmb, const int* tokens, int seqLen, int dim) {
    if (k_embedding_gather) k_embedding_gather(out, tokEmb, posEmb, tokens, seqLen, dim, 0);
}
void tw_k_fused_add_rmsnorm(const float* a, const float* b, float* out, const float* w, int seqLen, int dim) {
    if (k_fused_add_rmsnorm) k_fused_add_rmsnorm(a, b, out, w, seqLen, dim, 0);
}
void tw_k_add_out(const float* a, const float* b, float* out, int n) {
    if (k_add_out) k_add_out(a, b, out, n, 0);
}
void tw_k_causal_attention(const float* Q, const float* K, const float* V, float* out,
                            int seqLen, int dim, int numHeads) {
    if (k_causal_attention) k_causal_attention(Q, K, V, out, seqLen, dim, numHeads, 0);
}
void tw_k_adamw(float* param, const float* grad, float* m, float* v,
                float lr, float wd, float beta1, float beta2, float bc1, float bc2, int n) {
    if (k_adamw) k_adamw(param, grad, m, v, lr, wd, beta1, beta2, bc1, bc2, n, 0);
}
void tw_k_copy(void* dst, const void* src, size_t bytes) {
    if (k_copy) k_copy(dst, src, bytes, 0);
}
void tw_k_zero(void* ptr, size_t bytes) {
    if (k_zero) k_zero(ptr, bytes, 0);
}
const char* tw_cuda_check() {
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) return cudaGetErrorString(err);
    return NULL;
}
void tw_k_sync() {
    if (k_sync) k_sync(0);
}
void tw_k_relu_and_index(float* x, int* activeIdx, int* activeCount, int n) {
    if (k_relu_and_index) k_relu_and_index(x, activeIdx, activeCount, n, 0);
}
void tw_k_sparse_matmul(float* out, const float* WT, const float* x,
                         const int* activeIdx, int activeCount,
                         int rows, int cols) {
    if (k_sparse_matmul) k_sparse_matmul(out, WT, x, activeIdx, activeCount, rows, cols, 0);
}
void tw_k_relu_and_index_fp16(void* x, int* activeIdx, int* activeCount, int n) {
    if (k_relu_and_index_fp16) k_relu_and_index_fp16((float*)x, activeIdx, activeCount, n, 0);
}
void tw_k_sparse_matmul_fp16(void* out, const void* WT, const void* x,
                              const int* activeIdx, int activeCount,
                              int rows, int cols) {
    if (k_sparse_matmul_fp16) k_sparse_matmul_fp16((float*)out, (const float*)WT, (const float*)x, activeIdx, activeCount, rows, cols, 0);
}
void tw_k_silu_gate_mul(const float* gate, const float* up, float* out, int n) {
    if (k_silu_gate_mul) k_silu_gate_mul(gate, up, out, n, 0);
}
void tw_k_silu_gate_backward(const float* dOut, const float* gate, const float* up,
                              float* dGate, float* dUp, int n) {
    if (k_silu_gate_backward) k_silu_gate_backward(dOut, gate, up, dGate, dUp, n, 0);
}
void tw_k_rope(float* x, const float* cos_tab, const float* sin_tab,
               int seqLen, int dim, int headDim, int nHeads) {
    if (k_rope) k_rope(x, cos_tab, sin_tab, seqLen, dim, headDim, nHeads, 0);
}
void tw_k_rope_backward(float* dx, const float* cos_tab, const float* sin_tab,
                         int seqLen, int dim, int headDim, int nHeads) {
    if (k_rope_backward) k_rope_backward(dx, cos_tab, sin_tab, seqLen, dim, headDim, nHeads, 0);
}
void tw_k_scale_out(const float* x, float* out, float alpha, int n) {
    if (k_scale_out) k_scale_out(x, out, alpha, n, 0);
}
void tw_k_embed_gather2(float* out, const float* embed, const int* tokens, int seqLen, int dim) {
    if (k_embed_gather2) k_embed_gather2(out, embed, tokens, seqLen, dim, 0);
}
void tw_k_cross_entropy(const float* hidden, const float* embedW, int D, int vocabSize,
                         const int* targets, float* losses, float* dHidden, float invN, int nPos) {
    if (k_cross_entropy) k_cross_entropy(hidden, embedW, D, vocabSize, targets, losses, dHidden, invN, nPos, 0);
}
void tw_k_rmsnorm_save(const float* input, float* out, const float* weight, float* scales,
                        int seqLen, int dim) {
    if (k_rmsnorm_save) k_rmsnorm_save(input, out, weight, scales, seqLen, dim, 0);
}
void tw_k_rmsnorm_bwd(const float* dOut, const float* xIn, const float* weight,
                       const float* scales, float* dx, int seqLen, int dim) {
    if (k_rmsnorm_bwd) k_rmsnorm_bwd(dOut, xIn, weight, scales, dx, seqLen, dim, 0);
}
void tw_k_decode_attn(const float* Q, const float* K, const float* V, float* out,
                       int cacheLen, int dim, int kvDim, int numHeads, int numKVHeads) {
    if (k_decode_attn) k_decode_attn(Q, K, V, out, cacheLen, dim, kvDim, numHeads, numKVHeads, 0);
}
void tw_k_attn_gqa(const float* Q, const float* K, const float* V, float* out,
                    int seqLen, int dim, int kvDim, int numHeads, int numKVHeads) {
    if (k_attn_gqa) k_attn_gqa(Q, K, V, out, seqLen, dim, kvDim, numHeads, numKVHeads, 0);
}
void tw_k_attn_backward(const float* Q, const float* K, const float* V, const float* dOut,
                         float* dQ, float* dK, float* dV,
                         int seqLen, int dim, int kvDim, int numHeads, int numKVHeads) {
    if (k_attn_backward) k_attn_backward(Q, K, V, dOut, dQ, dK, dV,
        seqLen, dim, kvDim, numHeads, numKVHeads, 0);
}
void tw_k_softmax_ce(float* logits, const int* targets, float* losses, float* grad,
                     int nPos, int vocabSize, float invN) {
    if (k_softmax_ce) k_softmax_ce(logits, targets, losses, grad, nPos, vocabSize, invN, 0);
}
int tw_softmax_ce_loaded() { return k_softmax_ce != NULL ? 1 : 0; }
void tw_k_dequant_int8_fp16(const void* data, const float* scales, void* out, int rows, int cols) {
    if (k_dequant_int8_fp16) k_dequant_int8_fp16(data, scales, out, rows, cols, 0);
}
void tw_k_dequant_int8_fp32(const void* data, const float* scales, float* out, int rows, int cols) {
    if (k_dequant_int8_fp32) k_dequant_int8_fp32(data, scales, out, rows, cols, 0);
}
void tw_k_fp32_to_fp16(const float* in, void* out, int n) {
    if (k_fp32_to_fp16) k_fp32_to_fp16(in, out, n, 0);
}
void tw_k_fp16_to_fp32(const void* in, float* out, int n) {
    if (k_fp16_to_fp32) k_fp16_to_fp32(in, out, n, 0);
}
int tw_dequant_kernels_loaded() {
    return (k_dequant_int8_fp16 != NULL && k_dequant_int8_fp32 != NULL) ? 1 : 0;
}
int tw_train_kernels_loaded() {
    return (k_silu_gate_mul && k_rope && k_rmsnorm_out && k_adamw) ? 1 : 0;
}
int tw_attn_backward_loaded() {
    return k_attn_backward != NULL ? 1 : 0;
}
int tw_sparse_kernels_loaded() {
    return (k_relu_and_index != NULL && k_sparse_matmul != NULL) ? 1 : 0;
}
int tw_sparse_fp16_kernels_loaded() {
    return (k_relu_and_index_fp16 != NULL && k_sparse_matmul_fp16 != NULL) ? 1 : 0;
}
*/
import "C"

import (
	"log"
	"math"
	"os"
	"path/filepath"
	"unsafe"
)

// LoadKernels loads libmongoose_kernels.so from the given path or searches common locations.
func LoadKernels(paths ...string) bool {
	searchPaths := append(paths,
		"./libmongoose_kernels.so",
		"./kernels/libmongoose_kernels.so",
		filepath.Join(os.Getenv("HOME"), "tensorwire/libmongoose_kernels.so"),
		"/usr/local/lib/libmongoose_kernels.so",
	)

	for _, p := range searchPaths {
		cPath := C.CString(p)
		ret := C.tw_load_kernels(cPath)
		C.free(unsafe.Pointer(cPath))
		if ret == 0 {
			log.Printf("[mongoose] CUDA kernels loaded from %s", p)
			return true
		}
	}
	return false
}

// KernelsLoaded returns true if the CUDA kernel library was loaded.
func KernelsLoaded() bool {
	return C.tw_kernels_loaded() == 1
}

// GPU kernel operations — all data stays on GPU

// KRMSNorm applies RMSNorm on GPU in-place. x[seqLen,dim], weight[dim].
func KRMSNorm(xPtr, weightPtr unsafe.Pointer, seqLen, dim int) {
	C.tw_k_rmsnorm((*C.float)(xPtr), (*C.float)(weightPtr), C.int(seqLen), C.int(dim))
}

// KRMSNormOut: out = rmsnorm(input, weight). Input NOT modified. Zero-copy.
func KRMSNormOut(inputPtr, outPtr, weightPtr unsafe.Pointer, seqLen, dim int) {
	C.tw_k_rmsnorm_out((*C.float)(inputPtr), (*C.float)(outPtr), (*C.float)(weightPtr), C.int(seqLen), C.int(dim))
}

// KReLUOut: out = relu(input). Input NOT modified. Zero-copy.
func KReLUOut(inputPtr, outPtr unsafe.Pointer, n int) {
	C.tw_k_relu_out((*C.float)(inputPtr), (*C.float)(outPtr), C.int(n))
}

// KReLU applies ReLU on GPU in-place.
func KReLU(xPtr unsafe.Pointer, n int) {
	C.tw_k_relu((*C.float)(xPtr), C.int(n))
}

// KReLUBackward: out = dOut * (input > 0), all on GPU.
func KReLUBackward(outPtr, dOutPtr, inputPtr unsafe.Pointer, n int) {
	C.tw_k_relu_backward((*C.float)(outPtr), (*C.float)(dOutPtr), (*C.float)(inputPtr), C.int(n))
}

// KAddInPlace: a += b on GPU.
func KAddInPlace(aPtr, bPtr unsafe.Pointer, n int) {
	C.tw_k_add_inplace((*C.float)(aPtr), (*C.float)(bPtr), C.int(n))
}

// KScaleByWeight: x[i*dim+j] *= weight[j] on GPU.
func KScaleByWeight(xPtr, weightPtr unsafe.Pointer, seqLen, dim int) {
	C.tw_k_scale_by_weight((*C.float)(xPtr), (*C.float)(weightPtr), C.int(seqLen), C.int(dim))
}

// KEmbeddingGather: out[i] = tokEmb[tokens[i]] + posEmb[i] on GPU.
func KEmbeddingGather(outPtr, tokEmbPtr, posEmbPtr unsafe.Pointer, tokensPtr unsafe.Pointer, seqLen, dim int) {
	C.tw_k_embedding_gather((*C.float)(outPtr), (*C.float)(tokEmbPtr), (*C.float)(posEmbPtr), (*C.int)(tokensPtr), C.int(seqLen), C.int(dim))
}

// KFusedAddRMSNorm: out = rmsnorm(a + b, weight) — one dispatch instead of two.
func KFusedAddRMSNorm(aPtr, bPtr, outPtr, weightPtr unsafe.Pointer, seqLen, dim int) {
	C.tw_k_fused_add_rmsnorm((*C.float)(aPtr), (*C.float)(bPtr), (*C.float)(outPtr), (*C.float)(weightPtr), C.int(seqLen), C.int(dim))
}

// KAddOut: out = a + b, a and b NOT modified.
func KAddOut(aPtr, bPtr, outPtr unsafe.Pointer, n int) {
	C.tw_k_add_out((*C.float)(aPtr), (*C.float)(bPtr), (*C.float)(outPtr), C.int(n))
}

// KCausalAttention: multi-head causal self-attention entirely on GPU.
// Q,K,V are [seqLen, dim] on GPU, out is [seqLen, dim] on GPU.
func KCausalAttention(qPtr, kPtr, vPtr, outPtr unsafe.Pointer, seqLen, dim, numHeads int) {
	C.tw_k_causal_attention(
		(*C.float)(qPtr), (*C.float)(kPtr), (*C.float)(vPtr), (*C.float)(outPtr),
		C.int(seqLen), C.int(dim), C.int(numHeads),
	)
}

// KAdamW: full AdamW update on GPU. No CPU round-trip. No PCIe transfer.
func KAdamW(paramPtr, gradPtr, mPtr, vPtr unsafe.Pointer, lr, wd float32, step int, n int) {
	beta1 := float32(0.9)
	beta2 := float32(0.999)
	bc1 := C.float(1.0 - math.Pow(float64(beta1), float64(step)))
	bc2 := C.float(1.0 - math.Pow(float64(beta2), float64(step)))
	C.tw_k_adamw(
		(*C.float)(paramPtr), (*C.float)(gradPtr), (*C.float)(mPtr), (*C.float)(vPtr),
		C.float(lr), C.float(wd), C.float(beta1), C.float(beta2), bc1, bc2, C.int(n),
	)
}

// KCopy: device-to-device memcpy.
func KCopy(dst, src unsafe.Pointer, bytes int) {
	C.tw_k_copy(dst, src, C.size_t(bytes))
}

// KZero: memset zero on GPU.
func KZero(ptr unsafe.Pointer, bytes int) {
	C.tw_k_zero(ptr, C.size_t(bytes))
}

// CUDACheck returns the last CUDA error string, or empty string if no error.
func CUDACheck() string {
	s := C.tw_cuda_check()
	if s == nil { return "" }
	return C.GoString(s)
}

// KSync: synchronize default stream.
func KSync() {
	C.tw_k_sync()
}

// KReLUAndIndex applies ReLU in-place AND builds a compact index of non-zero dimensions.
// Fused kernel — no CPU round-trip for the sparsity scan.
// xPtr: [n] float32 on GPU, modified in-place
// activeIdxPtr: [n] int32 on GPU, filled with indices of non-zero elements
// activeCountPtr: [1] int32 on GPU, set to number of non-zero elements
func KReLUAndIndex(xPtr, activeIdxPtr, activeCountPtr unsafe.Pointer, n int) {
	C.tw_k_relu_and_index((*C.float)(xPtr), (*C.int)(activeIdxPtr), (*C.int)(activeCountPtr), C.int(n))
}

// SparseKernelsLoaded returns true if the sparse FFN kernels are available.
func SparseKernelsLoaded() bool {
	return C.tw_sparse_kernels_loaded() == 1
}

// KSparseMatMul computes out = WT_sparse @ x using only active columns (FP32).
func KSparseMatMul(outPtr, wtPtr, xPtr, activeIdxPtr unsafe.Pointer, activeCount, rows, cols int) {
	C.tw_k_sparse_matmul(
		(*C.float)(outPtr), (*C.float)(wtPtr), (*C.float)(xPtr),
		(*C.int)(activeIdxPtr), C.int(activeCount),
		C.int(rows), C.int(cols),
	)
}

// KReLUAndIndexFP16 applies ReLU in-place on FP16 data AND builds active index.
func KReLUAndIndexFP16(xPtr, activeIdxPtr, activeCountPtr unsafe.Pointer, n int) {
	C.tw_k_relu_and_index_fp16(xPtr, (*C.int)(activeIdxPtr), (*C.int)(activeCountPtr), C.int(n))
}

// SparseFP16KernelsLoaded returns true if FP16 sparse kernels are available.
func SparseFP16KernelsLoaded() bool {
	return C.tw_sparse_fp16_kernels_loaded() == 1
}

// KSparseMatMulFP16 computes sparse matmul with FP16 weights and activations.
// FP32 accumulation, FP16 output. Half the memory bandwidth of FP32 sparse.
func KSparseMatMulFP16(outPtr, wtPtr, xPtr, activeIdxPtr unsafe.Pointer, activeCount, rows, cols int) {
	C.tw_k_sparse_matmul_fp16(outPtr, wtPtr, xPtr,
		(*C.int)(activeIdxPtr), C.int(activeCount),
		C.int(rows), C.int(cols),
	)
}

// TrainKernelsLoaded returns true if SiLU, RoPE, RMSNorm, and AdamW kernels are available.
func TrainKernelsLoaded() bool {
	return C.tw_train_kernels_loaded() == 1
}

// KSiLUGateMul: out[i] = silu(gate[i]) * up[i]. Fused kernel, all GPU.
func KSiLUGateMul(gatePtr, upPtr, outPtr unsafe.Pointer, n int) {
	C.tw_k_silu_gate_mul((*C.float)(gatePtr), (*C.float)(upPtr), (*C.float)(outPtr), C.int(n))
}

// KSiLUGateBackward: compute dGate and dUp from dOut, gate, up. All GPU.
func KSiLUGateBackward(dOutPtr, gatePtr, upPtr, dGatePtr, dUpPtr unsafe.Pointer, n int) {
	C.tw_k_silu_gate_backward((*C.float)(dOutPtr), (*C.float)(gatePtr), (*C.float)(upPtr),
		(*C.float)(dGatePtr), (*C.float)(dUpPtr), C.int(n))
}

// KRoPE: apply rotary position embeddings in-place on GPU.
func KRoPE(xPtr, cosPtr, sinPtr unsafe.Pointer, seqLen, dim, headDim, nHeads int) {
	C.tw_k_rope((*C.float)(xPtr), (*C.float)(cosPtr), (*C.float)(sinPtr),
		C.int(seqLen), C.int(dim), C.int(headDim), C.int(nHeads))
}

// KRoPEBackward: RoPE backward pass (negate sin).
func KRoPEBackward(dxPtr, cosPtr, sinPtr unsafe.Pointer, seqLen, dim, headDim, nHeads int) {
	C.tw_k_rope_backward((*C.float)(dxPtr), (*C.float)(cosPtr), (*C.float)(sinPtr),
		C.int(seqLen), C.int(dim), C.int(headDim), C.int(nHeads))
}

// KScaleOut: out[i] = x[i] * alpha on GPU.
func KScaleOut(xPtr, outPtr unsafe.Pointer, alpha float32, n int) {
	C.tw_k_scale_out((*C.float)(xPtr), (*C.float)(outPtr), C.float(alpha), C.int(n))
}

// KEmbedGather2: out[pos] = embed[token[pos]], no position embedding.
func KEmbedGather2(outPtr, embedPtr, tokensPtr unsafe.Pointer, seqLen, dim int) {
	C.tw_k_embed_gather2((*C.float)(outPtr), (*C.float)(embedPtr), (*C.int)(tokensPtr),
		C.int(seqLen), C.int(dim))
}

// KRMSNormOutSave: out = rmsnorm(input, weight), saves per-row scale for backward.
func KRMSNormOutSave(inputPtr, outPtr, weightPtr, scalesPtr unsafe.Pointer, seqLen, dim int) {
	C.tw_k_rmsnorm_save((*C.float)(inputPtr), (*C.float)(outPtr), (*C.float)(weightPtr),
		(*C.float)(scalesPtr), C.int(seqLen), C.int(dim))
}

// KRMSNormBackward: GPU RMSNorm backward. dx = f(dOut, xIn, weight, scale).
func KRMSNormBackward(dOutPtr, xInPtr, weightPtr, scalesPtr, dxPtr unsafe.Pointer, seqLen, dim int) {
	C.tw_k_rmsnorm_bwd((*C.float)(dOutPtr), (*C.float)(xInPtr), (*C.float)(weightPtr),
		(*C.float)(scalesPtr), (*C.float)(dxPtr), C.int(seqLen), C.int(dim))
}

// KDecodeAttention: single-query attention against full KV cache. GQA-aware.
// Q[1,dim], K_cache[cacheLen,kvDim], V_cache[cacheLen,kvDim], out[1,dim].
func KDecodeAttention(qPtr, kCachePtr, vCachePtr, outPtr unsafe.Pointer,
	cacheLen, dim, kvDim, numHeads, numKVHeads int) {
	C.tw_k_decode_attn((*C.float)(qPtr), (*C.float)(kCachePtr), (*C.float)(vCachePtr),
		(*C.float)(outPtr), C.int(cacheLen), C.int(dim), C.int(kvDim),
		C.int(numHeads), C.int(numKVHeads))
}

// KCausalAttentionGQA: GQA-aware causal attention on GPU.
// Q[seqLen,dim], K[seqLen,kvDim], V[seqLen,kvDim], out[seqLen,dim].
func KCausalAttentionGQA(qPtr, kPtr, vPtr, outPtr unsafe.Pointer, seqLen, dim, kvDim, numHeads, numKVHeads int) {
	C.tw_k_attn_gqa((*C.float)(qPtr), (*C.float)(kPtr), (*C.float)(vPtr), (*C.float)(outPtr),
		C.int(seqLen), C.int(dim), C.int(kvDim), C.int(numHeads), C.int(numKVHeads))
}

// AttnBackwardLoaded returns true if the attention backward kernel is available.
func AttnBackwardLoaded() bool {
	return C.tw_attn_backward_loaded() == 1
}

// KCausalAttentionBackward: compute dQ, dK, dV from dOut. GQA-aware.
// Q[seqLen,dim], K[seqLen,kvDim], V[seqLen,kvDim], dOut[seqLen,dim]
// dQ[seqLen,dim], dK[seqLen,kvDim], dV[seqLen,kvDim] — zeroed and filled.
func KCausalAttentionBackward(qPtr, kPtr, vPtr, dOutPtr, dQPtr, dKPtr, dVPtr unsafe.Pointer,
	seqLen, dim, kvDim, numHeads, numKVHeads int) {
	C.tw_k_attn_backward(
		(*C.float)(qPtr), (*C.float)(kPtr), (*C.float)(vPtr), (*C.float)(dOutPtr),
		(*C.float)(dQPtr), (*C.float)(dKPtr), (*C.float)(dVPtr),
		C.int(seqLen), C.int(dim), C.int(kvDim), C.int(numHeads), C.int(numKVHeads))
}

// KCrossEntropy: fused cross-entropy loss + dHidden gradient. All GPU.
func KCrossEntropy(hiddenPtr, embedWPtr unsafe.Pointer, D, vocabSize int,
	targetsPtr, lossesPtr, dHiddenPtr unsafe.Pointer, invN float32, nPos int) {
	C.tw_k_cross_entropy((*C.float)(hiddenPtr), (*C.float)(embedWPtr),
		C.int(D), C.int(vocabSize), (*C.int)(targetsPtr),
		(*C.float)(lossesPtr), (*C.float)(dHiddenPtr), C.float(invN), C.int(nPos))
}

// KSoftmaxCE: fused softmax + cross-entropy + gradient on pre-computed logits.
// logits[nPos,vocabSize] modified in-place (becomes exp values).
// grad[nPos,vocabSize] = (softmax - one_hot) * invN.
// losses[nPos] = -log(softmax[target]).
// Zero PCIe — everything stays on GPU. Fallback when Xe is unavailable.
func KSoftmaxCE(logitsPtr, targetsPtr, lossesPtr, gradPtr unsafe.Pointer,
	nPos, vocabSize int, invN float32) {
	C.tw_k_softmax_ce((*C.float)(logitsPtr), (*C.int)(targetsPtr),
		(*C.float)(lossesPtr), (*C.float)(gradPtr),
		C.int(nPos), C.int(vocabSize), C.float(invN))
}

// SoftmaxCELoaded returns true if the tiled softmax+CE kernel is available.
func SoftmaxCELoaded() bool {
	return C.tw_softmax_ce_loaded() == 1
}

// DequantKernelsLoaded returns true if INT8 dequantization kernels are available.
func DequantKernelsLoaded() bool {
	return C.tw_dequant_kernels_loaded() == 1
}

// KDequantInt8ToFP16: dequantize INT8 weights to FP16 on GPU.
// data[rows,cols] INT8 + scales[rows] FP32 → out[rows,cols] FP16.
func KDequantInt8ToFP16(dataPtr, scalesPtr, outPtr unsafe.Pointer, rows, cols int) {
	C.tw_k_dequant_int8_fp16(dataPtr, (*C.float)(scalesPtr), outPtr, C.int(rows), C.int(cols))
}

// KDequantInt8ToFP32: dequantize INT8 weights to FP32 on GPU.
// data[rows,cols] INT8 + scales[rows] FP32 → out[rows,cols] FP32.
func KDequantInt8ToFP32(dataPtr, scalesPtr, outPtr unsafe.Pointer, rows, cols int) {
	C.tw_k_dequant_int8_fp32(dataPtr, (*C.float)(scalesPtr), (*C.float)(outPtr), C.int(rows), C.int(cols))
}

// KFP32ToFP16 converts FP32 tensor to FP16 on GPU. For mixed-precision matmul input prep.
func KFP32ToFP16(inPtr, outPtr unsafe.Pointer, n int) {
	C.tw_k_fp32_to_fp16((*C.float)(inPtr), outPtr, C.int(n))
}

// KFP16ToFP32 converts FP16 tensor to FP32 on GPU.
func KFP16ToFP32(inPtr, outPtr unsafe.Pointer, n int) {
	C.tw_k_fp16_to_fp32(inPtr, (*C.float)(outPtr), C.int(n))
}
