
//go:build darwin && cgo

package mongoose

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Foundation -framework Metal

int mtl_q8_gemm_init(void);
int mtl_q8_gemm_available(void);
int mtl_q8_quantize(const float* src, unsigned char* dst, int N, int K);
int mtl_q8_gemm_bt(const float* A, const unsigned char* W, float* C, int M, int K, int N);
*/
import "C"
import _ "unsafe"
import (
	"fmt"
	"unsafe"
)

// Q8BlockSize is the number of weight elements sharing one quantization scale.
// Matches GGUF's block_q8_0 and the existing block_q4_0 in the inference path.
const Q8BlockSize = 32

// Q8BlockBytes is the on-disk/in-memory size of one block: half scale + 32 int8.
const Q8BlockBytes = 2 + Q8BlockSize

// Q8Available reports whether the block-GEMM kernels loaded.
//
// The kernels borrow metal_impl_darwin.m's device and queue, so Metal must be
// initialized first; NewMetal() is idempotent and does that.
func Q8Available() bool {
	if NewMetal() == nil {
		return false
	}
	(C.mtl_q8_gemm_init)()
	return (C.mtl_q8_gemm_available)() == 1
}

// Q8QuantizedBytes returns the buffer size needed for an [n,k] weight matrix.
func Q8QuantizedBytes(n, k int) int {
	return n * (k / Q8BlockSize) * Q8BlockBytes
}

// Q8Quantize converts an [n,k] FP32 weight matrix to block_q8_0.
//
// Scales are per 32-element block rather than per row: block scales survive the
// wider dynamic range of a backward pass, where a per-row absmax dominated by a
// single outlier flattens small early-layer gradients toward zero.
//
// k must be a multiple of Q8BlockSize.
func Q8Quantize(src []float32, n, k int) ([]byte, error) {
	if k%Q8BlockSize != 0 {
		return nil, fmt.Errorf("q8: k must be a multiple of %d, got %d", Q8BlockSize, k)
	}
	if len(src) != n*k {
		return nil, fmt.Errorf("q8: src has %d elements, want n*k = %d", len(src), n*k)
	}
	if !Q8Available() {
		return nil, fmt.Errorf("q8: Metal block-GEMM kernels unavailable")
	}

	dst := make([]byte, Q8QuantizedBytes(n, k))
	rc := (C.mtl_q8_quantize)(
		(*C.float)(unsafe.Pointer(&src[0])),
		(*C.uchar)(unsafe.Pointer(&dst[0])),
		C.int(n), C.int(k))
	if rc != 0 {
		return nil, fmt.Errorf("q8: quantize failed (%d)", int(rc))
	}
	return dst, nil
}

// Q8MaxK is the largest reduction depth a single kernel launch can handle.
//
// matmul2d.run() overwrites rather than accumulates and cooperative tensors have
// no store_add(), so the whole reduction must happen in one run() — which means
// the full BN x K weight tile has to be resident in threadgroup memory. With
// BN=32 FP32 that caps K at 256. Q8GemmBTSplitK covers deeper reductions.
const Q8MaxK = 256

// Q8GemmBTSplitK computes C[m,n] = A[m,k] @ dequant(W)[n,k]^T for any k,
// splitting the reduction into Q8MaxK-deep chunks and summing the partials.
//
// Granite needs this: its reductions are dim (2560 on the 3B, 4096 on the 8B)
// and ffnDim (8192, 12800), all far past the single-launch ceiling.
//
// Partial sums accumulate in FP32 on the host, so the numerics match a single
// deep pass. k must be a multiple of Q8BlockSize; each chunk is block-aligned.
func Q8GemmBTSplitK(a []float32, w []byte, m, k, n int) ([]float32, error) {
	if k <= Q8MaxK {
		return Q8GemmBT(a, w, m, k, n)
	}
	if k%Q8BlockSize != 0 {
		return nil, fmt.Errorf("q8: k must be a multiple of %d, got %d", Q8BlockSize, k)
	}
	if len(a) != m*k {
		return nil, fmt.Errorf("q8: A has %d elements, want m*k = %d", len(a), m*k)
	}
	if want := Q8QuantizedBytes(n, k); len(w) != want {
		return nil, fmt.Errorf("q8: W has %d bytes, want %d", len(w), want)
	}

	// Chunk on a block boundary so each slice stays self-describing.
	chunk := Q8MaxK - (Q8MaxK % Q8BlockSize)
	blocksPerRow := k / Q8BlockSize

	out := make([]float32, m*n)
	aChunk := make([]float32, m*chunk)
	wChunk := make([]byte, 0, n*(chunk/Q8BlockSize)*Q8BlockBytes)

	for k0 := 0; k0 < k; k0 += chunk {
		ck := chunk
		if k0+ck > k {
			ck = k - k0
		}

		// Gather A's column slice (row-major, so this is a strided copy).
		aSlice := aChunk[:m*ck]
		for r := 0; r < m; r++ {
			copy(aSlice[r*ck:(r+1)*ck], a[r*k+k0:r*k+k0+ck])
		}

		// Gather W's block-column slice.
		blkStart := k0 / Q8BlockSize
		blkCount := ck / Q8BlockSize
		wSlice := wChunk[:0]
		for r := 0; r < n; r++ {
			off := (r*blocksPerRow + blkStart) * Q8BlockBytes
			wSlice = append(wSlice, w[off:off+blkCount*Q8BlockBytes]...)
		}

		part, err := Q8GemmBT(aSlice, wSlice, m, ck, n)
		if err != nil {
			return nil, fmt.Errorf("q8: split-k chunk at %d: %w", k0, err)
		}
		for i := range out {
			out[i] += part[i]
		}
	}
	return out, nil
}

// Q8GemmBT computes C[m,n] = A[m,k] @ dequant(W)[n,k]^T in a single launch.
//
// k must not exceed Q8MaxK; use Q8GemmBTSplitK for deeper reductions.
//
// This is the training-shaped GEMM (m = batch x seqlen), distinct from the
// decode-shaped q8_matvec in the inference pipeline. The accumulator is FP32
// even though the multiply operands are FP16, because FP16 accumulation over a
// multi-thousand-deep reduction discards the small gradient contributions that
// block scaling exists to preserve.
//
// k must be a multiple of Q8BlockSize.
func Q8GemmBT(a []float32, w []byte, m, k, n int) ([]float32, error) {
	if k%Q8BlockSize != 0 {
		return nil, fmt.Errorf("q8: k must be a multiple of %d, got %d", Q8BlockSize, k)
	}
	if len(a) != m*k {
		return nil, fmt.Errorf("q8: A has %d elements, want m*k = %d", len(a), m*k)
	}
	if want := Q8QuantizedBytes(n, k); len(w) != want {
		return nil, fmt.Errorf("q8: W has %d bytes, want %d", len(w), want)
	}
	if k > Q8MaxK {
		return nil, fmt.Errorf("q8: k=%d exceeds the single-launch limit %d; use Q8GemmBTSplitK",
			k, Q8MaxK)
	}
	if !Q8Available() {
		return nil, fmt.Errorf("q8: Metal block-GEMM kernels unavailable")
	}

	c := make([]float32, m*n)
	rc := (C.mtl_q8_gemm_bt)(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.uchar)(unsafe.Pointer(&w[0])),
		(*C.float)(unsafe.Pointer(&c[0])),
		C.int(m), C.int(k), C.int(n))
	if rc != 0 {
		return nil, fmt.Errorf("q8: gemm failed (%d)", int(rc))
	}
	return c, nil
}
