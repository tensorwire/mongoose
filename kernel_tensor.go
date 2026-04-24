package mongoose

import "unsafe"

// Tensor-based wrappers for inference kernel dispatch.
// These avoid requiring callers to import "unsafe" — DevicePtr() extraction
// happens here, in the package that already owns it.

func ptrAdd(p unsafe.Pointer, bytes int) unsafe.Pointer {
	return unsafe.Add(p, bytes)
}

// Q8Weight holds quantized weight data on the GPU for fused matvec dispatch.
type Q8Weight struct {
	Data   *Tensor // int8 data packed into float32 slots (or Q4 packed bytes)
	Scales *Tensor // per-row float32 scales [rows]
	Rows   int
	Cols   int
	Q4     bool // true = 4-bit quantized
}

// TRMSNorm applies RMSNorm in-place: x = rmsnorm(x, weight).
func TRMSNorm(x, weight *Tensor, seqLen, dim int) {
	KRMSNorm(x.DevicePtr(), weight.DevicePtr(), seqLen, dim)
}

// TRMSNormOut computes out = rmsnorm(input, weight). Input is not modified.
func TRMSNormOut(input, out, weight *Tensor, seqLen, dim int) {
	KRMSNormOut(input.DevicePtr(), out.DevicePtr(), weight.DevicePtr(), seqLen, dim)
}

// TQ8Matvec dispatches fused dequant+matvec: out = act @ dequant(w).
// Selects Q8 or Q4 kernel based on w.Q4.
func TQ8Matvec(act *Tensor, w Q8Weight, out *Tensor) {
	if w.Q4 {
		KQ4Matvec(act.DevicePtr(), w.Data.DevicePtr(), w.Scales.DevicePtr(), out.DevicePtr(), w.Rows, w.Cols)
	} else {
		KQ8Matvec(act.DevicePtr(), w.Data.DevicePtr(), w.Scales.DevicePtr(), out.DevicePtr(), w.Rows, w.Cols)
	}
}

// TRoPE applies rotary position embeddings in-place.
func TRoPE(x, cos, sin *Tensor, seqLen, dim, headDim, nHeads int) {
	KRoPE(x.DevicePtr(), cos.DevicePtr(), sin.DevicePtr(), seqLen, dim, headDim, nHeads)
}

// TDecodeAttention computes single-query GQA decode attention against the KV cache.
// Q[1,dim], kCache[cacheLen,kvDim], vCache[cacheLen,kvDim], out[1,dim].
func TDecodeAttention(q, kCache, vCache, out *Tensor, cacheLen, dim, kvDim, numHeads, numKVHeads int) {
	KDecodeAttention(q.DevicePtr(), kCache.DevicePtr(), vCache.DevicePtr(), out.DevicePtr(),
		cacheLen, dim, kvDim, numHeads, numKVHeads)
}

// TSiLUGateMul computes out[i] = silu(gate[i]) * up[i].
func TSiLUGateMul(gate, up, out *Tensor, n int) {
	KSiLUGateMul(gate.DevicePtr(), up.DevicePtr(), out.DevicePtr(), n)
}

// TCopy copies bytes from src to dst on the device.
func TCopy(dst, src *Tensor, bytes int) {
	KCopy(dst.DevicePtr(), src.DevicePtr(), bytes)
}

// TCopyAt copies kvDim floats from src into dst at byte offset dstByteOffset.
func TCopyAt(dst *Tensor, dstByteOffset int, src *Tensor, bytes int) {
	dstPtr := dst.DevicePtr()
	KCopy(ptrAdd(dstPtr, dstByteOffset), src.DevicePtr(), bytes)
}

// TAddInPlace computes a += b element-wise.
func TAddInPlace(a, b *Tensor, n int) {
	KAddInPlace(a.DevicePtr(), b.DevicePtr(), n)
}

// TRoPEAt applies RoPE using cos/sin at a specific byte offset within the tables.
func TRoPEAt(x *Tensor, cosTable, sinTable *Tensor, byteOffset int, seqLen, dim, headDim, nHeads int) {
	KRoPE(x.DevicePtr(), ptrAdd(cosTable.DevicePtr(), byteOffset), ptrAdd(sinTable.DevicePtr(), byteOffset), seqLen, dim, headDim, nHeads)
}

// TDecodeAttentionKV computes decode attention where KV caches are raw tensors.
func TDecodeAttentionKV(q, kCache, vCache, out *Tensor, cacheLen, dim, kvDim, numHeads, numKVHeads int) {
	KDecodeAttention(q.DevicePtr(), kCache.DevicePtr(), vCache.DevicePtr(), out.DevicePtr(),
		cacheLen, dim, kvDim, numHeads, numKVHeads)
}

// TGradScale multiplies all elements by scale.
func TGradScale(t *Tensor, scale float32, n int) {
	KGradScale(t.DevicePtr(), scale, n)
}

// TZero zeroes nBytes of GPU memory in t.
func TZero(t *Tensor, nBytes int) {
	KZero(t.DevicePtr(), nBytes)
}

// PackInt8ToFloat32 reinterprets int8 data as float32 for GPU upload.
func PackInt8ToFloat32(q []int8) []float32 {
	nBytes := len(q)
	nFloats := (nBytes + 3) / 4
	buf := make([]float32, nFloats)
	src := unsafe.Slice((*byte)(unsafe.Pointer(&q[0])), nBytes)
	dst := unsafe.Slice((*byte)(unsafe.Pointer(&buf[0])), nBytes)
	copy(dst, src)
	return buf
}

// PackBytesToFloat32 reinterprets raw bytes as float32 for GPU upload.
func PackBytesToFloat32(b []byte) []float32 {
	nBytes := len(b)
	nFloats := (nBytes + 3) / 4
	buf := make([]float32, nFloats)
	dst := unsafe.Slice((*byte)(unsafe.Pointer(&buf[0])), nBytes)
	copy(dst, b)
	return buf
}
