//go:build darwin && cgo

package mongoose

import (
	"math"
	"sort"
	"testing"
	"unsafe"
)

// testSQ4Pack quantizes FP32 weights to spec-format SQ4:
// separate magnitude (3-bit bitstream) and sign (1-bit packed) arrays.
func testSQ4Pack(data []float32, rows, cols int) (mag []byte, sign []byte, bands [8]float32, outlierIdx []uint32, outlierVal []float32) {
	n := rows * cols
	magBits := n * 3
	mag = make([]byte, (magBits+7)/8)
	sign = make([]byte, (n+7)/8)

	absVals := make([]float32, n)
	for i, v := range data {
		if v < 0 {
			absVals[i] = -v
		} else {
			absVals[i] = v
		}
	}
	sorted := make([]float32, n)
	copy(sorted, absVals)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })

	var boundaries [8]float32
	for b := 0; b < 8; b++ {
		lo := n * b / 8
		hi := n * (b + 1) / 8
		if hi > n {
			hi = n
		}
		boundaries[b] = sorted[hi-1]
		var sum float64
		for _, v := range sorted[lo:hi] {
			sum += float64(v)
		}
		bands[b] = float32(sum / float64(hi-lo))
	}

	outlierPos := n * 999 / 1000
	if outlierPos >= n-1 {
		outlierPos = n - 2
	}
	if outlierPos < 0 {
		outlierPos = 0
	}
	outlierThresh := sorted[outlierPos]

	for i := 0; i < n; i++ {
		v := data[i]
		absV := absVals[i]
		if absV >= outlierThresh && outlierThresh > 0 {
			outlierIdx = append(outlierIdx, uint32(i))
			outlierVal = append(outlierVal, v)
		}

		band := 7
		for b := 0; b < 7; b++ {
			if absV <= boundaries[b] {
				band = b
				break
			}
		}

		bitPos := i * 3
		byteIdx := bitPos / 8
		bitOff := uint(bitPos % 8)
		mag[byteIdx] |= byte(band&0x07) << bitOff
		if bitOff > 5 {
			mag[byteIdx+1] |= byte(band&0x07) >> (8 - bitOff)
		}

		if v < 0 {
			sign[i/8] |= 1 << uint(i%8)
		}
	}
	return
}

// testSQ4Dequant converts spec-format SQ4 back to FP32 with outlier correction.
func testSQ4Dequant(mag []byte, sign []byte, bands [8]float32, rows, cols int, outlierIdx []uint32, outlierVal []float32) []float32 {
	n := rows * cols
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		bitPos := i * 3
		byteIdx := bitPos / 8
		bitOff := uint(bitPos % 8)
		raw := mag[byteIdx] >> bitOff
		if bitOff > 5 {
			raw |= mag[byteIdx+1] << (8 - bitOff)
		}
		band := raw & 0x07
		val := bands[band]
		if sign[i/8]&(1<<uint(i%8)) != 0 {
			val = -val
		}
		out[i] = val
	}
	for i, idx := range outlierIdx {
		if int(idx) < len(out) {
			out[idx] = outlierVal[i]
		}
	}
	return out
}

// packNibbles combines spec-format mag (3-bit bitstream) + sign (1-bit packed) into
// GPU-format nibbles: [sign:1|band:3], 2 per byte.
func packNibbles(mag, sign []byte, n int) []byte {
	packed := make([]byte, n/2)
	for i := 0; i < n; i++ {
		bitPos := i * 3
		byteIdx := bitPos / 8
		bitOff := uint(bitPos % 8)
		raw := mag[byteIdx] >> bitOff
		if bitOff > 5 && byteIdx+1 < len(mag) {
			raw |= mag[byteIdx+1] << (8 - bitOff)
		}
		band := raw & 0x07
		signBit := (sign[i/8] >> uint(i%8)) & 1
		nibble := (signBit << 3) | band
		shift := uint((i & 1) * 4)
		packed[i/2] |= nibble << shift
	}
	return packed
}

func initSQ4Metal(t *testing.T) *SQ4Metal {
	t.Helper()
	eng := NewMetal()
	if eng == nil {
		t.Skip("Metal not available")
	}
	sq4 := NewSQ4Metal(eng)
	if sq4 == nil {
		t.Skip("sq4_matvec.metallib not found")
	}
	return sq4
}

func TestSQ4Metal_Matvec(t *testing.T) {
	sq4m := initSQ4Metal(t)
	eng := NewMetal()
	te := AsTensorEngine(eng)

	rows, cols := 4, 32
	fp32 := make([]float32, rows*cols)
	for i := range fp32 {
		fp32[i] = float32(i%17)*0.1 - 0.8
	}

	mag, sign, bands, outlierIdx, outlierVal := testSQ4Pack(fp32, rows, cols)
	dequanted := testSQ4Dequant(mag, sign, bands, rows, cols, outlierIdx, outlierVal)

	act := make([]float32, cols)
	for i := range act {
		act[i] = float32(i)*0.05 - 0.5
	}

	cpuOut := make([]float32, rows)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			cpuOut[r] += dequanted[r*cols+c] * act[c]
		}
	}

	packed := packNibbles(mag, sign, rows*cols)
	packedBuf := eng.AllocRaw(len(packed), len(packed), []int{len(packed)})
	eng.UploadRaw(packedBuf, unsafe.Pointer(&packed[0]), len(packed))
	bandsBuf := te.FromHost(bands[:], []int{8})
	actBuf := te.FromHost(act, []int{cols})
	outBuf := te.Zeros([]int{rows})

	sq4m.Matvec(actBuf, packedBuf, bandsBuf, outBuf, rows, cols)

	gpuOut := te.ToHost(outBuf)
	t.Logf("CPU: [%.4f, %.4f, %.4f, %.4f]", cpuOut[0], cpuOut[1], cpuOut[2], cpuOut[3])
	t.Logf("GPU: [%.4f, %.4f, %.4f, %.4f]", gpuOut[0], gpuOut[1], gpuOut[2], gpuOut[3])

	for r := 0; r < rows; r++ {
		diff := math.Abs(float64(gpuOut[r] - cpuOut[r]))
		if diff > 0.05 {
			t.Errorf("row %d: GPU=%.6f CPU=%.6f diff=%.6f", r, gpuOut[r], cpuOut[r], diff)
		}
	}
}

func TestSQ4Metal_OutlierCorrect(t *testing.T) {
	sq4m := initSQ4Metal(t)
	eng := NewMetal()
	te := AsTensorEngine(eng)

	rows, cols := 4, 32
	fp32 := make([]float32, rows*cols)
	for i := range fp32 {
		fp32[i] = float32(i%17)*0.1 - 0.8
	}
	fp32[5] = 100.0
	fp32[cols+10] = -50.0

	mag, sign, bands, outlierIdx, outlierVal := testSQ4Pack(fp32, rows, cols)
	dequanted := testSQ4Dequant(mag, sign, bands, rows, cols, outlierIdx, outlierVal)

	act := make([]float32, cols)
	for i := range act {
		act[i] = float32(i)*0.05 - 0.5
	}

	cpuOut := make([]float32, rows)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			cpuOut[r] += dequanted[r*cols+c] * act[c]
		}
	}

	packed := packNibbles(mag, sign, rows*cols)
	packedBuf := eng.AllocRaw(len(packed), len(packed), []int{len(packed)})
	eng.UploadRaw(packedBuf, unsafe.Pointer(&packed[0]), len(packed))
	bandsBuf := te.FromHost(bands[:], []int{8})
	actBuf := te.FromHost(act, []int{cols})
	outBuf := te.Zeros([]int{rows})

	sq4m.Matvec(actBuf, packedBuf, bandsBuf, outBuf, rows, cols)

	if len(outlierIdx) > 0 {
		idxBuf := eng.AllocRaw(len(outlierIdx)*4, len(outlierIdx), []int{len(outlierIdx)})
		eng.UploadRaw(idxBuf, unsafe.Pointer(&outlierIdx[0]), len(outlierIdx)*4)
		valBuf := te.FromHost(outlierVal, []int{len(outlierVal)})

		sq4m.OutlierCorrect(idxBuf, valBuf, packedBuf, bandsBuf, actBuf, outBuf,
			len(outlierIdx), cols)
		t.Logf("Applied %d outlier corrections", len(outlierIdx))
	}

	gpuOut := te.ToHost(outBuf)
	t.Logf("CPU: [%.4f, %.4f, %.4f, %.4f]", cpuOut[0], cpuOut[1], cpuOut[2], cpuOut[3])
	t.Logf("GPU: [%.4f, %.4f, %.4f, %.4f]", gpuOut[0], gpuOut[1], gpuOut[2], gpuOut[3])

	for r := 0; r < rows; r++ {
		diff := math.Abs(float64(gpuOut[r] - cpuOut[r]))
		if diff > 0.1 {
			t.Errorf("row %d: GPU=%.6f CPU=%.6f diff=%.6f", r, gpuOut[r], cpuOut[r], diff)
		}
	}
}

func TestSQ4Metal_MatvecFused(t *testing.T) {
	sq4m := initSQ4Metal(t)
	eng := NewMetal()
	te := AsTensorEngine(eng)

	rows, cols := 4, 32
	fp32 := make([]float32, rows*cols)
	for i := range fp32 {
		fp32[i] = float32(i%17)*0.1 - 0.8
	}
	fp32[5] = 100.0
	fp32[cols+10] = -50.0

	mag, sign, bands, outlierIdx, outlierVal := testSQ4Pack(fp32, rows, cols)
	dequanted := testSQ4Dequant(mag, sign, bands, rows, cols, outlierIdx, outlierVal)

	act := make([]float32, cols)
	for i := range act {
		act[i] = float32(i)*0.05 - 0.5
	}

	cpuOut := make([]float32, rows)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			cpuOut[r] += dequanted[r*cols+c] * act[c]
		}
	}

	packed := packNibbles(mag, sign, rows*cols)
	packedBuf := eng.AllocRaw(len(packed), len(packed), []int{len(packed)})
	eng.UploadRaw(packedBuf, unsafe.Pointer(&packed[0]), len(packed))
	bandsBuf := te.FromHost(bands[:], []int{8})
	actBuf := te.FromHost(act, []int{cols})
	outBuf := te.Zeros([]int{rows})

	idxBuf := eng.AllocRaw(len(outlierIdx)*4, len(outlierIdx), []int{len(outlierIdx)})
	eng.UploadRaw(idxBuf, unsafe.Pointer(&outlierIdx[0]), len(outlierIdx)*4)
	valBuf := te.FromHost(outlierVal, []int{len(outlierVal)})

	sq4m.MatvecFused(actBuf, packedBuf, bandsBuf, outBuf, rows, cols,
		idxBuf, valBuf, len(outlierIdx))

	gpuOut := te.ToHost(outBuf)
	t.Logf("Fused: CPU: [%.4f, %.4f, %.4f, %.4f]", cpuOut[0], cpuOut[1], cpuOut[2], cpuOut[3])
	t.Logf("Fused: GPU: [%.4f, %.4f, %.4f, %.4f]", gpuOut[0], gpuOut[1], gpuOut[2], gpuOut[3])

	for r := 0; r < rows; r++ {
		diff := math.Abs(float64(gpuOut[r] - cpuOut[r]))
		if diff > 0.1 {
			t.Errorf("fused row %d: GPU=%.6f CPU=%.6f diff=%.6f", r, gpuOut[r], cpuOut[r], diff)
		}
	}
}

func TestSQ4Metal_Matvec_LargeDim(t *testing.T) {
	sq4m := initSQ4Metal(t)
	eng := NewMetal()
	te := AsTensorEngine(eng)

	rows, cols := 896, 896
	fp32 := make([]float32, rows*cols)
	for i := range fp32 {
		fp32[i] = float32(i%31)*0.03 - 0.5
	}

	mag, sign, bands, outlierIdx, outlierVal := testSQ4Pack(fp32, rows, cols)
	dequanted := testSQ4Dequant(mag, sign, bands, rows, cols, outlierIdx, outlierVal)

	act := make([]float32, cols)
	for i := range act {
		act[i] = float32(i%13)*0.07 - 0.4
	}

	cpuOut := make([]float32, rows)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			cpuOut[r] += dequanted[r*cols+c] * act[c]
		}
	}

	packed := packNibbles(mag, sign, rows*cols)
	packedBuf := eng.AllocRaw(len(packed), len(packed), []int{len(packed)})
	eng.UploadRaw(packedBuf, unsafe.Pointer(&packed[0]), len(packed))
	bandsBuf := te.FromHost(bands[:], []int{8})
	actBuf := te.FromHost(act, []int{cols})
	outBuf := te.Zeros([]int{rows})

	sq4m.Matvec(actBuf, packedBuf, bandsBuf, outBuf, rows, cols)

	gpuOut := te.ToHost(outBuf)

	maxDiff := float64(0)
	for r := 0; r < rows; r++ {
		diff := math.Abs(float64(gpuOut[r] - cpuOut[r]))
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	t.Logf("LargeDim [%d,%d]: max diff=%.6f", rows, cols, maxDiff)
	t.Logf("Sample: CPU[0]=%.6f GPU[0]=%.6f  CPU[100]=%.6f GPU[100]=%.6f",
		cpuOut[0], gpuOut[0], cpuOut[100], gpuOut[100])

	if maxDiff > 0.1 {
		t.Errorf("max diff %.6f exceeds tolerance 0.1", maxDiff)
	}
}
