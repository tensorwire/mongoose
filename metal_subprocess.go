//go:build darwin && cgo

package mongoose

import (
	"encoding/binary"
	"io"
	"log"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"
)

// MetalSubprocess implements Engine by shelling out to the mongoose-metal Swift binary.
// This is the fallback path when CGo Metal is not available.
// Kept for compatibility — the CGo Metal backend (metal_cgo.go) is preferred.
type MetalSubprocess struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout io.ReadCloser
	mu     sync.Mutex
}

// NewMetalSubprocess starts the mongoose-metal Swift binary and returns an engine.
// Returns nil if the binary is not found.
func NewMetalSubprocess() *MetalSubprocess {
	searchPaths := []string{
		"./mongoose-metal",
		"./metal/mongoose-metal",
		filepath.Join(os.Getenv("HOME"), "tensorwire/mongoose-metal"),
		"/usr/local/bin/mongoose-metal",
	}

	var binPath string
	for _, p := range searchPaths {
		if _, err := os.Stat(p); err == nil {
			binPath = p
			break
		}
	}
	if binPath == "" {
		return nil
	}

	cmd := exec.Command(binPath)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil
	}
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		log.Printf("WARN mongoose => Metal binary failed to start: %v", err)
		return nil
	}

	m := &MetalSubprocess{cmd: cmd, stdin: stdin, stdout: stdout}
	time.Sleep(100 * time.Millisecond)
	log.Printf("[mongoose] Metal subprocess started via %s", binPath)
	return m
}

func (m *MetalSubprocess) Name() string { return "metal-subprocess/Apple Silicon" }

func (m *MetalSubprocess) MatMul(a, b []float32, rows, k, n int) []float32 {
	m.mu.Lock()
	defer m.mu.Unlock()

	buf := make([]byte, 1+12+len(a)*4+len(b)*4)
	buf[0] = 0x01
	binary.LittleEndian.PutUint32(buf[1:], uint32(rows))
	binary.LittleEndian.PutUint32(buf[5:], uint32(k))
	binary.LittleEndian.PutUint32(buf[9:], uint32(n))
	off := 13
	for _, v := range a {
		binary.LittleEndian.PutUint32(buf[off:], math.Float32bits(v))
		off += 4
	}
	for _, v := range b {
		binary.LittleEndian.PutUint32(buf[off:], math.Float32bits(v))
		off += 4
	}
	m.stdin.Write(buf)

	sizeBuf := make([]byte, 4)
	io.ReadFull(m.stdout, sizeBuf)
	size := binary.LittleEndian.Uint32(sizeBuf)

	dataBuf := make([]byte, size)
	io.ReadFull(m.stdout, dataBuf)

	result := make([]float32, size/4)
	for i := range result {
		result[i] = math.Float32frombits(binary.LittleEndian.Uint32(dataBuf[i*4:]))
	}
	return result
}

func (m *MetalSubprocess) RMSNorm(x, weight []float32, eps float32) {
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

func (m *MetalSubprocess) SoftMax(x []float32, n int) {
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
	inv := 1.0 / sum
	for i := 0; i < n; i++ {
		x[i] *= inv
	}
}

func (m *MetalSubprocess) ReLU(x []float32) {
	for i := range x {
		if x[i] < 0 {
			x[i] = 0
		}
	}
}

func (m *MetalSubprocess) VRAM() uint64 { return 0 }

func (m *MetalSubprocess) Benchmark() float64 {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.stdin.Write([]byte{0x04})

	sizeBuf := make([]byte, 4)
	io.ReadFull(m.stdout, sizeBuf)
	size := binary.LittleEndian.Uint32(sizeBuf)

	dataBuf := make([]byte, size)
	io.ReadFull(m.stdout, dataBuf)

	gflops := math.Float32frombits(binary.LittleEndian.Uint32(dataBuf[:4]))
	return float64(gflops)
}

func (m *MetalSubprocess) Close() {
	m.stdin.Write([]byte{0xFF})
	m.cmd.Wait()
}
