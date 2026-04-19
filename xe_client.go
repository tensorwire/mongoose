//go:build linux

package mongoose

/*
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <linux/memfd.h>

// Create anonymous shared memory via memfd_create.
// Returns fd, or -1 on error. No filesystem footprint.
static int mongoose_memfd_create(int size) {
    int fd = syscall(SYS_memfd_create, "mongoose-xe-arena", MFD_CLOEXEC);
    if (fd < 0) return -1;
    if (ftruncate(fd, size) < 0) { close(fd); return -1; }
    return fd;
}

// Clear the close-on-exec flag so the fd survives exec into the daemon.
static void mongoose_clear_cloexec(int fd) {
    int flags = fcntl(fd, F_GETFD);
    fcntl(fd, F_SETFD, flags & ~1); // clear FD_CLOEXEC
}
*/
import "C"

import (
	"bufio"
	"fmt"
	"log"
	"net"
	"os"
	"os/exec"
	"os/signal"
	"syscall"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
	"unsafe"
)

// XeDaemon manages the xe-daemon child process and communicates via Unix socket.
// The daemon holds the Level Zero context in a C process (avoids Go runtime + IGC crash).
// Split shared memory arena enables zero-copy data exchange for bulk operations.
//
// Arena layout:
//   [0 .. 128MB)           GO REGION   — Go writes logits + targets
//   [128MB .. 128MB+4KB)   GUARD       — mprotect(PROT_NONE), segfaults on access
//   [128MB+4KB .. 256MB)   XE REGION   — Xe writes losses + gradients
const (
	ArenaSize    = 256 * 1024 * 1024
	ArenaHalf    = ArenaSize / 2
	ArenaGuard   = 4096
	ArenaXeStart = ArenaHalf + ArenaGuard
)

type XeDaemon struct {
	conn    net.Conn
	proc    *exec.Cmd
	sock    string
	name    string
	memSize uint64
	mu      sync.Mutex

	// Split shared arena — memfd backed, zero-copy
	arenaFd   int    // memfd file descriptor (Go owns lifetime)
	arenaData []byte // mmap'd region
}

// NewXeDaemon starts the xe-daemon and connects via Unix socket.
// Returns nil if xe-daemon binary not found or no Intel GPU.
func NewXeDaemon() *XeDaemon {
	// Find xe-daemon binary
	paths := []string{
		"./xe-daemon/xe-daemon",
		"./xe-daemon",
		// Look relative to the binary's directory
		filepath.Join(filepath.Dir(os.Args[0]), "xe-daemon", "xe-daemon"),
		"/usr/local/bin/xe-daemon",
	}
	var binPath string
	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			binPath = p
			break
		}
	}
	if binPath == "" {
		return nil
	}

	// Clean up stale sockets from previous runs
	stale, _ := filepath.Glob("/tmp/mongoose-xe-*.sock")
	for _, s := range stale {
		os.Remove(s)
	}

	// Create memfd for shared arena BEFORE spawning daemon
	arenaFd := int(C.mongoose_memfd_create(C.int(ArenaSize)))
	if arenaFd >= 0 {
		// Clear CLOEXEC so fd survives exec into daemon
		C.mongoose_clear_cloexec(C.int(arenaFd))
	}

	sock := fmt.Sprintf("/tmp/mongoose-xe-%d.sock", os.Getpid())
	var cmd *exec.Cmd
	if arenaFd >= 0 {
		cmd = exec.Command(binPath, sock, fmt.Sprintf("%d", arenaFd))
		cmd.ExtraFiles = []*os.File{os.NewFile(uintptr(arenaFd), "arena")}
		// ExtraFiles[0] becomes fd 3 in the child. But we passed the actual fd number
		// as argv[2]. Since we cleared CLOEXEC, the original fd number survives exec.
	} else {
		cmd = exec.Command(binPath, sock)
	}
	cmd.Stderr = os.Stderr

	// Capture stdout for READY line
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil
	}

	if err := cmd.Start(); err != nil {
		return nil
	}

	// Wait for READY
	scanner := bufio.NewScanner(stdout)
	var name string
	var memSize uint64
	deadline := time.After(5 * time.Second)
	readyCh := make(chan bool, 1)
	go func() {
		if scanner.Scan() {
			line := scanner.Text()
			if strings.HasPrefix(line, "READY ") {
				parts := strings.SplitN(line[6:], " ", 2)
				name = parts[0]
				if len(parts) > 1 {
					// Name might have spaces — last token is mem size
					fields := strings.Fields(line[6:])
					if len(fields) >= 2 {
						name = strings.Join(fields[:len(fields)-1], " ")
						memSize, _ = strconv.ParseUint(fields[len(fields)-1], 10, 64)
					}
				}
				readyCh <- true
				return
			}
		}
		readyCh <- false
	}()

	select {
	case ok := <-readyCh:
		if !ok {
			cmd.Process.Kill()
			return nil
		}
	case <-deadline:
		cmd.Process.Kill()
		return nil
	}

	// Connect
	conn, err := net.Dial("unix", sock)
	if err != nil {
		cmd.Process.Kill()
		return nil
	}

	d := &XeDaemon{
		conn:    conn,
		proc:    cmd,
		sock:    sock,
		name:    name,
		memSize: memSize,
	}

	// Map the memfd arena in Go's address space
	if arenaFd >= 0 {
		data, err := syscall.Mmap(arenaFd, 0, ArenaSize,
			syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
		if err != nil {
			log.Printf("[xe] arena mmap failed: %v", err)
		} else {
			d.arenaFd = arenaFd
			d.arenaData = data
			log.Printf("[xe] arena: %d MB zero-copy (go: 0-%dMB, xe: %dMB-%dMB)",
				ArenaSize/(1024*1024), ArenaHalf/(1024*1024),
				ArenaXeStart/(1024*1024), ArenaSize/(1024*1024))
		}
	}

	log.Printf("[xe] daemon connected: %s (%d MB shared)", name, memSize/1024/1024)

	// Ensure cleanup on process exit (signal or normal)
	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
		<-sigCh
		d.Close()
		os.Exit(0)
	}()

	return d
}

// cmd sends a command and reads the response.
func (d *XeDaemon) cmd(format string, args ...interface{}) string {
	d.mu.Lock()
	defer d.mu.Unlock()
	msg := fmt.Sprintf(format, args...) + "\n"
	d.conn.Write([]byte(msg))
	buf := make([]byte, 4096)
	n, _ := d.conn.Read(buf)
	return strings.TrimSpace(string(buf[:n]))
}

// Close shuts down the daemon, unmaps the arena, and cleans up.
func (d *XeDaemon) Close() {
	if d.conn == nil {
		return
	}
	// Unmap arena before killing daemon
	if d.arenaData != nil {
		syscall.Munmap(d.arenaData)
		d.arenaData = nil
	}
	if d.arenaFd > 0 {
		syscall.Close(d.arenaFd)
		d.arenaFd = 0
	}
	d.cmd("quit")
	d.conn.Close()
	d.conn = nil
	// Kill if quit didn't work
	if d.proc != nil && d.proc.Process != nil {
		d.proc.Process.Kill()
		d.proc.Wait()
	}
	os.Remove(d.sock)
}

// Name returns the Xe device name.
func (d *XeDaemon) Name() string { return "xe/" + d.name }

// VRAM returns shared memory size.
func (d *XeDaemon) VRAM() uint64 { return d.memSize }

// LoadSPIRV loads a SPIR-V kernel from file. Returns kernel index.
func (d *XeDaemon) LoadSPIRV(path, entryPoint string) int {
	resp := d.cmd("load %s %s", path, entryPoint)
	if strings.HasPrefix(resp, "OK ") {
		idx, _ := strconv.Atoi(resp[3:])
		return idx
	}
	return -1
}

// Alloc allocates shared memory on the Xe device. Returns pointer.
func (d *XeDaemon) Alloc(bytes int) unsafe.Pointer {
	resp := d.cmd("alloc %d", bytes)
	if strings.HasPrefix(resp, "OK ") {
		addr, _ := strconv.ParseUint(strings.TrimPrefix(resp[3:], "0x"), 16, 64)
		return unsafe.Pointer(uintptr(addr))
	}
	return nil
}

// Free releases Xe shared memory.
func (d *XeDaemon) Free(ptr unsafe.Pointer) {
	d.cmd("free %p", ptr)
}

// DispatchRMSNorm dispatches RMSNorm on Xe GPU.
func (d *XeDaemon) DispatchRMSNorm(kernelIdx int, x, out, weight unsafe.Pointer, dim, seqLen int, eps float32) {
	d.cmd("rmsnorm %d %p %p %p %d %d %f", kernelIdx, x, out, weight, dim, seqLen, eps)
}

// DispatchSiLU dispatches fused SiLU-gate-mul on Xe.
func (d *XeDaemon) DispatchSiLU(kernelIdx int, gate, up, out unsafe.Pointer, n int) {
	d.cmd("silu %d %p %p %p %d", kernelIdx, gate, up, out, n)
}

// DispatchAdd dispatches element-wise add on Xe.
func (d *XeDaemon) DispatchAdd(kernelIdx int, a, b unsafe.Pointer, n int) {
	d.cmd("add %d %p %p %d", kernelIdx, a, b, n)
}

// Sync waits for all Xe GPU ops to complete.
func (d *XeDaemon) Sync() {
	d.cmd("sync")
}

// HasArena returns true if zero-copy shared memory is available.
func (d *XeDaemon) HasArena() bool {
	return d.arenaData != nil
}

// GoRegion returns a float32 slice in the Go write region (below ArenaHalf).
// Panics if the requested range exceeds the Go region.
func (d *XeDaemon) GoRegion(byteOffset, count int) []float32 {
	end := byteOffset + count*4
	if end > ArenaHalf {
		panic(fmt.Sprintf("xe arena: Go write at %d+%d exceeds Go region (%d)", byteOffset, count*4, ArenaHalf))
	}
	return (*[1 << 28]float32)(unsafe.Pointer(&d.arenaData[byteOffset]))[:count:count]
}

// GoRegionInt32 returns an int32 slice in the Go write region.
func (d *XeDaemon) GoRegionInt32(byteOffset, count int) []int32 {
	end := byteOffset + count*4
	if end > ArenaHalf {
		panic(fmt.Sprintf("xe arena: Go write at %d+%d exceeds Go region (%d)", byteOffset, count*4, ArenaHalf))
	}
	return (*[1 << 28]int32)(unsafe.Pointer(&d.arenaData[byteOffset]))[:count:count]
}

// XeRegion returns a float32 slice in the Xe write region (above ArenaXeStart).
// Go reads from here after Sync().
func (d *XeDaemon) XeRegion(byteOffset, count int) []float32 {
	if byteOffset < ArenaXeStart {
		panic(fmt.Sprintf("xe arena: Xe read at %d below Xe region (%d)", byteOffset, ArenaXeStart))
	}
	return (*[1 << 28]float32)(unsafe.Pointer(&d.arenaData[byteOffset]))[:count:count]
}

// DispatchCrossEntropy runs fused softmax + cross-entropy + gradient on Xe GPU.
//
// Protocol (memory ownership):
//   1. Go writes logits + targets to Go region  (Go owns arena)
//   2. Go calls DispatchCrossEntropy             (handoff → Xe owns arena)
//   3. Xe reads logits/targets, writes losses/grad to Xe region
//   4. Go calls Sync()                           (handoff → Go owns arena)
//   5. Go reads losses + grad from Xe region
//
// Socket dispatch + sync are the memory barriers. Zero data copies.
func (d *XeDaemon) DispatchCrossEntropy(kernelIdx int,
	logitsOff, targetsOff, lossesOff, gradOff uint32,
	nPos, vocabSize uint32, invN float32) {
	d.cmd("crossentropy %d %d %d %d %d %d %d %f",
		kernelIdx, logitsOff, targetsOff, lossesOff, gradOff,
		nPos, vocabSize, invN)
}
