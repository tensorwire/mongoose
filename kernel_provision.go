//go:build linux && cgo

package mongoose

import (
	_ "embed"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

//go:embed kernels/mongoose.cu
var embeddedKernelSource []byte

// ProvisionKernels attempts to compile CUDA kernels if no prebuilt .so is found.
// Returns the path to the compiled .so, or empty string on failure.
//
// Resolution order:
//  1. Detect GPU compute capability
//  2. Find nvcc in PATH
//  3. If nvcc missing, install nvidia-cuda-toolkit via apt (Ubuntu only)
//  4. Write embedded .cu source to cache dir
//  5. Compile with nvcc for detected arch
//  6. Return path to compiled .so
func ProvisionKernels() string {
	cc := detectComputeCapability()
	if cc == "" {
		log.Printf("[kernels] could not detect GPU compute capability")
		return ""
	}

	cacheDir := kernelCacheDir()
	soPath := filepath.Join(cacheDir, fmt.Sprintf("libmongoose_kernels_sm%s.so", strings.Replace(cc, ".", "", 1)))

	if _, err := os.Stat(soPath); err == nil {
		log.Printf("[kernels] cached kernel found: %s", soPath)
		return soPath
	}

	nvcc := findNVCC()
	if nvcc == "" {
		log.Printf("[kernels] nvcc not found — attempting install")
		if !installNVCC() {
			log.Printf("[kernels] failed to install nvcc. Install manually: sudo apt install nvidia-cuda-toolkit")
			return ""
		}
		nvcc = findNVCC()
		if nvcc == "" {
			log.Printf("[kernels] nvcc still not found after install")
			return ""
		}
	}

	os.MkdirAll(cacheDir, 0755)

	cuPath := filepath.Join(cacheDir, "mongoose.cu")
	if err := os.WriteFile(cuPath, embeddedKernelSource, 0644); err != nil {
		log.Printf("[kernels] failed to write kernel source: %v", err)
		return ""
	}

	arch := fmt.Sprintf("compute_%s", strings.Replace(cc, ".", "", 1))
	log.Printf("[kernels] compiling for %s (compute capability %s)...", arch, cc)

	cmd := exec.Command(nvcc,
		"-shared",
		"-o", soPath,
		cuPath,
		"-Xcompiler", "-fPIC",
		"-gencode", fmt.Sprintf("arch=%s,code=%s", arch, arch),
	)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		log.Printf("[kernels] nvcc compile failed: %v", err)
		os.Remove(soPath)
		return ""
	}

	log.Printf("[kernels] compiled: %s", soPath)
	return soPath
}

func detectComputeCapability() string {
	out, err := exec.Command("nvidia-smi",
		"--query-gpu=compute_cap",
		"--format=csv,noheader").Output()
	if err != nil {
		return ""
	}
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	if len(lines) == 0 {
		return ""
	}
	return strings.TrimSpace(lines[0])
}

func findNVCC() string {
	paths := []string{
		"/usr/local/cuda/bin/nvcc",
		"/usr/bin/nvcc",
	}
	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	if p, err := exec.LookPath("nvcc"); err == nil {
		return p
	}
	return ""
}

func installNVCC() bool {
	if _, err := os.Stat("/etc/debian_version"); err != nil {
		log.Printf("[kernels] auto-install only supported on Ubuntu/Debian")
		return false
	}

	log.Printf("[kernels] running: sudo apt-get install -y nvidia-cuda-toolkit")
	cmd := exec.Command("sudo", "apt-get", "install", "-y", "nvidia-cuda-toolkit")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run() == nil
}

func kernelCacheDir() string {
	if dir := os.Getenv("AI_KERNEL_CACHE"); dir != "" {
		return dir
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".ai", "kernels")
}
