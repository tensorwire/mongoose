NVCC ?= nvcc
# Generate PTX for sm_90 — driver JIT compiles for newer GPUs (Blackwell sm_120)
CUDA_ARCH ?= -gencode arch=compute_90,code=compute_90

libmongoose_kernels.so: mongoose.cu
	$(NVCC) -shared -o $@ $< -Xcompiler -fPIC $(CUDA_ARCH)

clean:
	rm -f libmongoose_kernels.so

.PHONY: clean
