NVCC ?= nvcc
CUDA_ARCH ?= -gencode arch=compute_90,code=compute_90

# Atomic kernel files — mongoose.cu includes them all via #include
KERNEL_DEPS := mongoose_rmsnorm.cu mongoose_rope.cu mongoose_attention.cu \
	mongoose_silu.cu mongoose_elementwise.cu mongoose_adamw.cu mongoose_loss.cu \
	mongoose_q8_matvec.cu mongoose_quant.cu mongoose_kv.cu mongoose_helix.cu \
	mongoose_sparse.cu mongoose_mlp_fused.cu

libmongoose_kernels.so: mongoose.cu $(KERNEL_DEPS)
	$(NVCC) -shared -o $@ mongoose.cu -Xcompiler -fPIC -rdc=true $(CUDA_ARCH)

# Metal kernels (macOS only)
mlp_train.metallib: mlp_train.metal
	xcrun metal -std=metal3.1 -O2 -c mlp_train.metal -o mlp_train.air
	xcrun metallib -o mlp_train.metallib mlp_train.air
	rm -f mlp_train.air

sq4_matvec.metallib: sq4_matvec.metal
	xcrun metal -std=metal3.1 -O2 -c sq4_matvec.metal -o sq4_matvec.air
	xcrun metallib -o sq4_matvec.metallib sq4_matvec.air
	rm -f sq4_matvec.air

clean:
	rm -f libmongoose_kernels.so mlp_train.metallib mlp_train.air sq4_matvec.metallib sq4_matvec.air

.PHONY: clean
