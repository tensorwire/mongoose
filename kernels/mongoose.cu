// mongoose CUDA kernels — atomized into per-concern files.
// Build: nvcc -shared -o libmongoose_kernels.so mongoose.cu -Xcompiler -fPIC

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <stdint.h>
#include <math.h>

extern "C" {

#include "mongoose_adamw.cu"
#include "mongoose_attention.cu"
#include "mongoose_elementwise.cu"
#include "mongoose_helix.cu"
#include "mongoose_kv.cu"
#include "mongoose_loss.cu"
#include "mongoose_q8_matvec.cu"
#include "mongoose_quant.cu"
#include "mongoose_rmsnorm.cu"
#include "mongoose_rope.cu"
#include "mongoose_silu.cu"
#include "mongoose_sparse.cu"
#include "mongoose_sq4_matvec.cu"
#include "mongoose_mlp_train.cu"

} // extern "C"

// Fused MLP kernel uses cooperative_groups (C++ templates) — must be outside extern "C"
#include "mongoose_mlp_fused.cu"
