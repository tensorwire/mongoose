// mongoose_mlp_train.cu — Fused MLP training mega kernel.
//
// Composes atomic kernel files into a single compilation unit for MLP training.
// Each sub-file provides individual __global__ kernels and extern "C" dispatch functions.
// This file adds the fused training step that chains them with cuBLAS matmul.
//
// Build: included by mongoose.cu (or compiled standalone with -lcublas -lcurand)
//
// Sub-kernels:
//   mongoose_bias_add.cu      — broadcast bias add + reduce_mean_rows (bias gradient)
//   mongoose_batchnorm.cu     — BatchNorm forward (with running stats) + backward
//   mongoose_bce_loss.cu      — BCEWithLogitsLoss + reduce_mean (scalar loss)
//   mongoose_dropout.cu       — Philox counter-based dropout (graph-safe) + backward

#include "mongoose_bias_add.cu"
#include "mongoose_batchnorm.cu"
#include "mongoose_bce_loss.cu"
#include "mongoose_dropout.cu"

// The fused graph dispatch will be added here once the Go-side
// MLPTrainGraph captures these kernels into a CUDA graph.
// For now, each kernel is individually callable via its mongoose_* entry point.
