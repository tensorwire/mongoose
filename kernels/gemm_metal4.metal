// Metal 4 GEMM kernels using matmul2d TensorOp with cooperative_tensor output.
// Pre-compiled to .metallib.
//
// Pattern from Liu Liu's example_matmul_metal4: tensor_inline inputs,
// cooperative_tensor accumulation, store() to tensor_inline output.
//
// Threadgroup grid: (ceil(N/BN), ceil(M/BM), 1), threads: (simdWidth*4, 1, 1).

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr int BM = 64;
constant constexpr int BN = 32;

// --- C = A @ B^T ---
kernel void gemm4_bt(
    device half* A_buf [[buffer(0)]],
    device half* B_buf [[buffer(1)]],
    device half* C_buf [[buffer(2)]],
    constant uint& M  [[buffer(3)]],
    constant uint& K  [[buffer(4)]],
    constant uint& N  [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    auto A = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        A_buf, dextents<int32_t, 2>(K, M));
    auto B = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        B_buf, dextents<int32_t, 2>(K, N));
    auto C = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        C_buf, dextents<int32_t, 2>(N, M));

    auto mA = A.slice(0, tgid.y * BM);
    auto mB = B.slice(0, tgid.x * BN);
    auto mC = C.slice(tgid.x * BN, tgid.y * BM);

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, static_cast<int>(dynamic_extent),
        false, true, false);
    matmul2d<desc, execution_simdgroups<4>> matmulOp;

    auto cT = matmulOp.get_destination_cooperative_tensor<
        decltype(mA), decltype(mB), half>();

    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i))
            cT[i] = 0;
    }

    matmulOp.run(mA, mB, cT);
    cT.store(mC);
}

// --- C = A @ B ---
kernel void gemm4_nn(
    device half* A_buf [[buffer(0)]],
    device half* B_buf [[buffer(1)]],
    device half* C_buf [[buffer(2)]],
    constant uint& M  [[buffer(3)]],
    constant uint& K  [[buffer(4)]],
    constant uint& N  [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    auto A = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        A_buf, dextents<int32_t, 2>(K, M));
    auto B = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        B_buf, dextents<int32_t, 2>(N, K));
    auto C = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        C_buf, dextents<int32_t, 2>(N, M));

    auto mA = A.slice(0, tgid.y * BM);
    auto mB = B.slice(tgid.x * BN, 0);
    auto mC = C.slice(tgid.x * BN, tgid.y * BM);

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, static_cast<int>(dynamic_extent),
        false, false, false);
    matmul2d<desc, execution_simdgroups<4>> matmulOp;

    auto cT = matmulOp.get_destination_cooperative_tensor<
        decltype(mA), decltype(mB), half>();

    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i))
            cT[i] = 0;
    }

    matmulOp.run(mA, mB, cT);
    cT.store(mC);
}

// ============================================================
// FP32 variants — read float, accumulate float, write float.
// No FP16 conversion. For GradMatMul/MatMulNN/DequantMatMulBwd
// where inputs are already FP32.
// ============================================================

// --- C = A @ B^T (FP32) ---
kernel void gemm4f_bt(
    device float* A_buf [[buffer(0)]],
    device float* B_buf [[buffer(1)]],
    device float* C_buf [[buffer(2)]],
    constant uint& M  [[buffer(3)]],
    constant uint& K  [[buffer(4)]],
    constant uint& N  [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    auto A = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        A_buf, dextents<int32_t, 2>(K, M));
    auto B = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        B_buf, dextents<int32_t, 2>(K, N));
    auto C = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        C_buf, dextents<int32_t, 2>(N, M));

    auto mA = A.slice(0, tgid.y * BM);
    auto mB = B.slice(0, tgid.x * BN);
    auto mC = C.slice(tgid.x * BN, tgid.y * BM);

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, static_cast<int>(dynamic_extent),
        false, true, false);
    matmul2d<desc, execution_simdgroups<4>> matmulOp;

    auto cT = matmulOp.get_destination_cooperative_tensor<
        decltype(mA), decltype(mB), float>();

    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i))
            cT[i] = 0;
    }

    matmulOp.run(mA, mB, cT);
    cT.store(mC);
}

// --- C = A @ B (FP32) ---
kernel void gemm4f_nn(
    device float* A_buf [[buffer(0)]],
    device float* B_buf [[buffer(1)]],
    device float* C_buf [[buffer(2)]],
    constant uint& M  [[buffer(3)]],
    constant uint& K  [[buffer(4)]],
    constant uint& N  [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    auto A = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        A_buf, dextents<int32_t, 2>(K, M));
    auto B = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        B_buf, dextents<int32_t, 2>(N, K));
    auto C = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        C_buf, dextents<int32_t, 2>(N, M));

    auto mA = A.slice(0, tgid.y * BM);
    auto mB = B.slice(tgid.x * BN, 0);
    auto mC = C.slice(tgid.x * BN, tgid.y * BM);

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, static_cast<int>(dynamic_extent),
        false, false, false);
    matmul2d<desc, execution_simdgroups<4>> matmulOp;

    auto cT = matmulOp.get_destination_cooperative_tensor<
        decltype(mA), decltype(mB), float>();

    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i))
            cT[i] = 0;
    }

    matmulOp.run(mA, mB, cT);
    cT.store(mC);
}

// --- C = A^T @ B (FP32) ---
kernel void gemm4f_tn(
    device float* A_buf [[buffer(0)]],
    device float* B_buf [[buffer(1)]],
    device float* C_buf [[buffer(2)]],
    constant uint& M  [[buffer(3)]],
    constant uint& K  [[buffer(4)]],
    constant uint& N  [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    // TN: C[K,N] = A[M,K]^T @ B[M,N] in row-major
    // Col-major: A stored as [K,M], transposed → [M,K]; B stored as [N,M]; C stored as [N,K]
    auto A = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        A_buf, dextents<int32_t, 2>(K, M));
    auto B = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        B_buf, dextents<int32_t, 2>(N, M));
    auto C = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        C_buf, dextents<int32_t, 2>(N, K));

    auto mA = A.slice(tgid.y * BM, 0);
    auto mB = B.slice(tgid.x * BN, 0);
    auto mC = C.slice(tgid.x * BN, tgid.y * BM);

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, static_cast<int>(dynamic_extent),
        true, false, false);
    matmul2d<desc, execution_simdgroups<4>> matmulOp;

    auto cT = matmulOp.get_destination_cooperative_tensor<
        decltype(mA), decltype(mB), float>();

    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i))
            cT[i] = 0;
    }

    matmulOp.run(mA, mB, cT);
    cT.store(mC);
}

// ============================================================
// FP16 variants (original)
// ============================================================

// --- C = A^T @ B ---
kernel void gemm4_tn(
    device half* A_buf [[buffer(0)]],
    device half* B_buf [[buffer(1)]],
    device half* C_buf [[buffer(2)]],
    constant uint& M  [[buffer(3)]],
    constant uint& K  [[buffer(4)]],
    constant uint& N  [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    // TN: C[K,N] = A[M,K]^T @ B[M,N] in row-major
    auto A = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        A_buf, dextents<int32_t, 2>(K, M));
    auto B = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        B_buf, dextents<int32_t, 2>(N, M));
    auto C = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        C_buf, dextents<int32_t, 2>(N, K));

    auto mA = A.slice(tgid.y * BM, 0);
    auto mB = B.slice(tgid.x * BN, 0);
    auto mC = C.slice(tgid.x * BN, tgid.y * BM);

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, static_cast<int>(dynamic_extent),
        true, false, false);
    matmul2d<desc, execution_simdgroups<4>> matmulOp;

    auto cT = matmulOp.get_destination_cooperative_tensor<
        decltype(mA), decltype(mB), half>();

    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i))
            cT[i] = 0;
    }

    matmulOp.run(mA, mB, cT);
    cT.store(mC);
}
