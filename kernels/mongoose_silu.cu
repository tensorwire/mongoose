// mongoose_silu.cu — silu kernels
// Auto-extracted from mongoose.cu. Do not edit directly.

// === Fused SiLU-Gate-Mul: out[i] = silu(gate[i]) * up[i] ===
// gate and up are [n], out is [n]. Eliminates 2 PCIe round-trips vs CPU.
__global__ void silu_gate_mul_kernel(const float* gate, const float* up, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        float sig = 1.0f / (1.0f + expf(-g));
        out[i] = g * sig * up[i];
    }
}

void mongoose_silu_gate_mul(const float* gate, const float* up, float* out, int n, cudaStream_t stream) {
    silu_gate_mul_kernel<<<(n+255)/256, 256, 0, stream>>>(gate, up, out, n);
}

// === SiLU-Gate-Mul backward: dGate and dUp from dOut ===
// dUp[i] = dOut[i] * silu(gate[i])
// dGate[i] = dOut[i] * up[i] * (sig + gate[i]*sig*(1-sig))
// where sig = sigmoid(gate[i])
__global__ void silu_gate_backward_kernel(
    const float* dOut, const float* gate, const float* up,
    float* dGate, float* dUp, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        float sig = 1.0f / (1.0f + expf(-g));
        float silu_g = g * sig;
        dUp[i] = dOut[i] * silu_g;
        dGate[i] = dOut[i] * up[i] * (sig + silu_g * (1.0f - sig));
    }
}

void mongoose_silu_gate_backward(
    const float* dOut, const float* gate, const float* up,
    float* dGate, float* dUp, int n, cudaStream_t stream
) {
    silu_gate_backward_kernel<<<(n+255)/256, 256, 0, stream>>>(dOut, gate, up, dGate, dUp, n);
}

__global__ void silu_gate_mul_fp16_kernel(const __half* gate, const __half* up, __half* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = __half2float(gate[i]);
        float sig = 1.0f / (1.0f + expf(-g));
        out[i] = __float2half(g * sig * __half2float(up[i]));
    }
}

void mongoose_silu_gate_mul_fp16(const void* gate, const void* up, void* out, int n, cudaStream_t stream) {
    silu_gate_mul_fp16_kernel<<<(n+255)/256, 256, 0, stream>>>(
        (const __half*)gate, (const __half*)up, (__half*)out, n);
}

__global__ void silu_gate_backward_fp16_kernel(
    const __half* dOut, const __half* gate, const __half* up,
    __half* dGate, __half* dUp, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = __half2float(gate[i]);
        float sig = 1.0f / (1.0f + expf(-g));
        float silu_g = g * sig;
        float dO = __half2float(dOut[i]);
        float u = __half2float(up[i]);
        dUp[i] = __float2half(dO * silu_g);
        dGate[i] = __float2half(dO * u * (sig + silu_g * (1.0f - sig)));
    }
}

void mongoose_silu_gate_backward_fp16(
    const void* dOut, const void* gate, const void* up,
    void* dGate, void* dUp, int n, cudaStream_t stream
) {
    silu_gate_backward_fp16_kernel<<<(n+255)/256, 256, 0, stream>>>(
        (const __half*)dOut, (const __half*)gate, (const __half*)up,
        (__half*)dGate, (__half*)dUp, n);
}

