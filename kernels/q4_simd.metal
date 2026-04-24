#include <metal_stdlib>
using namespace metal;

struct block_q4_0 { half d; uchar qs[16]; };

inline float block_q4_dot(device const block_q4_0* qb, float sumy,
                          thread const float* yl, int il) {
    float d = float(qb->d);
    device const ushort* qs = ((device const ushort*)(qb->qs)) + il/2;
    float a0=0, a1=0, a2=0, a3=0;
    for (int i = 0; i < 8; i += 2) {
        ushort v = qs[i/2];
        a0 += yl[i+0] * (v & 0x000F);
        a1 += yl[i+1] * (v & 0x0F00);
        a2 += yl[i+8] * (v & 0x00F0);
        a3 += yl[i+9] * (v & 0xF000);
    }
    return d * (sumy * -8.0f + a0 + a1 + a2 + a3);
}

// Q4_0 matvec with pre-computed row pointers and wider activation loads.
// NSG=2 simdgroups (64 threads), NR0=4 rows per simdgroup → 8 rows/tg.
// Matches llama.cpp mul_vec_q_n_f32_impl dispatch geometry.
kernel void q4_matvec_simd(
    device const float* act     [[buffer(0)]],
    device const uchar* weight  [[buffer(1)]],
    device float* out           [[buffer(2)]],
    device const uint* p_K      [[buffer(3)]],
    device const uint* p_N      [[buffer(4)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint K = p_K[0], N = p_N[0];
    const uint QK = 32;
    const short NR0 = 4;
    const short NQ = 16;
    const short NSG = 2;
    const uint nb = K / QK;
    const uint nb18 = nb * 18;

    const uint r0 = (tgpig.x * NSG + sgitg) * NR0;
    if (r0 >= N) return;

    device const block_q4_0* rows[NR0];
    for (short row = 0; row < NR0; row++) {
        uint r = min(r0 + (uint)row, N - 1);
        rows[row] = (device const block_q4_0*)(weight + r * nb18);
    }

    float sumf[NR0] = {0,0,0,0};

    const short ix = tiisg / 2;
    const short il = (tiisg % 2) * 8;
    const uint  ib0 = ix;

    device const float* yb = act + ib0 * QK + il;

    for (uint ib = ib0; ib < nb; ib += NQ) {
        float sumy0 = 0, sumy1 = 0;
        float yl[16];

        for (short i = 0; i < 8; i += 2) {
            float v0 = yb[i], v1 = yb[i+1];
            float v2 = yb[i+16], v3 = yb[i+17];
            sumy0 += v0 + v1;
            sumy1 += v2 + v3;
            yl[i+0] = v0;
            yl[i+1] = v1 / 256.0f;
            yl[i+8] = v2 / 16.0f;
            yl[i+9] = v3 / 4096.0f;
        }
        float sumy = sumy0 + sumy1;

        for (short row = 0; row < NR0; row++) {
            if (r0 + row < N)
                sumf[row] += block_q4_dot(rows[row] + ib, sumy, yl, il);
        }

        yb += QK * NQ;
    }

    for (short row = 0; row < NR0; row++) {
        float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && r0 + row < N)
            out[r0 + row] = tot;
    }
}
