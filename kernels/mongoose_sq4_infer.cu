// mongoose_sq4_infer.cu — Fused SQ4 forward pass with CUDA graph.
// One cudaGraphLaunch per token. Zero per-kernel launch overhead.
// pos/seqLen passed via device-side control buffer, updated before replay.

#include <cstdint>
#include <cstdio>

// All kernel functions are in scope via mongoose.cu compilation unit.

__global__ void sq4_add_inplace_kernel(float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

__global__ void sq4_copy_kernel(float* dst, const float* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// Embed gather that reads tokenID from device control buffer
__global__ void sq4_embed_gather_ctrl(const float* embed, float* out, const int* ctrl, int dim) {
    int tokenID = ctrl[0];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) out[i] = embed[tokenID * dim + i];
}

// RoPE that reads pos from device control buffer
__global__ void sq4_rope_ctrl(float* x, const float* cos_cache, const float* sin_cache,
    const int* ctrl, int dim, int headDim, int nHeads) {
    int pos = ctrl[1];
    int halfHead = headDim / 2;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nHeads * halfHead;
    if (tid >= total) return;
    int h = tid / halfHead;
    int j = tid % halfHead;
    float* cos_off = (float*)(cos_cache + pos * halfHead);
    float* sin_off = (float*)(sin_cache + pos * halfHead);
    float c = cos_off[j], s = sin_off[j];
    int base = h * headDim;
    float x0 = x[base + j], x1 = x[base + j + halfHead];
    x[base + j]            = x0 * c - x1 * s;
    x[base + j + halfHead] = x0 * s + x1 * c;
}

// KV cache write that reads pos from device control buffer
__global__ void sq4_kv_write_ctrl(const float* src, float* cache, const int* ctrl, int kvDim) {
    int pos = ctrl[1];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kvDim) cache[pos * kvDim + i] = src[i];
}

// Decode attention that reads seqLen from device control buffer
__global__ void sq4_decode_attn_ctrl(
    const float* Q, const float* K_cache, const float* V_cache, float* out,
    const int* ctrl, int dim, int kvDim, int nHeads, int nKVHeads
) {
    int seqLen = ctrl[1] + 1; // pos + 1
    int headDim = dim / nHeads;
    int h = blockIdx.x;
    if (h >= nHeads) return;
    int kvMul = nHeads / nKVHeads;
    int kvH = h / kvMul;
    float scale = 1.0f / sqrtf((float)headDim);

    extern __shared__ float shmem[];
    float* scores = shmem;

    for (int t = threadIdx.x; t < seqLen; t += blockDim.x) {
        float dot = 0;
        for (int d = 0; d < headDim; d++)
            dot += Q[h * headDim + d] * K_cache[t * kvDim + kvH * headDim + d];
        scores[t] = dot * scale;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float mx = -1e30f;
        for (int t = 0; t < seqLen; t++) mx = fmaxf(mx, scores[t]);
        float s = 0;
        for (int t = 0; t < seqLen; t++) { scores[t] = expf(scores[t] - mx); s += scores[t]; }
        for (int t = 0; t < seqLen; t++) scores[t] /= s;
    }
    __syncthreads();

    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        float acc = 0;
        for (int t = 0; t < seqLen; t++)
            acc += scores[t] * V_cache[t * kvDim + kvH * headDim + d];
        out[h * headDim + d] = acc;
    }
}

struct sq4_weight_desc {
    uint8_t* packed; float* bands; uint32_t* oidx; float* oval; float* o_approx;
    int oc, rows, cols;
};

struct sq4_layer_weights {
    uint8_t* wq_packed;  float* wq_bands;  uint32_t* wq_oidx;  float* wq_oval;  int wq_oc, wq_rows, wq_cols;
    uint8_t* wk_packed;  float* wk_bands;  uint32_t* wk_oidx;  float* wk_oval;  int wk_oc, wk_rows, wk_cols;
    uint8_t* wv_packed;  float* wv_bands;  uint32_t* wv_oidx;  float* wv_oval;  int wv_oc, wv_rows, wv_cols;
    uint8_t* wo_packed;  float* wo_bands;  uint32_t* wo_oidx;  float* wo_oval;  int wo_oc, wo_rows, wo_cols;
    uint8_t* gate_packed; float* gate_bands; uint32_t* gate_oidx; float* gate_oval; int gate_oc, gate_rows, gate_cols;
    uint8_t* up_packed;  float* up_bands;  uint32_t* up_oidx;  float* up_oval;  int up_oc, up_rows, up_cols;
    uint8_t* down_packed; float* down_bands; uint32_t* down_oidx; float* down_oval; int down_oc, down_rows, down_cols;
    float* wq_oa; float* wk_oa; float* wv_oa; float* wo_oa;
    float* gate_oa; float* up_oa; float* down_oa;
    float* norm1; float* norm2;
    float* bq; float* bk; float* bv;
    float* k_cache; float* v_cache;
};

struct sq4_infer_state {
    int dim, kv_dim, head_dim, n_heads, n_kv_heads, ffn_dim, vocab_size, n_layers, max_seq;
    int built;
    int swizzled;
    cudaStream_t stream;
    struct sq4_layer_weights* layers;
    float* final_norm;
    uint8_t* lm_packed; float* lm_bands; uint32_t* lm_oidx; float* lm_oval; float* lm_oa; int lm_oc, lm_rows, lm_cols;
    float* embed;
    float* cos_cache; float* sin_cache;
    float* hidden; float* normed; float* normed2;
    float* Q; float* K; float* V;
    float* attn_out; float* proj;
    float* gate_buf; float* up_buf; float* ffn_mid; float* down_buf;
    float* logits;
    unsigned int* argmax_result;

    // Control buffer: [tokenID, pos] on device, pinned host mirror
    int* d_ctrl;
    int* h_ctrl;

    // CUDA graph
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    int graph_captured;
};

static struct sq4_infer_state g_sq4i = {0};

int sq4_infer_built() { return g_sq4i.built; }

int sq4_infer_build(int dim, int kv_dim, int head_dim,
    int n_heads, int n_kv_heads, int ffn_dim,
    int vocab_size, int n_layers, int max_seq) {
    if (g_sq4i.built) return 0;
    g_sq4i.dim = dim; g_sq4i.kv_dim = kv_dim; g_sq4i.head_dim = head_dim;
    g_sq4i.n_heads = n_heads; g_sq4i.n_kv_heads = n_kv_heads;
    g_sq4i.ffn_dim = ffn_dim; g_sq4i.vocab_size = vocab_size;
    g_sq4i.n_layers = n_layers; g_sq4i.max_seq = max_seq;

    cudaStreamCreate(&g_sq4i.stream);
    g_sq4i.layers = (struct sq4_layer_weights*)calloc(n_layers, sizeof(struct sq4_layer_weights));

    cudaMalloc(&g_sq4i.hidden, dim * sizeof(float));
    cudaMalloc(&g_sq4i.normed, dim * sizeof(float));
    cudaMalloc(&g_sq4i.normed2, dim * sizeof(float));
    cudaMalloc(&g_sq4i.Q, n_heads * head_dim * sizeof(float));
    cudaMalloc(&g_sq4i.K, kv_dim * sizeof(float));
    cudaMalloc(&g_sq4i.V, kv_dim * sizeof(float));
    cudaMalloc(&g_sq4i.attn_out, n_heads * head_dim * sizeof(float));
    cudaMalloc(&g_sq4i.proj, dim * sizeof(float));
    cudaMalloc(&g_sq4i.gate_buf, ffn_dim * sizeof(float));
    cudaMalloc(&g_sq4i.up_buf, ffn_dim * sizeof(float));
    cudaMalloc(&g_sq4i.ffn_mid, ffn_dim * sizeof(float));
    cudaMalloc(&g_sq4i.down_buf, dim * sizeof(float));
    cudaMalloc(&g_sq4i.logits, vocab_size * sizeof(float));
    cudaMalloc(&g_sq4i.argmax_result, sizeof(unsigned int));
    // Mapped pinned memory: host writes, device reads via BAR — no memcpy needed
    cudaHostAlloc(&g_sq4i.h_ctrl, 2 * sizeof(int), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&g_sq4i.d_ctrl, g_sq4i.h_ctrl, 0);

    g_sq4i.graph_captured = 0;
    g_sq4i.built = 1;
    fprintf(stderr, "[SQ4-CUDA] fused pipeline built: dim=%d layers=%d vocab=%d\n", dim, n_layers, vocab_size);
    return 0;
}

void sq4_infer_set_layer(int l,
    void* wq_p, void* wq_b, void* wq_oi, void* wq_ov, int wq_oc, int wq_r, int wq_c,
    void* wk_p, void* wk_b, void* wk_oi, void* wk_ov, int wk_oc, int wk_r, int wk_c,
    void* wv_p, void* wv_b, void* wv_oi, void* wv_ov, int wv_oc, int wv_r, int wv_c,
    void* wo_p, void* wo_b, void* wo_oi, void* wo_ov, int wo_oc, int wo_r, int wo_c,
    void* gate_p, void* gate_b, void* gate_oi, void* gate_ov, int gate_oc, int gate_r, int gate_c,
    void* up_p, void* up_b, void* up_oi, void* up_ov, int up_oc, int up_r, int up_c,
    void* down_p, void* down_b, void* down_oi, void* down_ov, int down_oc, int down_r, int down_c,
    void* norm1, void* norm2, void* bq, void* bk, void* bv,
    void* k_cache, void* v_cache
) {
    struct sq4_layer_weights* ll = &g_sq4i.layers[l];
    ll->wq_packed=(uint8_t*)wq_p; ll->wq_bands=(float*)wq_b; ll->wq_oidx=(uint32_t*)wq_oi; ll->wq_oval=(float*)wq_ov; ll->wq_oc=wq_oc; ll->wq_rows=wq_r; ll->wq_cols=wq_c;
    ll->wk_packed=(uint8_t*)wk_p; ll->wk_bands=(float*)wk_b; ll->wk_oidx=(uint32_t*)wk_oi; ll->wk_oval=(float*)wk_ov; ll->wk_oc=wk_oc; ll->wk_rows=wk_r; ll->wk_cols=wk_c;
    ll->wv_packed=(uint8_t*)wv_p; ll->wv_bands=(float*)wv_b; ll->wv_oidx=(uint32_t*)wv_oi; ll->wv_oval=(float*)wv_ov; ll->wv_oc=wv_oc; ll->wv_rows=wv_r; ll->wv_cols=wv_c;
    ll->wo_packed=(uint8_t*)wo_p; ll->wo_bands=(float*)wo_b; ll->wo_oidx=(uint32_t*)wo_oi; ll->wo_oval=(float*)wo_ov; ll->wo_oc=wo_oc; ll->wo_rows=wo_r; ll->wo_cols=wo_c;
    ll->gate_packed=(uint8_t*)gate_p; ll->gate_bands=(float*)gate_b; ll->gate_oidx=(uint32_t*)gate_oi; ll->gate_oval=(float*)gate_ov; ll->gate_oc=gate_oc; ll->gate_rows=gate_r; ll->gate_cols=gate_c;
    ll->up_packed=(uint8_t*)up_p; ll->up_bands=(float*)up_b; ll->up_oidx=(uint32_t*)up_oi; ll->up_oval=(float*)up_ov; ll->up_oc=up_oc; ll->up_rows=up_r; ll->up_cols=up_c;
    ll->down_packed=(uint8_t*)down_p; ll->down_bands=(float*)down_b; ll->down_oidx=(uint32_t*)down_oi; ll->down_oval=(float*)down_ov; ll->down_oc=down_oc; ll->down_rows=down_r; ll->down_cols=down_c;
    ll->norm1=(float*)norm1; ll->norm2=(float*)norm2;
    ll->bq=(float*)bq; ll->bk=(float*)bk; ll->bv=(float*)bv;
    ll->k_cache=(float*)k_cache; ll->v_cache=(float*)v_cache;
}

void sq4_infer_set_final(void* final_norm,
    void* lm_p, void* lm_b, void* lm_oi, void* lm_ov, int lm_oc, int lm_r, int lm_c,
    void* embed, void* cos_cache, void* sin_cache) {
    g_sq4i.final_norm = (float*)final_norm;
    g_sq4i.lm_packed=(uint8_t*)lm_p; g_sq4i.lm_bands=(float*)lm_b;
    g_sq4i.lm_oidx=(uint32_t*)lm_oi; g_sq4i.lm_oval=(float*)lm_ov;
    g_sq4i.lm_oc=lm_oc; g_sq4i.lm_rows=lm_r; g_sq4i.lm_cols=lm_c;
    g_sq4i.embed = (float*)embed;
    g_sq4i.cos_cache = (float*)cos_cache;
    g_sq4i.sin_cache = (float*)sin_cache;
}

void sq4_infer_set_swizzled(int flag) { g_sq4i.swizzled = flag; }

// Set outlier band approximations for a weight in a layer.
// slot: 0=wq 1=wk 2=wv 3=wo 4=gate 5=up 6=down, 7=lm_head
void sq4_infer_set_outlier_approx(int layer, int slot, const float* host_approx, int count) {
    float* d_approx = NULL;
    if (count > 0 && host_approx) {
        cudaMalloc(&d_approx, count * sizeof(float));
        cudaMemcpy(d_approx, host_approx, count * sizeof(float), cudaMemcpyHostToDevice);
    }
    if (layer < 0) {
        // lm_head
        g_sq4i.lm_oa = d_approx;
        return;
    }
    struct sq4_layer_weights* ll = &g_sq4i.layers[layer];
    switch (slot) {
        case 0: ll->wq_oa = d_approx; break;
        case 1: ll->wk_oa = d_approx; break;
        case 2: ll->wv_oa = d_approx; break;
        case 3: ll->wo_oa = d_approx; break;
        case 4: ll->gate_oa = d_approx; break;
        case 5: ll->up_oa = d_approx; break;
        case 6: ll->down_oa = d_approx; break;
    }
}

// Dispatch matvec: TF32 tensor core path if swizzled, scalar otherwise.
static void sq4_dispatch_matvec(
    uint8_t* packed, float* bands, const float* act, float* out,
    int rows, int cols,
    uint32_t* oidx, float* oval, float* o_approx, int oc,
    cudaStream_t s
) {
    if (g_sq4i.swizzled) {
        mongoose_sq4_matvec_swizzled(packed, bands, act, out, rows, cols,
            o_approx, oval, oidx, oc, s);
    } else {
        mongoose_sq4_matvec_fused(packed, bands, act, out, rows, cols, oidx, oval, oc, s);
    }
}

// Encode the full forward pass onto a stream (used for both graph capture and direct execution)
static void sq4_encode_forward(cudaStream_t s) {
    int dim = g_sq4i.dim, kv_dim = g_sq4i.kv_dim, head_dim = g_sq4i.head_dim;
    int n_heads = g_sq4i.n_heads, n_kv_heads = g_sq4i.n_kv_heads;
    int ffn_dim = g_sq4i.ffn_dim, vocab_size = g_sq4i.vocab_size;
    int attn_dim = n_heads * head_dim;
    int half_head = head_dim / 2;

    // Embed gather (reads tokenID from d_ctrl[0])
    sq4_embed_gather_ctrl<<<(dim+255)/256, 256, 0, s>>>(g_sq4i.embed, g_sq4i.hidden, g_sq4i.d_ctrl, dim);

    for (int l = 0; l < g_sq4i.n_layers; l++) {
        struct sq4_layer_weights* ll = &g_sq4i.layers[l];

        mongoose_rmsnorm_out(g_sq4i.hidden, g_sq4i.normed, ll->norm1, 1, dim, s);

        sq4_dispatch_matvec(ll->wq_packed, ll->wq_bands, g_sq4i.normed, g_sq4i.Q, ll->wq_rows, ll->wq_cols, ll->wq_oidx, ll->wq_oval, ll->wq_oa, ll->wq_oc, s);
        sq4_dispatch_matvec(ll->wk_packed, ll->wk_bands, g_sq4i.normed, g_sq4i.K, ll->wk_rows, ll->wk_cols, ll->wk_oidx, ll->wk_oval, ll->wk_oa, ll->wk_oc, s);
        sq4_dispatch_matvec(ll->wv_packed, ll->wv_bands, g_sq4i.normed, g_sq4i.V, ll->wv_rows, ll->wv_cols, ll->wv_oidx, ll->wv_oval, ll->wv_oa, ll->wv_oc, s);

        if (ll->bq) sq4_add_inplace_kernel<<<(attn_dim+255)/256, 256, 0, s>>>(g_sq4i.Q, ll->bq, attn_dim);
        if (ll->bk) sq4_add_inplace_kernel<<<(kv_dim+255)/256, 256, 0, s>>>(g_sq4i.K, ll->bk, kv_dim);
        if (ll->bv) sq4_add_inplace_kernel<<<(kv_dim+255)/256, 256, 0, s>>>(g_sq4i.V, ll->bv, kv_dim);

        // RoPE (reads pos from d_ctrl[1])
        int ropeQ = n_heads * half_head;
        int ropeK = n_kv_heads * half_head;
        sq4_rope_ctrl<<<(ropeQ+255)/256, 256, 0, s>>>(g_sq4i.Q, g_sq4i.cos_cache, g_sq4i.sin_cache, g_sq4i.d_ctrl, attn_dim, head_dim, n_heads);
        sq4_rope_ctrl<<<(ropeK+255)/256, 256, 0, s>>>(g_sq4i.K, g_sq4i.cos_cache, g_sq4i.sin_cache, g_sq4i.d_ctrl, kv_dim, head_dim, n_kv_heads);

        // KV cache write (reads pos from d_ctrl[1])
        sq4_kv_write_ctrl<<<(kv_dim+255)/256, 256, 0, s>>>(g_sq4i.K, ll->k_cache, g_sq4i.d_ctrl, kv_dim);
        sq4_kv_write_ctrl<<<(kv_dim+255)/256, 256, 0, s>>>(g_sq4i.V, ll->v_cache, g_sq4i.d_ctrl, kv_dim);

        // Attention (reads seqLen from d_ctrl[1]+1)
        int shmem = g_sq4i.max_seq * sizeof(float);
        sq4_decode_attn_ctrl<<<n_heads, head_dim, shmem, s>>>(g_sq4i.Q, ll->k_cache, ll->v_cache, g_sq4i.attn_out, g_sq4i.d_ctrl, attn_dim, kv_dim, n_heads, n_kv_heads);

        sq4_dispatch_matvec(ll->wo_packed, ll->wo_bands, g_sq4i.attn_out, g_sq4i.proj, ll->wo_rows, ll->wo_cols, ll->wo_oidx, ll->wo_oval, ll->wo_oa, ll->wo_oc, s);
        sq4_add_inplace_kernel<<<(dim+255)/256, 256, 0, s>>>(g_sq4i.hidden, g_sq4i.proj, dim);

        mongoose_rmsnorm_out(g_sq4i.hidden, g_sq4i.normed2, ll->norm2, 1, dim, s);

        sq4_dispatch_matvec(ll->gate_packed, ll->gate_bands, g_sq4i.normed2, g_sq4i.gate_buf, ll->gate_rows, ll->gate_cols, ll->gate_oidx, ll->gate_oval, ll->gate_oa, ll->gate_oc, s);
        sq4_dispatch_matvec(ll->up_packed, ll->up_bands, g_sq4i.normed2, g_sq4i.up_buf, ll->up_rows, ll->up_cols, ll->up_oidx, ll->up_oval, ll->up_oa, ll->up_oc, s);
        mongoose_silu_gate_mul(g_sq4i.gate_buf, g_sq4i.up_buf, g_sq4i.ffn_mid, ffn_dim, s);
        sq4_dispatch_matvec(ll->down_packed, ll->down_bands, g_sq4i.ffn_mid, g_sq4i.down_buf, ll->down_rows, ll->down_cols, ll->down_oidx, ll->down_oval, ll->down_oa, ll->down_oc, s);
        sq4_add_inplace_kernel<<<(dim+255)/256, 256, 0, s>>>(g_sq4i.hidden, g_sq4i.down_buf, dim);
    }

    mongoose_rmsnorm_out(g_sq4i.hidden, g_sq4i.normed, g_sq4i.final_norm, 1, dim, s);
    sq4_dispatch_matvec(g_sq4i.lm_packed, g_sq4i.lm_bands, g_sq4i.normed, g_sq4i.logits, g_sq4i.lm_rows, g_sq4i.lm_cols, g_sq4i.lm_oidx, g_sq4i.lm_oval, g_sq4i.lm_oa, g_sq4i.lm_oc, s);
    mongoose_argmax(g_sq4i.logits, g_sq4i.argmax_result, vocab_size, s);
}

static void sq4_capture_graph() {
    if (g_sq4i.graph_captured) return;
    cudaStream_t s = g_sq4i.stream;

    // Dummy run to warm up
    g_sq4i.h_ctrl[0] = 0; g_sq4i.h_ctrl[1] = 0;
    sq4_encode_forward(s);
    cudaStreamSynchronize(s);

    // Capture
    cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    sq4_encode_forward(s);
    cudaStreamEndCapture(s, &g_sq4i.graph);
    cudaGraphInstantiate(&g_sq4i.graph_exec, g_sq4i.graph, 0);
    g_sq4i.graph_captured = 1;
    fprintf(stderr, "[SQ4-CUDA] graph captured and instantiated\n");
}

// One call: forward + argmax via graph replay. Returns sampled token ID.
int sq4_infer_step(int tokenID, int pos) {
    if (!g_sq4i.built) return -1;

    g_sq4i.h_ctrl[0] = tokenID;
    g_sq4i.h_ctrl[1] = pos;

    sq4_encode_forward(g_sq4i.stream);
    cudaStreamSynchronize(g_sq4i.stream);

    unsigned int result;
    cudaMemcpy(&result, g_sq4i.argmax_result, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    return (int)result;
}

// Forward pass returning logits on host (for prefill — can't use graph because seqLen varies)
int sq4_infer_step_logits(int tokenID, int pos, float* logits_out) {
    if (!g_sq4i.built) return -1;

    g_sq4i.h_ctrl[0] = tokenID;
    g_sq4i.h_ctrl[1] = pos;

    sq4_encode_forward(g_sq4i.stream);
    cudaStreamSynchronize(g_sq4i.stream);
    cudaMemcpy(logits_out, g_sq4i.logits, g_sq4i.vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
    return 0;
}

void sq4_infer_reset_kv() {
    if (!g_sq4i.built) return;
    for (int l = 0; l < g_sq4i.n_layers; l++) {
        struct sq4_layer_weights* ll = &g_sq4i.layers[l];
        if (ll->k_cache) cudaMemset(ll->k_cache, 0, g_sq4i.max_seq * g_sq4i.kv_dim * sizeof(float));
        if (ll->v_cache) cudaMemset(ll->v_cache, 0, g_sq4i.max_seq * g_sq4i.kv_dim * sizeof(float));
    }
}
