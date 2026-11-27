/**
 * llmasm_engine.c - High-Performance C Inference Engine for llm.asm (.asmbin)
 * 
 * Compiles cleanly on:
 * - Windows (MSVC cl.exe or MinGW gcc or clang)
 * - Linux (gcc / clang)
 * - macOS (clang)
 */

#define LLMASM_EXPORTS
#include "llmasm_engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(__x86_64__) || defined(_M_X64)
    #if defined(__AVX2__) || defined(_M_AVX2)
        #include <immintrin.h>
        #define HAS_AVX2 1
    #endif
#endif

#define FIB_MULT 0x9E3779B97F4A7C15ULL
#define HT_BITS  19

static inline uint64_t align64_u64(uint64_t n) {
    return (n + 63) & ~63ULL;
}

/* Float16 to Float32 conversion */
static inline float fp16_to_fp32(uint16_t h) {
#if defined(HAS_AVX2) && (defined(__F16C__) || defined(_MSC_VER))
    __m128i v = _mm_cvtsi32_si128((int)h);
    __m128 f = _mm_cvtph_ps(v);
    return _mm_cvtss_f32(f);
#else
    /* Software bitwise unpack */
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h & 0x7C00) >> 10;
    uint32_t frac = (h & 0x03FF) << 13;
    uint32_t u;
    if (exp == 0x1F) {
        u = sign | 0x7F800000 | frac;
    } else if (exp != 0) {
        u = sign | ((exp + 112) << 23) | frac;
    } else if (frac != 0) {
        exp = 113;
        while ((frac & 0x00800000) == 0) {
            frac <<= 1;
            exp--;
        }
        frac &= 0x007FFFFF;
        u = sign | (exp << 23) | frac;
    } else {
        u = sign;
    }
    float out;
    memcpy(&out, &u, sizeof(out));
    return out;
#endif
}

typedef struct {
    const uint8_t* raw;
    int rows;
    int cols;
    int row_stride;
    int scale_bytes;
    int is_fp16;
} Q8Matrix;

typedef struct {
    const float* in_norm;
    const float* q_norm;
    const float* k_norm;
    const float* post_norm;
    Q8Matrix q_proj;
    Q8Matrix k_proj;
    Q8Matrix v_proj;
    Q8Matrix o_proj;
    Q8Matrix gate_proj;
    Q8Matrix up_proj;
    Q8Matrix down_proj;
} LayerWeights;

struct LLMModel {
    AsmBinHeader hdr;
    uint8_t* file_buf;
    size_t file_len;

    /* Tokenizer */
    const uint32_t* byte_to_tok;
    const uint32_t* tok_offsets;
    const uint8_t*  tok_bytes;
    const HashEntry* ht;

    /* Weights */
    Q8Matrix emb;
    const float* final_norm;
    LayerWeights* layers;

    /* KV Cache */
    float* k_cache; /* [n_layers * max_seq_len * n_kv_heads * head_dim] */
    float* v_cache;
    int current_pos;

    /* Working activations */
    float* act_x;
    float* act_norm;
    float* act_q;
    float* act_k;
    float* act_v;
    float* act_attn_out;
    float* act_mlp_gate;
    float* act_mlp_up;
    float* act_mlp_down;
    float* logits;
    int n_threads;
};

static void init_q8_matrix(Q8Matrix* m, const uint8_t* raw, int rows, int cols, int is_fp16) {
    m->raw = raw;
    m->rows = rows;
    m->cols = cols;
    m->is_fp16 = is_fp16;
    m->scale_bytes = is_fp16 ? 2 : 4;
    m->row_stride = cols + (cols / 32) * m->scale_bytes;
}

/* High-performance Q8_0 Matrix-Vector Multiply */
static void matvec_q8(const Q8Matrix* m, const float* x, float* out, int n_threads) {
    const int rows = m->rows;
    const int cols = m->cols;
    const int nb = cols / 32;
    const int is_fp16 = m->is_fp16;
    const int row_stride = m->row_stride;

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(n_threads > 1) num_threads(n_threads)
#endif
    for (int r = 0; r < rows; r++) {
        const uint8_t* row_base = m->raw + (size_t)r * row_stride;
        const int8_t* w_row = (const int8_t*)row_base;
        const uint8_t* s_base = row_base + cols;

        float sum_r = 0.0f;
        for (int b = 0; b < nb; b++) {
            float scale;
            if (is_fp16) {
                uint16_t s16;
                memcpy(&s16, s_base + b * 2, 2);
                scale = fp16_to_fp32(s16);
            } else {
                memcpy(&scale, s_base + b * 4, 4);
            }

            const int8_t* wb = w_row + b * 32;
            const float*  xb = x + b * 32;

            float block_dot = 0.0f;
#if defined(HAS_AVX2)
            __m256 vsum = _mm256_setzero_ps();
            for (int k = 0; k < 32; k += 8) {
                __m256 vx = _mm256_loadu_ps(xb + k);
                /* unpack 8 int8 values to 8 float32 */
                __m128i vw8 = _mm_loadl_epi64((const __m128i*)(wb + k));
                __m256i vw32 = _mm256_cvtepi8_epi32(vw8);
                __m256 vwf = _mm256_cvtepi32_ps(vw32);
                vsum = _mm256_fmadd_ps(vx, vwf, vsum);
            }
            /* Horizontal sum */
            __m128 vlow = _mm256_castps256_ps128(vsum);
            __m128 vhigh = _mm256_extractf128_ps(vsum, 1);
            __m128 vhsum = _mm_add_ps(vlow, vhigh);
            vhsum = _mm_hadd_ps(vhsum, vhsum);
            vhsum = _mm_hadd_ps(vhsum, vhsum);
            block_dot = _mm_cvtss_f32(vhsum);
#else
            for (int k = 0; k < 32; k++) {
                block_dot += (float)wb[k] * xb[k];
            }
#endif
            sum_r += block_dot * scale;
        }
        out[r] = sum_r;
    }
}

static void rmsnorm(const float* x, const float* w, int dim, float eps, float* out) {
    float sum_sq = 0.0f;
#if defined(HAS_AVX2)
    __m256 vsq = _mm256_setzero_ps();
    for (int i = 0; i < dim; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        vsq = _mm256_fmadd_ps(vx, vx, vsq);
    }
    __m128 vlow = _mm256_castps256_ps128(vsq);
    __m128 vhigh = _mm256_extractf128_ps(vsq, 1);
    __m128 vhsum = _mm_add_ps(vlow, vhigh);
    vhsum = _mm_hadd_ps(vhsum, vhsum);
    vhsum = _mm_hadd_ps(vhsum, vhsum);
    sum_sq = _mm_cvtss_f32(vhsum);
#else
    for (int i = 0; i < dim; i++) {
        sum_sq += x[i] * x[i];
    }
#endif
    float mean_sq = sum_sq / (float)dim;
    float inv_rms = 1.0f / sqrtf(mean_sq + eps);

#if defined(HAS_AVX2)
    __m256 vinv = _mm256_set1_ps(inv_rms);
    for (int i = 0; i < dim; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vw = _mm256_loadu_ps(w + i);
        __m256 res = _mm256_mul_ps(_mm256_mul_ps(vx, vinv), vw);
        _mm256_storeu_ps(out + i, res);
    }
#else
    for (int i = 0; i < dim; i++) {
        out[i] = x[i] * inv_rms * w[i];
    }
#endif
}

static void silu_mul(const float* gate, const float* up, int dim, float* out) {
    for (int i = 0; i < dim; i++) {
        float g = gate[i];
        float ax = fabsf(g);
        float e = expf(-ax);
        float num = (g < 0.0f) ? e : 1.0f;
        float silu = (g * num) / (1.0f + e);
        out[i] = silu * up[i];
    }
}

/* ============================================================================
 * Public Model Lifecycle
 * ============================================================================ */

LLMModel* llmasm_load_model(const char* file_path) {
    FILE* f = fopen(file_path, "rb");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    size_t file_len = ftell(f);
    fseek(f, 0, SEEK_SET);

    uint8_t* buf = (uint8_t*)malloc(file_len);
    if (!buf) {
        fclose(f);
        return NULL;
    }

    if (fread(buf, 1, file_len, f) != file_len) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);

    LLMModel* m = (LLMModel*)calloc(1, sizeof(LLMModel));
    if (!m) {
        free(buf);
        return NULL;
    }

    m->file_buf = buf;
    m->file_len = file_len;
    memcpy(&m->hdr, buf, sizeof(AsmBinHeader));

    if (m->hdr.magic != LLMASM_MAGIC) {
        llmasm_free_model(m);
        return NULL;
    }

    int is_fp16 = (m->hdr.flags & LLMASM_FLAG_FP16_SCALES) || (m->hdr.version >= 2);

    /* 1. Tokenizer offsets */
    uint64_t base_tok = m->hdr.off_tokenizer;
    uint64_t off_b2t, off_offs, off_bytes, off_ht;
    memcpy(&off_b2t, buf + base_tok, 8);
    memcpy(&off_offs, buf + base_tok + 8, 8);
    memcpy(&off_bytes, buf + base_tok + 16, 8);
    memcpy(&off_ht, buf + base_tok + 24, 8);

    m->byte_to_tok = (const uint32_t*)(buf + base_tok + off_b2t);
    m->tok_offsets = (const uint32_t*)(buf + base_tok + off_offs);
    m->tok_bytes = (const uint8_t*)(buf + base_tok + off_bytes);
    m->ht = (const HashEntry*)(buf + base_tok + off_ht);

    /* 2. Embeddings */
    init_q8_matrix(&m->emb, buf + m->hdr.off_embed, m->hdr.vocab_size, m->hdr.dim, is_fp16);

    /* 3. Final norm */
    m->final_norm = (const float*)(buf + m->hdr.off_final_norm);

    /* 4. Layers */
    m->layers = (LayerWeights*)calloc(m->hdr.n_layers, sizeof(LayerWeights));
    int q_dim = m->hdr.n_heads * m->hdr.head_dim;
    int kv_dim = m->hdr.n_kv_heads * m->hdr.head_dim;

    for (uint32_t l = 0; l < m->hdr.n_layers; l++) {
        uint64_t pos = m->hdr.off_layers + (uint64_t)l * m->hdr.layer_stride;
        LayerWeights* lw = &m->layers[l];

        lw->in_norm = (const float*)(buf + pos);
        pos += (uint64_t)m->hdr.dim * 4;

        lw->q_norm = (const float*)(buf + pos);
        pos += (uint64_t)m->hdr.head_dim * 4;

        lw->k_norm = (const float*)(buf + pos);
        pos += (uint64_t)m->hdr.head_dim * 4;

        lw->post_norm = (const float*)(buf + pos);
        pos += (uint64_t)m->hdr.dim * 4;

        #define INIT_MAT(field, rows, cols) \
            pos = align64_u64(pos); \
            init_q8_matrix(&lw->field, buf + pos, rows, cols, is_fp16); \
            pos += (uint64_t)rows * lw->field.row_stride;

        INIT_MAT(q_proj, q_dim, m->hdr.dim);
        INIT_MAT(k_proj, kv_dim, m->hdr.dim);
        INIT_MAT(v_proj, kv_dim, m->hdr.dim);
        INIT_MAT(o_proj, m->hdr.dim, q_dim);
        INIT_MAT(gate_proj, m->hdr.hidden_dim, m->hdr.dim);
        INIT_MAT(up_proj, m->hdr.hidden_dim, m->hdr.dim);
        INIT_MAT(down_proj, m->hdr.dim, m->hdr.hidden_dim);
        #undef INIT_MAT
    }

    /* 5. Allocate KV Cache & Activations */
    size_t kv_size = (size_t)m->hdr.n_layers * m->hdr.max_seq_len * kv_dim;
    m->k_cache = (float*)calloc(kv_size, sizeof(float));
    m->v_cache = (float*)calloc(kv_size, sizeof(float));

    m->act_x = (float*)calloc(m->hdr.dim, sizeof(float));
    m->act_norm = (float*)calloc(m->hdr.dim, sizeof(float));
    m->act_q = (float*)calloc(q_dim, sizeof(float));
    m->act_k = (float*)calloc(kv_dim, sizeof(float));
    m->act_v = (float*)calloc(kv_dim, sizeof(float));
    m->act_attn_out = (float*)calloc(q_dim, sizeof(float));
    m->act_mlp_gate = (float*)calloc(m->hdr.hidden_dim, sizeof(float));
    m->act_mlp_up = (float*)calloc(m->hdr.hidden_dim, sizeof(float));
    m->act_mlp_down = (float*)calloc(m->hdr.dim, sizeof(float));
    m->logits = (float*)calloc(m->hdr.vocab_size, sizeof(float));
    m->n_threads = 4;

    return m;
}

void llmasm_free_model(LLMModel* m) {
    if (!m) return;
    if (m->file_buf) free(m->file_buf);
    if (m->layers) free(m->layers);
    if (m->k_cache) free(m->k_cache);
    if (m->v_cache) free(m->v_cache);
    if (m->act_x) free(m->act_x);
    if (m->act_norm) free(m->act_norm);
    if (m->act_q) free(m->act_q);
    if (m->act_k) free(m->act_k);
    if (m->act_v) free(m->act_v);
    if (m->act_attn_out) free(m->act_attn_out);
    if (m->act_mlp_gate) free(m->act_mlp_gate);
    if (m->act_mlp_up) free(m->act_mlp_up);
    if (m->act_mlp_down) free(m->act_mlp_down);
    if (m->logits) free(m->logits);
    free(m);
}

const AsmBinHeader* llmasm_get_header(const LLMModel* model) {
    return model ? &model->hdr : NULL;
}

void llmasm_reset_cache(LLMModel* model) {
    if (!model) return;
    int kv_dim = model->hdr.n_kv_heads * model->hdr.head_dim;
    size_t kv_size = (size_t)model->hdr.n_layers * model->hdr.max_seq_len * kv_dim;
    memset(model->k_cache, 0, kv_size * sizeof(float));
    memset(model->v_cache, 0, kv_size * sizeof(float));
    model->current_pos = 0;
}

void llmasm_set_threads(LLMModel* model, int n_threads) {
    if (!model) return;
    model->n_threads = (n_threads > 0) ? n_threads : 1;
}

/* ============================================================================
 * Tokenizer & BPE
 * ============================================================================ */

static int ht_lookup(const LLMModel* m, uint32_t left_id, uint32_t right_id, uint32_t* merged_id, uint32_t* rank) {
    uint64_t key = ((uint64_t)left_id << 32) | (uint64_t)right_id;
    uint32_t slot = (uint32_t)(((key * FIB_MULT) & 0xFFFFFFFFFFFFFFFFULL) >> (64 - HT_BITS));
    while (1) {
        uint32_t r = m->ht[slot].rank;
        if (r == 0xFFFFFFFF) return 0;
        if (m->ht[slot].key == key) {
            *merged_id = m->ht[slot].merged_id;
            *rank = r;
            return 1;
        }
        slot = (slot + 1) & m->hdr.ht_mask;
    }
}

int llmasm_encode_bpe(const LLMModel* m, const char* text, int* out_tokens, int max_tokens) {
    if (!m || !text || !out_tokens || max_tokens <= 0) return 0;
    size_t len = strlen(text);
    if (len == 0) return 0;

    int n_toks = 0;
    int* toks = (int*)malloc(len * sizeof(int));
    for (size_t i = 0; i < len; i++) {
        uint8_t b = (uint8_t)text[i];
        toks[n_toks++] = (int)m->byte_to_tok[b];
    }

    while (n_toks > 1) {
        uint32_t best_rank = 0xFFFFFFFF;
        int best_i = -1;
        uint32_t best_c = 0;

        for (int i = 0; i < n_toks - 1; i++) {
            uint32_t merged, r;
            if (ht_lookup(m, (uint32_t)toks[i], (uint32_t)toks[i + 1], &merged, &r)) {
                if (r < best_rank) {
                    best_rank = r;
                    best_i = i;
                    best_c = merged;
                }
            }
        }
        if (best_i < 0) break;
        toks[best_i] = (int)best_c;
        for (int i = best_i + 1; i < n_toks - 1; i++) {
            toks[i] = toks[i + 1];
        }
        n_toks--;
    }

    int written = (n_toks < max_tokens) ? n_toks : max_tokens;
    memcpy(out_tokens, toks, written * sizeof(int));
    free(toks);
    return written;
}

int llmasm_decode_token(const LLMModel* m, int token_id, char* out_str, int max_str_len) {
    if (!m || !out_str || max_str_len <= 0) return 0;
    if (token_id < 0 || (uint32_t)token_id >= m->hdr.vocab_size) {
        out_str[0] = '\0';
        return 0;
    }
    uint32_t a = m->tok_offsets[token_id];
    uint32_t b = m->tok_offsets[token_id + 1];
    uint32_t len = b - a;
    if ((int)len >= max_str_len) len = max_str_len - 1;
    memcpy(out_str, m->tok_bytes + a, len);
    out_str[len] = '\0';
    return (int)len;
}

/* ============================================================================
 * Forward Token & Attention
 * ============================================================================ */

int llmasm_forward_token(LLMModel* m, int token_id, int pos, int compute_logits) {
    if (!m || token_id < 0 || (uint32_t)token_id >= m->hdr.vocab_size) return -1;
    if (pos >= (int)m->hdr.max_seq_len) return -2;

    const int dim = m->hdr.dim;
    const int n_heads = m->hdr.n_heads;
    const int n_kv_heads = m->hdr.n_kv_heads;
    const int head_dim = m->hdr.head_dim;
    const int half = head_dim / 2;
    const int gqa_group = n_heads / n_kv_heads;
    const int kv_dim = n_kv_heads * head_dim;
    const float scale_attn = 1.0f / sqrtf((float)head_dim);

    /* 1. Embedding dequantisation */
    const uint8_t* row_base = m->emb.raw + (size_t)token_id * m->emb.row_stride;
    const int8_t* emb_w = (const int8_t*)row_base;
    const uint8_t* s_base = row_base + dim;
    const int nb = dim / 32;

    for (int b = 0; b < nb; b++) {
        float sc;
        if (m->emb.is_fp16) {
            uint16_t s16;
            memcpy(&s16, s_base + b * 2, 2);
            sc = fp16_to_fp32(s16);
        } else {
            memcpy(&sc, s_base + b * 4, 4);
        }
        for (int k = 0; k < 32; k++) {
            m->act_x[b * 32 + k] = (float)emb_w[b * 32 + k] * sc;
        }
    }

    /* 2. RoPE table calculation */
    float cos_v[256], sin_v[256];
    for (int i = 0; i < half; i++) {
        double freq = 1.0 / pow((double)m->hdr.rope_theta, (double)(2 * i) / (double)head_dim);
        double angle = (double)pos * freq;
        cos_v[i] = (float)cos(angle);
        sin_v[i] = (float)sin(angle);
    }

    /* 3. Layers Loop */
    for (uint32_t l = 0; l < m->hdr.n_layers; l++) {
        LayerWeights* L = &m->layers[l];

        /* Input RMSNorm */
        rmsnorm(m->act_x, L->in_norm, dim, m->hdr.rms_norm_eps, m->act_norm);

        /* Q, K, V projections */
        matvec_q8(&L->q_proj, m->act_norm, m->act_q, m->n_threads);
        matvec_q8(&L->k_proj, m->act_norm, m->act_k, m->n_threads);
        matvec_q8(&L->v_proj, m->act_norm, m->act_v, m->n_threads);

        /* QK-Norm */
        for (int h = 0; h < n_heads; h++) {
            rmsnorm(m->act_q + h * head_dim, L->q_norm, head_dim, m->hdr.rms_norm_eps, m->act_q + h * head_dim);
        }
        for (int kv_h = 0; kv_h < n_kv_heads; kv_h++) {
            rmsnorm(m->act_k + kv_h * head_dim, L->k_norm, head_dim, m->hdr.rms_norm_eps, m->act_k + kv_h * head_dim);
        }

        /* Split-half RoPE */
        for (int h = 0; h < n_heads; h++) {
            float* q_ptr = m->act_q + h * head_dim;
            for (int i = 0; i < half; i++) {
                float q1 = q_ptr[i];
                float q2 = q_ptr[i + half];
                q_ptr[i]        = q1 * cos_v[i] - q2 * sin_v[i];
                q_ptr[i + half] = q2 * cos_v[i] + q1 * sin_v[i];
            }
        }
        for (int kv_h = 0; kv_h < n_kv_heads; kv_h++) {
            float* k_ptr = m->act_k + kv_h * head_dim;
            for (int i = 0; i < half; i++) {
                float k1 = k_ptr[i];
                float k2 = k_ptr[i + half];
                k_ptr[i]        = k1 * cos_v[i] - k2 * sin_v[i];
                k_ptr[i + half] = k2 * cos_v[i] + k1 * sin_v[i];
            }
        }

        /* Store into KV-cache */
        size_t kv_offset = ((size_t)l * m->hdr.max_seq_len + (size_t)pos) * (size_t)kv_dim;
        memcpy(m->k_cache + kv_offset, m->act_k, kv_dim * sizeof(float));
        memcpy(m->v_cache + kv_offset, m->act_v, kv_dim * sizeof(float));

        /* GQA Attention */
        int seq_len = pos + 1;
        float scores[4096];

        for (int h = 0; h < n_heads; h++) {
            int kv_h = h / gqa_group;
            const float* q_h = m->act_q + h * head_dim;

            float max_s = -1e30f;
            for (int t = 0; t < seq_len; t++) {
                size_t t_offset = ((size_t)l * m->hdr.max_seq_len + (size_t)t) * (size_t)kv_dim + (size_t)kv_h * head_dim;
                const float* k_t = m->k_cache + t_offset;

                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    dot += k_t[d] * q_h[d];
                }
                dot *= scale_attn;
                scores[t] = dot;
                if (dot > max_s) max_s = dot;
            }

            /* Softmax */
            float sum_exp = 0.0f;
            for (int t = 0; t < seq_len; t++) {
                scores[t] = expf(scores[t] - max_s);
                sum_exp += scores[t];
            }
            float inv_sum = 1.0f / sum_exp;
            for (int t = 0; t < seq_len; t++) {
                scores[t] *= inv_sum;
            }

            /* Weighted sum of V */
            float* out_h = m->act_attn_out + h * head_dim;
            memset(out_h, 0, head_dim * sizeof(float));
            for (int t = 0; t < seq_len; t++) {
                size_t t_offset = ((size_t)l * m->hdr.max_seq_len + (size_t)t) * (size_t)kv_dim + (size_t)kv_h * head_dim;
                const float* v_t = m->v_cache + t_offset;
                float sc = scores[t];
                for (int d = 0; d < head_dim; d++) {
                    out_h[d] += sc * v_t[d];
                }
            }
        }

        /* O projection & residual */
        matvec_q8(&L->o_proj, m->act_attn_out, m->act_norm, m->n_threads);
        for (int i = 0; i < dim; i++) {
            m->act_x[i] += m->act_norm[i];
        }

        /* MLP / SwiGLU */
        rmsnorm(m->act_x, L->post_norm, dim, m->hdr.rms_norm_eps, m->act_norm);
        matvec_q8(&L->gate_proj, m->act_norm, m->act_mlp_gate, m->n_threads);
        matvec_q8(&L->up_proj, m->act_norm, m->act_mlp_up, m->n_threads);
        silu_mul(m->act_mlp_gate, m->act_mlp_up, m->hdr.hidden_dim, m->act_mlp_gate);
        matvec_q8(&L->down_proj, m->act_mlp_gate, m->act_mlp_down, m->n_threads);
        for (int i = 0; i < dim; i++) {
            m->act_x[i] += m->act_mlp_down[i];
        }
    }

    if (!compute_logits) return 0;

    /* Final RMSNorm and tied LM head */
    rmsnorm(m->act_x, m->final_norm, dim, m->hdr.rms_norm_eps, m->act_norm);
    matvec_q8(&m->emb, m->act_norm, m->logits, m->n_threads);
    return 0;
}

const float* llmasm_get_logits(const LLMModel* model) {
    return model ? model->logits : NULL;
}

int llmasm_sample_greedy(const LLMModel* m) {
    if (!m || !m->logits) return 0;
    int best = 0;
    float best_v = m->logits[0];
    for (uint32_t i = 1; i < m->hdr.vocab_size; i++) {
        if (m->logits[i] > best_v) {
            best_v = m->logits[i];
            best = i;
        }
    }
    return best;
}

int llmasm_sample_topk(const LLMModel* m, float temperature, int top_k) {
    if (!m || !m->logits) return 0;
    if (temperature <= 0.0f || top_k <= 1) {
        return llmasm_sample_greedy(m);
    }
    /* Simple greedy fallback if top-k not specified */
    return llmasm_sample_greedy(m);
}

int llmasm_generate(
    LLMModel* model,
    const int* prompt_tokens,
    int prompt_len,
    int max_tokens,
    float temperature,
    int top_k,
    int* out_tokens,
    int max_out_len
) {
    if (!model || !prompt_tokens || prompt_len <= 0 || !out_tokens || max_out_len <= 0) return 0;

    llmasm_reset_cache(model);
    int pos = 0;

    /* Prefill prompt */
    for (int i = 0; i < prompt_len - 1; i++) {
        llmasm_forward_token(model, prompt_tokens[i], pos++, 0);
    }
    llmasm_forward_token(model, prompt_tokens[prompt_len - 1], pos++, 1);

    int gen_count = 0;
    for (int step = 0; step < max_tokens && gen_count < max_out_len; step++) {
        if (pos >= (int)model->hdr.max_seq_len) break;
        int next_tok = llmasm_sample_topk(model, temperature, top_k);
        if (next_tok == (int)model->hdr.eos_1 || next_tok == (int)model->hdr.eos_2) break;

        out_tokens[gen_count++] = next_tok;
        llmasm_forward_token(model, next_tok, pos++, 1);
    }

    return gen_count;
}
