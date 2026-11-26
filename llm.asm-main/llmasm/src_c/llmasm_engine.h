/**
 * llmasm_engine.h - High-Performance C Inference Engine for llm.asm (.asmbin)
 * 
 * Supports:
 * - Direct memory-mapped / in-memory .asmbin v1 & v2 models
 * - Per-row split Q8_0 quantisation (with AVX2+FMA SIMD and scalar fallback)
 * - RMSNorm, SwiGLU, Split-Half RoPE, Grouped Query Attention (GQA)
 * - Fibonacci open-addressing BPE tokenizer
 * - Contiguous KV-cache
 * - C API callable from Python via ctypes
 */

#ifndef LLMASM_ENGINE_H
#define LLMASM_ENGINE_H

#include <stdint.h>
#include <stddef.h>

#ifdef _WIN32
    #ifdef LLMASM_EXPORTS
        #define LLMASM_API __declspec(dllexport)
    #else
        #define LLMASM_API __declspec(dllimport)
    #endif
#else
    #define LLMASM_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Header flags */
#define LLMASM_FLAG_TIED_EMB     1
#define LLMASM_FLAG_QK_NORM      2
#define LLMASM_FLAG_FP16_SCALES  4

#define LLMASM_MAGIC 0x4C4D5341 /* "ASML" */

#pragma pack(push, 1)
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t dim;
    uint32_t hidden_dim;
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t vocab_size;
    uint32_t max_seq_len;
    float    rms_norm_eps;
    float    rope_theta;
    uint32_t eos_1;
    uint32_t eos_2;
    uint32_t ht_mask;
    uint32_t flags;
    uint64_t off_tokenizer;
    uint64_t off_embed;
    uint64_t off_layers;
    uint64_t layer_stride;
    uint64_t off_final_norm;
    uint64_t file_size;
    uint8_t  reserved[16];
} AsmBinHeader;

typedef struct {
    uint64_t key;
    uint32_t merged_id;
    uint32_t rank;
} HashEntry;
#pragma pack(pop)

/* Model forward and context opaque struct */
typedef struct LLMModel LLMModel;

/**
 * Loads an .asmbin model file into memory.
 * Returns pointer to LLMModel or NULL on error.
 */
LLMASM_API LLMModel* llmasm_load_model(const char* file_path);

/**
 * Frees a loaded model and its KV-cache.
 */
LLMASM_API void llmasm_free_model(LLMModel* model);

/**
 * Gets model metadata.
 */
LLMASM_API const AsmBinHeader* llmasm_get_header(const LLMModel* model);

/**
 * Resets KV-cache position to 0.
 */
LLMASM_API void llmasm_reset_cache(LLMModel* model);

/**
 * Sets the number of CPU worker threads (OpenMP/multithreading).
 */
LLMASM_API void llmasm_set_threads(LLMModel* model, int n_threads);

/**
 * Encodes text into token IDs using BPE Fibonacci hash table.
 * Returns the number of tokens written into out_tokens.
 */
LLMASM_API int llmasm_encode_bpe(const LLMModel* model, const char* text, int* out_tokens, int max_tokens);

/**
 * Decodes a token ID into UTF-8 text.
 * Returns bytes written into out_str.
 */
LLMASM_API int llmasm_decode_token(const LLMModel* model, int token_id, char* out_str, int max_str_len);

/**
 * Runs forward pass for a single token at position pos.
 * If compute_logits != 0, computes and stores output logits.
 * Returns 0 on success, negative on error.
 */
LLMASM_API int llmasm_forward_token(LLMModel* model, int token_id, int pos, int compute_logits);

/**
 * Returns pointer to the current output logits [vocab_size].
 */
LLMASM_API const float* llmasm_get_logits(const LLMModel* model);

/**
 * Returns argmax token from current logits (greedy sample).
 */
LLMASM_API int llmasm_sample_greedy(const LLMModel* model);

/**
 * Samples next token with temperature and top-k.
 */
LLMASM_API int llmasm_sample_topk(const LLMModel* model, float temperature, int top_k);

/**
 * Generates tokens up to max_tokens.
 * Returns total tokens generated.
 */
LLMASM_API int llmasm_generate(
    LLMModel* model,
    const int* prompt_tokens,
    int prompt_len,
    int max_tokens,
    float temperature,
    int top_k,
    int* out_tokens,
    int max_out_len
);

#ifdef __cplusplus
}
#endif

#endif /* LLMASM_ENGINE_H */
