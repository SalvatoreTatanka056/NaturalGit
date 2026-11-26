"""
Python ctypes wrapper for the high-performance C inference engine (libllmasm).
Directly executes .asmbin models in compiled C machine code.
"""

import ctypes
from ctypes import (
    c_char_p,
    c_float,
    c_int,
    c_uint8,
    c_uint32,
    c_uint64,
    c_void_p,
    POINTER,
    Structure,
)
import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

from .c_builder import build_c_library, find_compiled_lib


class CAsmBinHeader(Structure):
    _pack_ = 1
    _fields_ = [
        ("magic", c_uint32),
        ("version", c_uint32),
        ("dim", c_uint32),
        ("hidden_dim", c_uint32),
        ("n_layers", c_uint32),
        ("n_heads", c_uint32),
        ("n_kv_heads", c_uint32),
        ("head_dim", c_uint32),
        ("vocab_size", c_uint32),
        ("max_seq_len", c_uint32),
        ("rms_norm_eps", c_float),
        ("rope_theta", c_float),
        ("eos_1", c_uint32),
        ("eos_2", c_uint32),
        ("ht_mask", c_uint32),
        ("flags", c_uint32),
        ("off_tokenizer", c_uint64),
        ("off_embed", c_uint64),
        ("off_layers", c_uint64),
        ("layer_stride", c_uint64),
        ("off_final_norm", c_uint64),
        ("file_size", c_uint64),
        ("reserved", c_uint8 * 16),
    ]


class CLibBinding:
    """Loads and wraps the C dynamic library symbols."""

    def __init__(self, lib_path: Optional[Union[str, Path]] = None):
        if lib_path is None:
            found = find_compiled_lib()
            if not found:
                found = build_c_library(verbose=False)
            if not found:
                raise RuntimeError(
                    "C shared library (llmasm.dll / libllmasm.so) not found and could not be built automatically. "
                    "Compile it with: cd src_c && make (or cmake)."
                )
            lib_path = found

        self.lib_path = str(lib_path)
        self.cdll = ctypes.CDLL(self.lib_path)
        self._setup_signatures()

    def _setup_signatures(self):
        # LLMModel* llmasm_load_model(const char* file_path);
        self.cdll.llmasm_load_model.argtypes = [c_char_p]
        self.cdll.llmasm_load_model.restype = c_void_p

        # void llmasm_free_model(LLMModel* model);
        self.cdll.llmasm_free_model.argtypes = [c_void_p]
        self.cdll.llmasm_free_model.restype = None

        # const AsmBinHeader* llmasm_get_header(const LLMModel* model);
        self.cdll.llmasm_get_header.argtypes = [c_void_p]
        self.cdll.llmasm_get_header.restype = POINTER(CAsmBinHeader)

        # void llmasm_reset_cache(LLMModel* model);
        self.cdll.llmasm_reset_cache.argtypes = [c_void_p]
        self.cdll.llmasm_reset_cache.restype = None

        # void llmasm_set_threads(LLMModel* model, int n_threads);
        self.cdll.llmasm_set_threads.argtypes = [c_void_p, c_int]
        self.cdll.llmasm_set_threads.restype = None

        # int llmasm_encode_bpe(const LLMModel* model, const char* text, int* out_tokens, int max_tokens);
        self.cdll.llmasm_encode_bpe.argtypes = [c_void_p, c_char_p, POINTER(c_int), c_int]
        self.cdll.llmasm_encode_bpe.restype = c_int

        # int llmasm_decode_token(const LLMModel* model, int token_id, char* out_str, int max_str_len);
        self.cdll.llmasm_decode_token.argtypes = [c_void_p, c_int, c_char_p, c_int]
        self.cdll.llmasm_decode_token.restype = c_int

        # int llmasm_forward_token(LLMModel* model, int token_id, int pos, int compute_logits);
        self.cdll.llmasm_forward_token.argtypes = [c_void_p, c_int, c_int, c_int]
        self.cdll.llmasm_forward_token.restype = c_int

        # const float* llmasm_get_logits(const LLMModel* model);
        self.cdll.llmasm_get_logits.argtypes = [c_void_p]
        self.cdll.llmasm_get_logits.restype = POINTER(c_float)

        # int llmasm_sample_greedy(const LLMModel* model);
        self.cdll.llmasm_sample_greedy.argtypes = [c_void_p]
        self.cdll.llmasm_sample_greedy.restype = c_int

        # int llmasm_sample_topk(const LLMModel* model, float temperature, int top_k);
        self.cdll.llmasm_sample_topk.argtypes = [c_void_p, c_float, c_int]
        self.cdll.llmasm_sample_topk.restype = c_int

        # int llmasm_generate(...)
        self.cdll.llmasm_generate.argtypes = [
            c_void_p,
            POINTER(c_int),
            c_int,
            c_int,
            c_float,
            c_int,
            POINTER(c_int),
            c_int,
        ]
        self.cdll.llmasm_generate.restype = c_int


class CAsmBinModel:
    """
    High-level Python wrapper executing directly via compiled C inference engine.
    """

    def __init__(self, model_path: str, lib_path: Optional[Union[str, Path]] = None, threads: int = 4):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.path = os.path.abspath(model_path)
        self._binding = CLibBinding(lib_path)
        self._handle = self._binding.cdll.llmasm_load_model(self.path.encode("utf-8"))

        if not self._handle:
            raise RuntimeError(f"Failed to load model in C engine from: {model_path}")

        header_ptr = self._binding.cdll.llmasm_get_header(self._handle)
        self.hdr = header_ptr.contents
        self.set_threads(threads)

    def set_threads(self, n: int):
        """Sets the number of CPU worker threads."""
        self._threads = max(1, int(n))
        if getattr(self, "_handle", None):
            self._binding.cdll.llmasm_set_threads(self._handle, self._threads)

    def close(self):
        """Frees model in C memory."""
        if getattr(self, "_handle", None):
            self._binding.cdll.llmasm_free_model(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def reset_cache(self):
        """Resets C KV-cache."""
        self._binding.cdll.llmasm_reset_cache(self._handle)

    def encode(self, text: str, max_tokens: int = 4096) -> List[int]:
        """Encodes text using C Fibonacci BPE hash table."""
        out_buf = (c_int * max_tokens)()
        n = self._binding.cdll.llmasm_encode_bpe(
            self._handle,
            text.encode("utf-8"),
            out_buf,
            max_tokens,
        )
        return list(out_buf[:n])

    def decode_token(self, token_id: int, max_str_len: int = 256) -> str:
        """Decodes a single token into UTF-8 text via C engine."""
        buf = ctypes.create_string_buffer(max_str_len)
        n = self._binding.cdll.llmasm_decode_token(self._handle, token_id, buf, max_str_len)
        return buf.raw[:n].decode("utf-8", errors="replace")

    def decode(self, token_ids: List[int]) -> str:
        """Decodes token list."""
        return "".join(self.decode_token(tid) for tid in token_ids)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 64,
        temperature: float = 0.0,
        top_k: int = 1,
        stream: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Runs generation directly inside native C.
        """
        prompt_tokens = self.encode(prompt)
        if not prompt_tokens:
            prompt_tokens = [self.hdr.eos_1]

        if stream:
            def _stream_gen():
                self.reset_cache()
                pos = 0
                for tid in prompt_tokens[:-1]:
                    self._binding.cdll.llmasm_forward_token(self._handle, tid, pos, 0)
                    pos += 1

                last_tok = prompt_tokens[-1]
                self._binding.cdll.llmasm_forward_token(self._handle, last_tok, pos, 1)
                pos += 1

                for _ in range(max_tokens):
                    if pos >= self.hdr.max_seq_len:
                        break
                    next_tok = self._binding.cdll.llmasm_sample_topk(
                        self._handle, c_float(temperature), top_k
                    )
                    if next_tok in (self.hdr.eos_1, self.hdr.eos_2):
                        break
                    yield self.decode_token(next_tok)
                    self._binding.cdll.llmasm_forward_token(self._handle, next_tok, pos, 1)
                    pos += 1

            return _stream_gen()

        # Direct C buffer generation
        c_prompt = (c_int * len(prompt_tokens))(*prompt_tokens)
        c_out = (c_int * max_tokens)()
        gen_count = self._binding.cdll.llmasm_generate(
            self._handle,
            c_prompt,
            len(prompt_tokens),
            max_tokens,
            c_float(temperature),
            top_k,
            c_out,
            max_tokens,
        )
        tokens = list(c_out[:gen_count])
        return self.decode(tokens)

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "engine": "C (Native Shared Library)",
            "dim": self.hdr.dim,
            "hidden_dim": self.hdr.hidden_dim,
            "n_layers": self.hdr.n_layers,
            "n_heads": self.hdr.n_heads,
            "n_kv_heads": self.hdr.n_kv_heads,
            "head_dim": self.hdr.head_dim,
            "vocab_size": self.hdr.vocab_size,
            "max_seq_len": self.hdr.max_seq_len,
            "file_size_mb": round(self.hdr.file_size / (1024 * 1024), 2),
        }
