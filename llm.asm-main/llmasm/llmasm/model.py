"""
Core model inference engine for .asmbin models.
Provides memory-mapped zero-copy weights, Q8_0 dequantization,
KV-cache management, and autoregressive text generation.
"""

from dataclasses import dataclass
import math
import mmap
import os
import struct
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import numpy as np

from .tokenizer import AsmTokenizer


def align64(n: int) -> int:
    """Aligns an offset to 64 bytes for cache line and AVX2 alignment."""
    return (n + 63) & ~63


@dataclass
class GenerationOutput:
    """Output from text generation."""
    text: str
    tokens: List[int]
    prompt_tokens: int
    generated_tokens: int
    elapsed_seconds: float
    tokens_per_second: float


class AsmBinModel:
    """
    Model loader and inference runtime for .asmbin format (Qwen3 / Transformer architecture).
    """

    def __init__(self, model_path: str, threads: int = 4):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self._threads = 4
        self.set_threads(threads)
        self.path = model_path
        self._f = open(model_path, "rb")
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        self.buf = memoryview(self._mm)

        # Parse 128-byte header:
        # magic (4s), version (I), dim (I), hidden_dim (I), n_layers (I), n_heads (I),
        # n_kv_heads (I), head_dim (I), vocab_size (I), max_seq_len (I), rms_norm_eps (f),
        # rope_theta (f), eos_1 (I), eos_2 (I), ht_mask (I), flags (I),
        # off_tokenizer (Q), off_embed (Q), off_layers (Q), layer_stride (Q),
        # off_final_norm (Q), file_size (Q)
        hdr = struct.unpack_from("<4sIIIIIIIIIffIIIIQQQQQQ", self.buf, 0)
        if hdr[0] != b"ASML":
            raise ValueError(f"Invalid magic header: expected b'ASML', got {hdr[0]}")
        if hdr[1] not in (1, 2):
            raise ValueError(f"Unsupported .asmbin version: {hdr[1]}")

        (
            self.magic,
            self.version,
            self.dim,
            self.hidden_dim,
            self.n_layers,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
            self.vocab_size,
            self.max_seq_len,
            self.rms_norm_eps,
            self.rope_theta,
            self.eos_1,
            self.eos_2,
            self.ht_mask,
            self.flags,
            self.off_tokenizer,
            self.off_embed,
            self.off_layers,
            self.layer_stride,
            self.off_final_norm,
            self.file_size,
        ) = hdr

        self.fp16_scales = bool(self.flags & 4) or (self.version >= 2)
        self.scale_bytes = 2 if self.fp16_scales else 4
        self.q_dim = self.n_heads * self.head_dim
        self.kv_dim = self.n_kv_heads * self.head_dim

        self._load_tokenizer()
        self._load_weights()

        # Allocate persistent KV-cache
        self._k_cache = np.zeros(
            (self.n_layers, self.max_seq_len, self.n_kv_heads, self.head_dim),
            dtype=np.float32,
        )
        self._v_cache = np.zeros(
            (self.n_layers, self.max_seq_len, self.n_kv_heads, self.head_dim),
            dtype=np.float32,
        )
        self._current_pos = 0

    def set_threads(self, n: int):
        """Sets the number of CPU worker threads for NumPy / BLAS / OpenMP."""
        import ctypes
        from pathlib import Path

        n = max(1, int(n))
        self._threads = n
        n_str = str(n)

        for var in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[var] = n_str

        # Try threadpoolctl if installed
        try:
            import threadpoolctl
            threadpoolctl.threadpool_limits(limits=n)
        except Exception:
            pass

        # Directly configure loaded OpenBLAS / MKL shared libraries in numpy.libs / numpy/.libs
        try:
            np_root = Path(np.__file__).resolve().parent
            lib_dirs = [
                np_root.parent / "numpy.libs",
                np_root / ".libs",
                np_root.parent / "scipy_openblas64" / "lib",
                np_root.parent / "scipy_openblas32" / "lib",
            ]
            for lib_dir in lib_dirs:
                if not lib_dir.exists():
                    continue
                for lib_file in lib_dir.iterdir():
                    if not (lib_file.suffix in (".dll", ".so", ".dylib") or ".so." in lib_file.name):
                        continue
                    fname = lib_file.name.lower()
                    if "openblas" in fname or "blas" in fname or "mkl" in fname:
                        try:
                            dll = ctypes.CDLL(str(lib_file))
                            for fn_name in (
                                "scipy_openblas_set_num_threads64_",
                                "scipy_openblas_set_num_threads",
                                "openblas_set_num_threads",
                                "MKL_Set_Num_Threads",
                            ):
                                fn = getattr(dll, fn_name, None)
                                if fn is not None:
                                    fn.argtypes = [ctypes.c_int]
                                    fn(n)
                        except Exception:
                            pass
        except Exception:
            pass

    def _load_tokenizer(self):
        """Loads BPE tokenizer data structures directly from mmap buffer."""
        base = self.off_tokenizer
        off_b2t, off_offs, off_bytes, off_ht = struct.unpack_from("<QQQQ", self.buf, base)

        byte_to_tok = np.frombuffer(self.buf, dtype="<u4", count=256, offset=base + off_b2t)
        tok_offsets = np.frombuffer(
            self.buf, dtype="<u4", count=self.vocab_size + 1, offset=base + off_offs
        )
        bytes_len = int(tok_offsets[-1])
        tok_bytes = bytes(self.buf[base + off_bytes : base + off_bytes + bytes_len])

        ht_dtype = np.dtype([("key", "<u8"), ("merged_id", "<u4"), ("rank", "<u4")])
        ht = np.frombuffer(self.buf, dtype=ht_dtype, count=self.ht_mask + 1, offset=base + off_ht)

        self.tokenizer = AsmTokenizer(
            byte_to_tok=byte_to_tok,
            tok_offsets=tok_offsets,
            tok_bytes=tok_bytes,
            ht_keys=ht["key"],
            ht_merged=ht["merged_id"],
            ht_rank=ht["rank"],
            ht_mask=self.ht_mask,
            vocab_size=self.vocab_size,
        )

    def _view_q8_0(self, offset: int, rows: int, cols: int) -> Tuple[np.ndarray, np.ndarray]:
        """Creates zero-copy views for Q8_0 quantised matrix: (int8 weights, float32 scales)."""
        nb = cols // 32
        row_stride = cols + nb * self.scale_bytes
        raw = np.frombuffer(self.buf, dtype=np.uint8, count=rows * row_stride, offset=offset).reshape(
            rows, row_stride
        )
        q = raw[:, :cols].view(np.int8)
        if self.fp16_scales:
            s = np.ascontiguousarray(raw[:, cols:]).view("<f2").reshape(rows, nb).astype(np.float32)
        else:
            s = np.ascontiguousarray(raw[:, cols:]).view("<f4").reshape(rows, nb)
        return q, s

    @staticmethod
    def _get_available_ram_bytes() -> int:
        """Returns available physical RAM in bytes (Windows & Linux/macOS)."""
        import sys
        if sys.platform == "win32":
            try:
                import ctypes
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                    return int(stat.ullAvailPhys)
            except Exception:
                pass
        else:
            try:
                with open("/proc/meminfo", "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("MemAvailable:"):
                            return int(line.split()[1]) * 1024
            except Exception:
                pass
        return 4 * 1024 * 1024 * 1024  # Conservative 4GB fallback

    @staticmethod
    def _dequant_q8(wq: np.ndarray, ws: np.ndarray) -> np.ndarray:
        """Dequantizes Q8_0 (int8, float32 scales) into contiguous float32 matrix."""
        M, K = wq.shape
        nb = K // 32
        return np.ascontiguousarray(
            (wq.reshape(M, nb, 32).astype(np.float32) * ws[:, :, None]).reshape(M, K)
        )

    def _load_weights(self):
        """Builds tensor views for all layers and pre-caches FP32 if RAM allows."""
        self.emb_q, self.emb_s = self._view_q8_0(self.off_embed, self.vocab_size, self.dim)
        self.final_norm = np.frombuffer(self.buf, dtype="<f4", count=self.dim, offset=self.off_final_norm)
        self._rope_freqs = 1.0 / (
            self.rope_theta ** (np.arange(0, self.head_dim, 2, dtype=np.float64) / self.head_dim)
        )

        self.layers = []
        for l in range(self.n_layers):
            pos = self.off_layers + l * self.layer_stride
            in_norm = np.frombuffer(self.buf, dtype="<f4", count=self.dim, offset=pos)
            pos += self.dim * 4
            q_norm = np.frombuffer(self.buf, dtype="<f4", count=self.head_dim, offset=pos)
            pos += self.head_dim * 4
            k_norm = np.frombuffer(self.buf, dtype="<f4", count=self.head_dim, offset=pos)
            pos += self.head_dim * 4
            post_norm = np.frombuffer(self.buf, dtype="<f4", count=self.dim, offset=pos)
            pos += self.dim * 4

            def next_mat(rows, cols):
                nonlocal pos
                pos = align64(pos)
                mat = self._view_q8_0(pos, rows, cols)
                pos += rows * (cols + (cols // 32) * self.scale_bytes)
                return mat

            self.layers.append({
                "in_norm": in_norm,
                "q_norm": q_norm,
                "k_norm": k_norm,
                "post_norm": post_norm,
                "q_proj": next_mat(self.q_dim, self.dim),
                "k_proj": next_mat(self.kv_dim, self.dim),
                "v_proj": next_mat(self.kv_dim, self.dim),
                "o_proj": next_mat(self.dim, self.q_dim),
                "gate_proj": next_mat(self.hidden_dim, self.dim),
                "up_proj": next_mat(self.hidden_dim, self.dim),
                "down_proj": next_mat(self.dim, self.hidden_dim),
            })

        # Estimate FP32 footprint; if sufficient RAM, pre-dequantize for ~10x OpenBLAS speedup
        per_layer_params = (
            self.q_dim * self.dim
            + self.kv_dim * self.dim * 2
            + self.dim * self.q_dim
            + self.hidden_dim * self.dim * 3
        )
        est_f32_bytes = (self.n_layers * per_layer_params + self.vocab_size * self.dim) * 4
        avail_ram = self._get_available_ram_bytes()
        self._use_f32_cache = avail_ram > int(est_f32_bytes * 1.4)
        self.emb_f32 = None

        if self._use_f32_cache:
            self.emb_f32 = self._dequant_q8(self.emb_q, self.emb_s)
            for L in self.layers:
                for k in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
                    L[k + "_f32"] = self._dequant_q8(*L[k])

    @staticmethod
    def matvec_q8_f32(wq: np.ndarray, ws: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Matrix-vector multiplication for Q8_0 weights with FP32 vector."""
        M, K = wq.shape
        nb = K // 32
        wb = wq.reshape(M, nb, 32).astype(np.float32)
        xb = x.reshape(nb, 32).astype(np.float32)
        dots = np.einsum("mnb,nb->mn", wb, xb, optimize=False).astype(np.float32)
        return np.sum(dots * ws, axis=1, dtype=np.float32)

    @staticmethod
    def rmsnorm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
        """Root Mean Square Layer Normalization."""
        ss = np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32)
        inv = np.float32(1.0) / np.sqrt(ss + np.float32(eps))
        return (x * inv * w).astype(np.float32)

    @staticmethod
    def silu_mul(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
        """SwiGLU activation: SiLU(gate) * up."""
        ax = np.abs(gate)
        e = np.exp(-ax).astype(np.float32)
        num = np.where(gate < 0, e, np.float32(1.0))
        silu = (gate * num) / (np.float32(1.0) + e)
        return (silu * up).astype(np.float32)

    def _proj(self, L: Dict[str, Any], name: str, x: np.ndarray) -> np.ndarray:
        """Fast projection using cached FP32 BLAS if available, else Q8_0."""
        f32_key = name + "_f32"
        if f32_key in L:
            return L[f32_key] @ x
        return self.matvec_q8_f32(*L[name], x)

    def forward_prompt(
        self,
        tokens: List[int],
        start_pos: int = 0,
        compute_logits: bool = True,
    ) -> Optional[np.ndarray]:
        """Batched prompt prefill using SGEMM for high-speed prompt processing."""
        T = len(tokens)
        if T == 0:
            return None
        if T == 1 or not self._use_f32_cache:
            out = None
            for i, tid in enumerate(tokens):
                out = self.forward_token(tid, start_pos + i, compute_logits=(compute_logits and i == T - 1))
            return out

        # Gather embeddings: [T, dim]
        x = self.emb_f32[tokens].copy()
        half = self.head_dim // 2
        positions = np.arange(start_pos, start_pos + T, dtype=np.float64)[:, None]
        angles = positions * self._rope_freqs[None, :]  # [T, half]
        cos_v = np.cos(angles).astype(np.float32)[:, None, :]  # [T, 1, half]
        sin_v = np.sin(angles).astype(np.float32)[:, None, :]
        scale = np.float32(1.0 / math.sqrt(self.head_dim))
        gqa_group = self.n_heads // self.n_kv_heads

        # Causal mask for prefill [T, T]
        causal_mask = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)

        for l in range(self.n_layers):
            L = self.layers[l]
            h = self.rmsnorm(x, L["in_norm"], self.rms_norm_eps)
            q = (h @ L["q_proj_f32"].T).reshape(T, self.n_heads, self.head_dim)
            k = (h @ L["k_proj_f32"].T).reshape(T, self.n_kv_heads, self.head_dim)
            v = (h @ L["v_proj_f32"].T).reshape(T, self.n_kv_heads, self.head_dim)

            q = self.rmsnorm(q, L["q_norm"], self.rms_norm_eps)
            k = self.rmsnorm(k, L["k_norm"], self.rms_norm_eps)

            q1, q2 = q[..., :half], q[..., half:]
            q = np.concatenate([q1 * cos_v - q2 * sin_v, q2 * cos_v + q1 * sin_v], axis=-1)
            k1, k2 = k[..., :half], k[..., half:]
            k = np.concatenate([k1 * cos_v - k2 * sin_v, k2 * cos_v + k1 * sin_v], axis=-1)

            self._k_cache[l, start_pos : start_pos + T] = k
            self._v_cache[l, start_pos : start_pos + T] = v

            # Vectorized causal GQA attention
            k_ctx = self._k_cache[l, : start_pos + T]  # [S, kv_heads, head_dim]
            v_ctx = self._v_cache[l, : start_pos + T]

            q_g = q.reshape(T, self.n_kv_heads, gqa_group, self.head_dim).transpose(1, 2, 0, 3)  # [kv, g, T, D]
            k_g = k_ctx.transpose(1, 2, 0)[:, None, :, :]  # [kv, 1, D, S]
            scores = np.matmul(q_g, k_g) * scale  # [kv, g, T, S]
            if start_pos == 0:
                scores += causal_mask[None, None, :, :]
            else:
                scores[:, :, :, start_pos:] += causal_mask[None, None, :, :]

            scores -= np.max(scores, axis=-1, keepdims=True)
            weights = np.exp(scores)
            weights /= np.sum(weights, axis=-1, keepdims=True)

            v_g = v_ctx.transpose(1, 0, 2)[:, None, :, :]  # [kv, 1, S, D]
            attn_out = np.matmul(weights, v_g).transpose(2, 0, 1, 3).reshape(T, self.q_dim)

            x += attn_out @ L["o_proj_f32"].T

            # FFN
            h_mlp = self.rmsnorm(x, L["post_norm"], self.rms_norm_eps)
            gate = h_mlp @ L["gate_proj_f32"].T
            up = h_mlp @ L["up_proj_f32"].T
            swiglu = self.silu_mul(gate, up)
            x += swiglu @ L["down_proj_f32"].T

        if not compute_logits:
            return None

        x_norm = self.rmsnorm(x[-1], self.final_norm, self.rms_norm_eps)
        return self.emb_f32 @ x_norm

    def forward_token(
        self,
        token_id: int,
        pos: int,
        compute_logits: bool = True,
    ) -> Optional[np.ndarray]:
        """Executes a single token forward pass updating internal KV-cache."""
        if self.emb_f32 is not None:
            x = self.emb_f32[token_id].copy()
        else:
            x = (
                self.emb_q[token_id].reshape(-1, 32).astype(np.float32)
                * self.emb_s[token_id][:, None]
            ).reshape(-1)

        half = self.head_dim // 2
        angles = pos * self._rope_freqs
        cos_v = np.cos(angles).astype(np.float32)[None, :]
        sin_v = np.sin(angles).astype(np.float32)[None, :]
        scale = np.float32(1.0 / math.sqrt(self.head_dim))
        gqa_group = self.n_heads // self.n_kv_heads

        for l in range(self.n_layers):
            L = self.layers[l]
            h = self.rmsnorm(x, L["in_norm"], self.rms_norm_eps)
            q = self._proj(L, "q_proj", h).reshape(self.n_heads, self.head_dim)
            k = self._proj(L, "k_proj", h).reshape(self.n_kv_heads, self.head_dim)
            v = self._proj(L, "v_proj", h).reshape(self.n_kv_heads, self.head_dim)

            # QK-Norm per head
            q = self.rmsnorm(q, L["q_norm"], self.rms_norm_eps)
            k = self.rmsnorm(k, L["k_norm"], self.rms_norm_eps)

            # Split-half RoPE
            q1, q2 = q[:, :half], q[:, half:]
            q = np.concatenate([q1 * cos_v - q2 * sin_v, q2 * cos_v + q1 * sin_v], axis=-1)
            k1, k2 = k[:, :half], k[:, half:]
            k = np.concatenate([k1 * cos_v - k2 * sin_v, k2 * cos_v + k1 * sin_v], axis=-1)

            self._k_cache[l, pos] = k
            self._v_cache[l, pos] = v

            # Vectorized Grouped Query Attention (GQA)
            k_ctx = self._k_cache[l, : pos + 1]  # [T, kv_heads, head_dim]
            v_ctx = self._v_cache[l, : pos + 1]

            q_g = q.reshape(self.n_kv_heads, gqa_group, self.head_dim)        # [kv, g, D]
            k_g = k_ctx.transpose(1, 2, 0)                                     # [kv, D, T]
            scores = np.matmul(q_g, k_g) * scale                               # [kv, g, T]
            scores -= np.max(scores, axis=-1, keepdims=True)
            weights = np.exp(scores)
            weights /= np.sum(weights, axis=-1, keepdims=True)
            v_g = v_ctx.transpose(1, 0, 2)                                     # [kv, T, D]
            attn_vec = np.matmul(weights, v_g).reshape(-1)                     # [n_heads * D]

            x += self._proj(L, "o_proj", attn_vec)

            # Feed-Forward Network (SwiGLU)
            h_mlp = self.rmsnorm(x, L["post_norm"], self.rms_norm_eps)
            gate = self._proj(L, "gate_proj", h_mlp)
            up = self._proj(L, "up_proj", h_mlp)
            swiglu = self.silu_mul(gate, up)
            x += self._proj(L, "down_proj", swiglu)

        if not compute_logits:
            return None

        # Final RMSNorm and tied LM head projection
        x_norm = self.rmsnorm(x, self.final_norm, self.rms_norm_eps)
        if self.emb_f32 is not None:
            return self.emb_f32 @ x_norm
        return self.matvec_q8_f32(self.emb_q, self.emb_s, x_norm)

    def reset_cache(self):
        """Resets the KV-cache and position pointer."""
        self._k_cache.fill(0)
        self._v_cache.fill(0)
        self._current_pos = 0

    def generate(
        self,
        prompt: str,
        max_tokens: int = 64,
        temperature: float = 0.0,
        top_k: int = 1,
        chat_format: bool = False,
        think: bool = False,
        stream: bool = False,
        stop_tokens: Optional[List[int]] = None,
        stop_strings: Optional[List[str]] = None,
    ) -> Union[GenerationOutput, Generator[str, None, None]]:
        """
        Generates text completion.
        If stream=True, returns a generator yielding decoded token strings.
        """
        import time

        if chat_format:
            prompt_tokens = self.tokenizer.format_chat_prompt(prompt, think=think)
        else:
            prompt_tokens = self.tokenizer.encode(prompt)

        if not prompt_tokens:
            prompt_tokens = [self.eos_1]

        if stop_tokens is None:
            stop_tokens = [self.eos_1, self.eos_2]

        def _generator():
            self.reset_cache()
            t_start = time.perf_counter()
            pos = 0

            # Prefill prompt tokens (batched SGEMM if FP32 cache is active)
            logits = self.forward_prompt(prompt_tokens, start_pos=0, compute_logits=True)
            pos = len(prompt_tokens)

            gen_tokens = []
            accumulated = ""
            for _ in range(max_tokens):
                if pos >= self.max_seq_len:
                    break

                if temperature <= 0.0 or top_k == 1:
                    next_tok = int(np.argmax(logits))
                else:
                    logits_adj = logits / temperature
                    if top_k > 0 and top_k < len(logits_adj):
                        indices = np.argpartition(logits_adj, -top_k)[-top_k:]
                        top_logits = logits_adj[indices]
                        probs = np.exp(top_logits - np.max(top_logits))
                        probs /= np.sum(probs)
                        next_tok = int(np.random.choice(indices, p=probs))
                    else:
                        probs = np.exp(logits_adj - np.max(logits_adj))
                        probs /= np.sum(probs)
                        next_tok = int(np.random.choice(len(probs), p=probs))

                if next_tok in stop_tokens:
                    break

                gen_tokens.append(next_tok)
                piece = self.tokenizer.decode([next_tok])
                accumulated += piece
                yield piece

                if stop_strings and any(s in accumulated for s in stop_strings):
                    break

                logits = self.forward_token(next_tok, pos, compute_logits=True)
                pos += 1

        if stream:
            return _generator()

        # Non-streaming collection
        t0 = time.perf_counter()
        gen_pieces = list(_generator())
        elapsed = max(time.perf_counter() - t0, 1e-6)
        text = "".join(gen_pieces)

        return GenerationOutput(
            text=text,
            tokens=[],
            prompt_tokens=len(prompt_tokens),
            generated_tokens=len(gen_pieces),
            elapsed_seconds=elapsed,
            tokens_per_second=len(gen_pieces) / elapsed,
        )

    def chat(
        self,
        user_message: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
        stream: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """High-level chat helper with ChatML formatting."""
        res = self.generate(
            user_message,
            max_tokens=max_tokens,
            temperature=temperature,
            chat_format=True,
            stream=stream,
        )
        if stream:
            return res
        return res.text

    @property
    def info(self) -> Dict[str, Any]:
        """Returns model architecture information."""
        return {
            "path": self.path,
            "version": self.version,
            "scale_format": "fp16" if self.fp16_scales else "fp32",
            "dim": self.dim,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "file_size_mb": round(self.file_size / (1024 * 1024), 2),
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
        }

    def close(self):
        """Releases memory map and file handles."""
        self.layers.clear()
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        for attr in ("emb_q", "emb_s", "final_norm"):
            if hasattr(self, attr):
                delattr(self, attr)
        if hasattr(self, "buf"):
            try:
                self.buf.release()
            except Exception:
                pass
            del self.buf
        if hasattr(self, "_mm") and self._mm:
            try:
                self._mm.close()
            except Exception:
                pass
            del self._mm
        if hasattr(self, "_f") and self._f:
            try:
                self._f.close()
            except Exception:
                pass
            del self._f
        import gc
        gc.collect()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
