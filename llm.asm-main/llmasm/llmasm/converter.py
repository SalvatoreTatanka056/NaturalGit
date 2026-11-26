"""
Model weight conversion and exporter for llm.asm format (.asmbin).
Converts Hugging Face safetensors (e.g. Qwen3-0.6B) into 64-byte aligned,
per-row split Q8_0 weights with pre-built Fibonacci hash table tokenizer.
"""

import json
import os
import struct
import time
from typing import Callable, Optional
import numpy as np

MAGIC = b"ASML"
VERSION_V1 = 1
VERSION_V2 = 2
FLAG_TIED_EMB = 1
FLAG_QK_NORM = 2
FLAG_FP16_SCALES = 4
HEADER_SIZE = 128
TOK_HEADER_SIZE = 64
HT_BITS = 19
HT_SIZE = 1 << HT_BITS  # 524,288 slots
HT_MASK = HT_SIZE - 1
FIB_MULT = 0x9E3779B97F4A7C15


def align64(n: int) -> int:
    return (n + 63) & ~63


def pad64(f) -> int:
    pos = f.tell()
    aligned = align64(pos)
    if aligned > pos:
        f.write(b"\x00" * (aligned - pos))
    return aligned


def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def quantize_matrix_q8_0(w: np.ndarray, scale_dtype: str = "fp16") -> bytes:
    """
    Quantizes [M, K] float32 matrix into per-row split Q8_0 bytes.
    Matches AVX2 kernel memory layout.
    """
    assert w.ndim == 2
    M, K = w.shape
    assert K % 32 == 0, f"K={K} must be a multiple of 32"
    nb = K // 32
    blocks = w.reshape(M, nb, 32).astype(np.float32, copy=False)
    amax = np.max(np.abs(blocks), axis=2)  # [M, nb]
    scale_f32 = np.where(amax > 0.0, amax / np.float32(127.0), np.float32(0.0)).astype(np.float32)

    if scale_dtype == "fp16":
        scale_f16 = scale_f32.astype(np.float16)
        scale_eff = scale_f16.astype(np.float32)
        inv_scale = np.where(scale_eff > 0.0, np.float32(1.0) / scale_eff, np.float32(0.0)).astype(np.float32)
        q = np.clip(np.rint(blocks * inv_scale[:, :, None]), -127, 127).astype(np.int8)
        s_bytes = scale_f16.astype("<f2").view(np.uint8).reshape(M, nb * 2)
        scale_bytes_per_block = 2
    elif scale_dtype == "fp32":
        inv_scale = np.where(scale_f32 > 0.0, np.float32(1.0) / scale_f32, np.float32(0.0)).astype(np.float32)
        q = np.clip(np.rint(blocks * inv_scale[:, :, None]), -127, 127).astype(np.int8)
        s_bytes = scale_f32.astype("<f4").view(np.uint8).reshape(M, nb * 4)
        scale_bytes_per_block = 4
    else:
        raise ValueError(f"Unsupported scale_dtype: {scale_dtype}")

    q_flat = q.reshape(M, K)
    row_packed = np.concatenate([q_flat.view(np.uint8), s_bytes], axis=1)
    return row_packed.tobytes()


def build_tokenizer_blob(tok_json_path: str, vocab_size: int) -> bytes:
    """Builds serialized tokenizer block with open-addressing Fibonacci merge table."""
    with open(tok_json_path, "r", encoding="utf-8") as f:
        tj = json.load(f)

    vocab = tj["model"]["vocab"]
    merges = tj["model"]["merges"]
    added_tokens = tj.get("added_tokens", [])

    b2u = bytes_to_unicode()
    u2b = {v: k for k, v in b2u.items()}

    # 1. byte_to_tok[256]
    byte_to_tok = np.array([vocab[b2u[b]] for b in range(256)], dtype=np.uint32)

    # 2. Raw decoded UTF-8 byte string for every token ID
    tok_raw = [b""] * vocab_size
    for piece, tid in vocab.items():
        if tid < vocab_size:
            tok_raw[tid] = bytes(u2b[ch] for ch in piece)
    for at in added_tokens:
        tid = at["id"]
        if tid < vocab_size:
            tok_raw[tid] = at["content"].encode("utf-8")

    tok_offsets = np.zeros(vocab_size + 1, dtype=np.uint32)
    chunks = []
    cur = 0
    for tid in range(vocab_size):
        b = tok_raw[tid]
        tok_offsets[tid] = cur
        chunks.append(b)
        cur += len(b)
    tok_offsets[vocab_size] = cur
    tok_bytes = b"".join(chunks)

    # 3. Pre-built Fibonacci-hashed merge table
    ht_keys = np.zeros(HT_SIZE, dtype=np.uint64)
    ht_merged = np.zeros(HT_SIZE, dtype=np.uint32)
    ht_rank = np.full(HT_SIZE, 0xFFFFFFFF, dtype=np.uint32)

    for rank, m in enumerate(merges):
        if isinstance(m, list):
            a, b = m[0], m[1]
        else:
            a, b = m.split(" ")
        ia = vocab[a]
        ib = vocab[b]
        ic = vocab[a + b]
        key = (ia << 32) | ib
        slot = ((key * FIB_MULT) & 0xFFFFFFFFFFFFFFFF) >> (64 - HT_BITS)
        while ht_rank[slot] != 0xFFFFFFFF:
            slot = (slot + 1) & HT_MASK
        ht_keys[slot] = key
        ht_merged[slot] = ic
        ht_rank[slot] = rank

    ht_dtype = np.dtype([("key", "<u8"), ("merged_id", "<u4"), ("rank", "<u4")])
    ht_arr = np.empty(HT_SIZE, dtype=ht_dtype)
    ht_arr["key"] = ht_keys
    ht_arr["merged_id"] = ht_merged
    ht_arr["rank"] = ht_rank
    ht_bytes = ht_arr.tobytes()

    off_byte_to_tok = TOK_HEADER_SIZE
    off_tok_offsets = align64(off_byte_to_tok + byte_to_tok.nbytes)
    off_tok_bytes = align64(off_tok_offsets + tok_offsets.nbytes)
    off_merge_ht = align64(off_tok_bytes + len(tok_bytes))
    total_tok_size = align64(off_merge_ht + len(ht_bytes))

    buf = bytearray(total_tok_size)
    struct.pack_into("<QQQQ", buf, 0, off_byte_to_tok, off_tok_offsets, off_tok_bytes, off_merge_ht)
    buf[off_byte_to_tok : off_byte_to_tok + byte_to_tok.nbytes] = byte_to_tok.tobytes()
    buf[off_tok_offsets : off_tok_offsets + tok_offsets.nbytes] = tok_offsets.tobytes()
    buf[off_tok_bytes : off_tok_bytes + len(tok_bytes)] = tok_bytes
    buf[off_merge_ht : off_merge_ht + len(ht_bytes)] = ht_bytes
    return bytes(buf)


def load_safetensors_header(path: str):
    with open(path, "rb") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hdr_len))
    data_offset = 8 + hdr_len
    return hdr, data_offset


def read_tensor_f32(f, hdr, data_offset: int, name: str) -> np.ndarray:
    meta = hdr[name]
    dtype = meta["dtype"]
    shape = tuple(meta["shape"])
    start, end = meta["data_offsets"]
    f.seek(data_offset + start)
    raw = f.read(end - start)
    if dtype == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16)
        f32 = (u16.astype(np.uint32) << 16).view(np.float32)
        return f32.reshape(shape)
    elif dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
    elif dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
    else:
        raise ValueError(f"Unsupported dtype {dtype} for tensor {name}")


def export_to_asmbin(
    model_dir: str,
    out_path: str,
    max_seq_len: int = 4096,
    scale_dtype: str = "fp16",
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> str:
    """
    Exports Hugging Face model directory to .asmbin file.
    """
    t0 = time.time()
    use_fp16 = scale_dtype == "fp16"
    version = VERSION_V2 if use_fp16 else VERSION_V1

    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {model_dir}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    dim = int(cfg["hidden_size"])
    hidden_dim = int(cfg["intermediate_size"])
    n_layers = int(cfg["num_hidden_layers"])
    n_heads = int(cfg["num_attention_heads"])
    n_kv_heads = int(cfg["num_key_value_heads"])
    head_dim = int(cfg.get("head_dim", dim // n_heads))
    vocab_size = int(cfg["vocab_size"])
    rms_norm_eps = float(cfg.get("rms_norm_eps", 1e-6))
    rope_theta = float(cfg.get("rope_theta", 1000000.0))
    tie_emb = bool(cfg.get("tie_word_embeddings", True))
    eos_1 = 151643
    eos_2 = 151645

    if progress_callback:
        progress_callback("Building tokenizer blob", 0.05)

    tok_path = os.path.join(model_dir, "tokenizer.json")
    tok_blob = build_tokenizer_blob(tok_path, vocab_size)

    # Locate safetensors
    st_path = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(st_path):
        for fname in os.listdir(model_dir):
            if fname.endswith(".safetensors"):
                st_path = os.path.join(model_dir, fname)
                break

    st_file = open(st_path, "rb")
    st_hdr, data_offset = load_safetensors_header(st_path)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out_file = open(out_path, "wb")
    out_file.write(b"\x00" * HEADER_SIZE)

    off_tokenizer = pad64(out_file)
    out_file.write(tok_blob)

    off_embed = pad64(out_file)
    if progress_callback:
        progress_callback("Quantizing embedding", 0.15)
    w_emb = read_tensor_f32(st_file, st_hdr, data_offset, "model.embed_tokens.weight")
    out_file.write(quantize_matrix_q8_0(w_emb, scale_dtype))

    off_layers = pad64(out_file)
    layer_starts = []

    for l in range(n_layers):
        if progress_callback:
            progress_callback(f"Quantizing layer {l+1}/{n_layers}", 0.2 + (0.75 * l / n_layers))
        layer_start = pad64(out_file)
        layer_starts.append(layer_start)
        pfx = f"model.layers.{l}."

        in_norm = read_tensor_f32(st_file, st_hdr, data_offset, pfx + "input_layernorm.weight")
        out_file.write(in_norm.astype("<f4").tobytes())

        q_norm = read_tensor_f32(st_file, st_hdr, data_offset, pfx + "self_attn.q_norm.weight")
        out_file.write(q_norm.astype("<f4").tobytes())

        k_norm = read_tensor_f32(st_file, st_hdr, data_offset, pfx + "self_attn.k_norm.weight")
        out_file.write(k_norm.astype("<f4").tobytes())

        post_norm = read_tensor_f32(st_file, st_hdr, data_offset, pfx + "post_attention_layernorm.weight")
        out_file.write(post_norm.astype("<f4").tobytes())

        for proj in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
            pad64(out_file)
            sub = "self_attn." if "proj" in proj and "gate" not in proj and "up" not in proj and "down" not in proj else "mlp."
            w = read_tensor_f32(st_file, st_hdr, data_offset, pfx + sub + proj + ".weight")
            out_file.write(quantize_matrix_q8_0(w, scale_dtype))

    layer_stride = layer_starts[1] - layer_starts[0] if len(layer_starts) > 1 else (out_file.tell() - layer_starts[0])

    off_final_norm = pad64(out_file)
    fn = read_tensor_f32(st_file, st_hdr, data_offset, "model.norm.weight")
    out_file.write(fn.astype("<f4").tobytes())

    file_size = pad64(out_file)

    flags = (FLAG_TIED_EMB if tie_emb else 0) | FLAG_QK_NORM | (FLAG_FP16_SCALES if use_fp16 else 0)

    # Write final header at offset 0
    hdr = struct.pack(
        "<4sIIIIIIIIIffIIIIQQQQQQ",
        MAGIC,
        version,
        dim,
        hidden_dim,
        n_layers,
        n_heads,
        n_kv_heads,
        head_dim,
        vocab_size,
        max_seq_len,
        rms_norm_eps,
        rope_theta,
        eos_1,
        eos_2,
        HT_MASK,
        flags,
        off_tokenizer,
        off_embed,
        off_layers,
        layer_stride,
        off_final_norm,
        file_size,
    )
    hdr_padded = hdr + b"\x00" * (HEADER_SIZE - len(hdr))
    out_file.seek(0)
    out_file.write(hdr_padded)

    st_file.close()
    out_file.close()

    if progress_callback:
        progress_callback("Finished export", 1.0)

    return out_path
