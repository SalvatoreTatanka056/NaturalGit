"""
Unit tests for llmasm: Tokenizer, Quantization, AsmBin Model Loading,
C Struct Alignment, and Inference.
"""

import ctypes
import os
import struct
import tempfile
import unittest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmasm.tokenizer import AsmTokenizer, FIB_MULT, HT_BITS
from llmasm.converter import quantize_matrix_q8_0, align64
from llmasm.model import AsmBinModel
from llmasm.c_engine import CAsmBinHeader


class TestAsmTokenizer(unittest.TestCase):
    def setUp(self):
        # Create minimal 256 byte_to_tok and dummy hash table
        self.byte_to_tok = np.arange(256, dtype=np.uint32)
        self.tok_offsets = np.arange(257, dtype=np.uint32)
        self.tok_bytes = bytes(range(256))
        
        ht_size = 1 << HT_BITS
        self.ht_keys = np.zeros(ht_size, dtype=np.uint64)
        self.ht_merged = np.zeros(ht_size, dtype=np.uint32)
        self.ht_rank = np.full(ht_size, 0xFFFFFFFF, dtype=np.uint32)
        self.ht_mask = ht_size - 1

        self.tokenizer = AsmTokenizer(
            byte_to_tok=self.byte_to_tok,
            tok_offsets=self.tok_offsets,
            tok_bytes=self.tok_bytes,
            ht_keys=self.ht_keys,
            ht_merged=self.ht_merged,
            ht_rank=self.ht_rank,
            ht_mask=self.ht_mask,
            vocab_size=256,
        )

    def test_encode_decode_ascii(self):
        text = "Hello World"
        tokens = self.tokenizer.encode(text)
        self.assertEqual(len(tokens), len(text))
        decoded = self.tokenizer.decode(tokens)
        self.assertEqual(decoded, text)

    def test_chat_prompt_formatting(self):
        prompt = "Hello"
        tokens = self.tokenizer.format_chat_prompt(prompt, think=False)
        self.assertTrue(len(tokens) > len(prompt))
        # Ensure user start (151644) is present
        self.assertEqual(tokens[0], 151644)


class TestQuantization(unittest.TestCase):
    def test_quantize_q8_0_shape_and_stride(self):
        # 4 rows, 64 cols (multiple of 32)
        w = np.random.randn(4, 64).astype(np.float32)
        # v2: 64 int8 weights + 2 blocks * 2 bytes fp16 = 68 bytes per row
        q_bytes = quantize_matrix_q8_0(w, scale_dtype="fp16")
        expected_stride = 64 + (64 // 32) * 2
        self.assertEqual(len(q_bytes), 4 * expected_stride)

        # v1: 64 int8 weights + 2 blocks * 4 bytes fp32 = 72 bytes per row
        q_bytes_v1 = quantize_matrix_q8_0(w, scale_dtype="fp32")
        expected_stride_v1 = 64 + (64 // 32) * 4
        self.assertEqual(len(q_bytes_v1), 4 * expected_stride_v1)


class TestCStructAlignment(unittest.TestCase):
    def test_c_header_size(self):
        # CAsmBinHeader in C must match exactly 128 bytes
        self.assertEqual(ctypes.sizeof(CAsmBinHeader), 128)


class TestSyntheticModelInference(unittest.TestCase):
    def create_synthetic_asmbin(self, filename: str):
        """Creates a minimal valid .asmbin model for testing."""
        dim = 32
        hidden_dim = 64
        n_layers = 1
        n_heads = 2
        n_kv_heads = 1
        head_dim = 16
        vocab_size = 256
        max_seq_len = 64
        rms_norm_eps = 1e-6
        rope_theta = 10000.0

        with open(filename, "wb") as f:
            f.write(b"\x00" * 128)

            # Tokenizer blob
            off_tokenizer = align64(f.tell())
            f.seek(off_tokenizer)
            # 4 offsets: b2t, offs, bytes, ht
            off_b2t = 64
            off_offs = off_b2t + 256 * 4
            off_bytes = off_offs + (vocab_size + 1) * 4
            off_ht = off_bytes + 256
            f.write(struct.pack("<QQQQ", off_b2t, off_offs, off_bytes, off_ht))
            f.seek(off_tokenizer + off_b2t)
            f.write(np.arange(256, dtype="<u4").tobytes())
            f.seek(off_tokenizer + off_offs)
            f.write(np.arange(vocab_size + 1, dtype="<u4").tobytes())
            f.seek(off_tokenizer + off_bytes)
            f.write(bytes(range(256)))
            f.seek(off_tokenizer + off_ht)
            # Empty HT
            ht_size = 1 << HT_BITS
            ht_dtype = np.dtype([("key", "<u8"), ("merged_id", "<u4"), ("rank", "<u4")])
            ht = np.empty(ht_size, dtype=ht_dtype)
            ht["rank"] = 0xFFFFFFFF
            f.write(ht.tobytes())

            # Embeddings
            off_embed = align64(f.tell())
            f.seek(off_embed)
            w_emb = np.random.randn(vocab_size, dim).astype(np.float32)
            f.write(quantize_matrix_q8_0(w_emb, "fp16"))

            # Layers
            off_layers = align64(f.tell())
            f.seek(off_layers)
            l_start = f.tell()

            # Norms (dim * 4 bytes)
            f.write(np.ones(dim, dtype="<f4").tobytes())       # in_norm
            f.write(np.ones(head_dim, dtype="<f4").tobytes())  # q_norm
            f.write(np.ones(head_dim, dtype="<f4").tobytes())  # k_norm
            f.write(np.ones(dim, dtype="<f4").tobytes())       # post_norm

            # Projections
            def write_mat(rows, cols):
                pos = align64(f.tell())
                f.seek(pos)
                w = np.random.randn(rows, cols).astype(np.float32) * 0.01
                f.write(quantize_matrix_q8_0(w, "fp16"))

            q_dim = n_heads * head_dim
            kv_dim = n_kv_heads * head_dim
            write_mat(q_dim, dim)        # q_proj
            write_mat(kv_dim, dim)       # k_proj
            write_mat(kv_dim, dim)       # v_proj
            write_mat(dim, q_dim)        # o_proj
            write_mat(hidden_dim, dim)   # gate_proj
            write_mat(hidden_dim, dim)   # up_proj
            write_mat(dim, hidden_dim)   # down_proj

            layer_stride = align64(f.tell()) - l_start

            # Final norm
            off_final_norm = align64(f.tell())
            f.seek(off_final_norm)
            f.write(np.ones(dim, dtype="<f4").tobytes())

            file_size = align64(f.tell())

            # Write header
            hdr = struct.pack(
                "<4sIIIIIIIIIffIIIIQQQQQQ",
                b"ASML",
                2,
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
                0,
                0,
                ht_size - 1,
                4,  # FLAG_FP16_SCALES
                off_tokenizer,
                off_embed,
                off_layers,
                layer_stride,
                off_final_norm,
                file_size,
            )
            f.seek(0)
            f.write(hdr)

    def test_synthetic_model_loading_and_generation(self):
        with tempfile.NamedTemporaryFile(suffix=".asmbin", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self.create_synthetic_asmbin(tmp_path)
            model = AsmBinModel(tmp_path)
            self.assertEqual(model.dim, 32)
            self.assertEqual(model.n_layers, 1)

            # Test generation
            output = model.generate("Hi", max_tokens=5, temperature=0.0)
            self.assertIsInstance(output.text, str)
            self.assertEqual(output.prompt_tokens, 2)
            self.assertEqual(output.generated_tokens, 5)
            model.close()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
