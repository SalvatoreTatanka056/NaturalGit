"""
BPE Tokenizer compatible with llm.asm binary format (.asmbin).
Implements Fibonacci open-addressing hash table lookup matching the pure x86-64 NASM tokenizer.
"""

from typing import List, Optional, Tuple, Union
import numpy as np

FIB_MULT = 0x9E3779B97F4A7C15
HT_BITS = 19


class AsmTokenizer:
    """Byte-pair encoding (BPE) tokenizer used by llm.asm."""

    def __init__(
        self,
        byte_to_tok: np.ndarray,
        tok_offsets: np.ndarray,
        tok_bytes: bytes,
        ht_keys: np.ndarray,
        ht_merged: np.ndarray,
        ht_rank: np.ndarray,
        ht_mask: int,
        vocab_size: int,
    ):
        self.byte_to_tok = byte_to_tok
        self.tok_offsets = tok_offsets
        self.tok_bytes = tok_bytes
        self.ht_keys = ht_keys
        self.ht_merged = ht_merged
        self.ht_rank = ht_rank
        self.ht_mask = ht_mask
        self.vocab_size = vocab_size

    def ht_lookup(self, left_id: int, right_id: int) -> Optional[Tuple[int, int]]:
        """Look up pair merge in the open-addressing Fibonacci hash table."""
        key = (left_id << 32) | right_id
        slot = ((key * FIB_MULT) & 0xFFFFFFFFFFFFFFFF) >> (64 - HT_BITS)
        while True:
            r = int(self.ht_rank[slot])
            if r == 0xFFFFFFFF:
                return None
            if int(self.ht_keys[slot]) == key:
                return int(self.ht_merged[slot]), r
            slot = (slot + 1) & self.ht_mask

    def encode(self, text: str) -> List[int]:
        """Encode text string into BPE token IDs."""
        raw = text.encode("utf-8")
        if not raw:
            return []
        toks = [int(self.byte_to_tok[b]) for b in raw]
        while len(toks) > 1:
            best_rank = 0xFFFFFFFF
            best_i = -1
            best_c = -1
            for i in range(len(toks) - 1):
                res = self.ht_lookup(toks[i], toks[i + 1])
                if res is not None and res[1] < best_rank:
                    best_c, best_rank = res
                    best_i = i
            if best_i < 0:
                break
            toks[best_i] = best_c
            del toks[best_i + 1]
        return toks

    def decode(self, token_ids: List[int]) -> str:
        """Decode a sequence of token IDs back into a UTF-8 string."""
        out = bytearray()
        for tid in token_ids:
            if 0 <= tid < self.vocab_size:
                a = int(self.tok_offsets[tid])
                b = int(self.tok_offsets[tid + 1])
                out.extend(self.tok_bytes[a:b])
        return out.decode("utf-8", errors="replace")

    def decode_token(self, token_id: int) -> str:
        """Decode a single token ID."""
        return self.decode([token_id])

    def format_chat_prompt(self, user_prompt: str, think: bool = False) -> List[int]:
        """
        Formats user prompt into ChatML / Qwen3 template token IDs:
        <|im_start|>user\n + BPE(user_prompt) + <|im_end|>\n<|im_start|>assistant\n
        Optional think block bypass: <think>\\n\\n</think>\\n\\n
        """
        # <|im_start|> (151644) user (872) \n (198)
        ids = [151644, 872, 198] + self.encode(user_prompt) + [151645, 198, 151644, 77091, 198]
        if not think:
            # Bypass thinking tokens: <think>\n\n</think>\n\n
            ids += [151667, 271, 151668, 271]
        return ids
