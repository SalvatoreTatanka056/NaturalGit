"""
llmasm - Python library & C runtime for llm.asm models (.asmbin format)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Native C Inference Engine (AVX2+FMA SIMD via ctypes)
- Pure Python & NumPy Fallback Runtime
- Byte-level BPE Tokenizer (Fibonacci Hash Table)
- Safetensors / HuggingFace to .asmbin Exporter & Quantizer
"""

from typing import Optional, Union
from pathlib import Path

from .tokenizer import AsmTokenizer
from .converter import export_to_asmbin
from .model import AsmBinModel, GenerationOutput
from .c_engine import CAsmBinModel, CLibBinding, CAsmBinHeader
from .c_builder import build_c_library, find_compiled_lib

__version__ = "1.0.0"
__all__ = [
    "load_model",
    "AsmBinModel",
    "CAsmBinModel",
    "GenerationOutput",
    "AsmTokenizer",
    "export_to_asmbin",
    "CLibBinding",
    "build_c_library",
    "find_compiled_lib",
]


def load_model(
    model_path: Union[str, Path],
    backend: str = "auto",
    threads: int = 4,
    lib_path: Optional[Union[str, Path]] = None,
) -> Union[CAsmBinModel, AsmBinModel]:
    """
    Loads an .asmbin model using the desired backend:
    - 'auto': Attempts to use the native C engine; falls back to NumPy if C engine is uncompiled.
    - 'c': Enforces the native C inference engine (libllmasm).
    - 'numpy': Uses the pure Python/NumPy inference engine.
    - 'threads': Number of CPU worker threads (default 4).
    """
    model_path_str = str(model_path)
    backend = backend.lower()

    if backend in ("c", "auto"):
        try:
            return CAsmBinModel(model_path_str, lib_path=lib_path, threads=threads)
        except Exception:
            # If C engine is not yet compiled, gracefully fallback to NumPy
            pass

    return AsmBinModel(model_path_str, threads=threads)
