"""
Auto-detection and compilation helper for the C inference engine.
Locates pre-compiled .dll / .so / .dylib or builds it dynamically if a C compiler is available.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

SRC_C_DIR = Path(__file__).resolve().parent.parent / "src_c"


def get_shared_lib_name() -> str:
    """Returns platform-specific shared library filename."""
    if sys.platform == "win32":
        return "llmasm.dll"
    elif sys.platform == "darwin":
        return "libllmasm.dylib"
    else:
        return "libllmasm.so"


def find_compiled_lib() -> Optional[Path]:
    """Finds existing compiled C library in package dirs, src_c, or system lib paths."""
    target_name = get_shared_lib_name()
    candidates = [
        Path(__file__).resolve().parent / target_name,
        SRC_C_DIR / target_name,
        Path.cwd() / target_name,
        Path("/usr/lib") / target_name,
        Path("/usr/local/lib") / target_name,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def build_c_library(verbose: bool = True) -> Optional[Path]:
    """Compiles llmasm_engine.c into a shared library using available compilers."""
    target_path = Path(__file__).resolve().parent / get_shared_lib_name()
    src_c_file = SRC_C_DIR / "llmasm_engine.c"

    if not src_c_file.exists():
        if verbose:
            print(f"[llmasm] Source file not found: {src_c_file}")
        return None

    # Try gcc
    gcc = shutil.which("gcc")
    if gcc:
        cmd = [
            gcc, "-O3", "-shared", "-fPIC",
            "-mavx2", "-mfma", "-mf16c",
            str(src_c_file), "-o", str(target_path)
        ]
        if sys.platform != "win32" and sys.platform != "darwin":
            cmd.append("-lm")
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0 and target_path.exists():
                if verbose:
                    print(f"[llmasm] Successfully compiled C engine with gcc -> {target_path}")
                return target_path
        except Exception:
            pass

    # Try clang
    clang = shutil.which("clang")
    if clang:
        cmd = [
            clang, "-O3", "-shared", "-fPIC",
            "-mavx2", "-mfma", "-mf16c",
            str(src_c_file), "-o", str(target_path)
        ]
        if sys.platform != "win32" and sys.platform != "darwin":
            cmd.append("-lm")
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0 and target_path.exists():
                if verbose:
                    print(f"[llmasm] Successfully compiled C engine with clang -> {target_path}")
                return target_path
        except Exception:
            pass

    # Try MSVC (cl.exe) on Windows
    cl = shutil.which("cl")
    if cl:
        cmd = [
            cl, "/O2", "/LD", "/arch:AVX2", "/DLL",
            f"/Fe:{target_path}", str(src_c_file)
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0 and target_path.exists():
                if verbose:
                    print(f"[llmasm] Successfully compiled C engine with MSVC cl -> {target_path}")
                return target_path
        except Exception:
            pass

    return None
