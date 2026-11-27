"""
Auto-detection and compilation helper for the C inference engine.
Locates pre-compiled .dll / .so / .dylib or builds it dynamically if a C compiler is available.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

PACKAGE_DIR = Path(__file__).resolve().parent
SRC_C_DIR = PACKAGE_DIR / "src_c"
if not SRC_C_DIR.is_dir():
    SRC_C_DIR = PACKAGE_DIR.parent / "src_c"


def get_build_dir() -> Path:
    """Returns a writable per-user directory for the compiled native library."""
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "NaturalGit" / "llmasm"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "llmasm"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "llmasm"


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
        PACKAGE_DIR / target_name,
        SRC_C_DIR / target_name,
        get_build_dir() / target_name,
        Path.cwd() / target_name,
        Path("/usr/lib") / target_name,
        Path("/usr/local/lib") / target_name,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _run_compiler(
    name: str, cmd: List[str], target_path: Path
) -> Tuple[Optional[Path], Optional[str]]:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as exc:
        return None, f"{name}: {exc}"

    if result.returncode == 0 and target_path.exists():
        return target_path, None

    detail = (result.stderr or result.stdout).strip()
    if not detail:
        detail = f"compiler exited with status {result.returncode}"
    return None, f"{name}: {detail}"


def build_c_library(verbose: bool = True) -> Optional[Path]:
    """Compiles llmasm_engine.c into a shared library using available compilers."""
    target_path = get_build_dir() / get_shared_lib_name()
    src_c_file = SRC_C_DIR / "llmasm_engine.c"

    if not src_c_file.exists():
        if verbose:
            print(f"[llmasm] Source file not found: {src_c_file}")
        return None

    target_path.parent.mkdir(parents=True, exist_ok=True)
    commands = []

    gcc = shutil.which("gcc")
    if gcc:
        cmd = [
            gcc, "-O3", "-shared", "-fPIC", "-fopenmp",
            "-mavx2", "-mfma", "-mf16c",
            str(src_c_file), "-o", str(target_path)
        ]
        if sys.platform != "win32" and sys.platform != "darwin":
            cmd.append("-lm")
        commands.append(("gcc", cmd))

    clang = shutil.which("clang")
    if clang:
        cmd = [
            clang, "-O3", "-shared", "-fPIC", "-fopenmp",
            "-mavx2", "-mfma", "-mf16c",
            str(src_c_file), "-o", str(target_path)
        ]
        if sys.platform != "win32" and sys.platform != "darwin":
            cmd.append("-lm")
        commands.append(("clang", cmd))

    cl = shutil.which("cl")
    if cl:
        cmd = [
            cl, "/O2", "/LD", "/openmp", "/arch:AVX2",
            str(src_c_file), "/link", f"/OUT:{target_path}"
        ]
        commands.append(("MSVC cl", cmd))

    failures = []
    for name, cmd in commands:
        compiled, error = _run_compiler(name, cmd, target_path)
        if compiled:
            if verbose:
                print(f"[llmasm] Successfully compiled C engine with {name} -> {target_path}")
            return compiled
        if error:
            failures.append(error)

    if verbose:
        if not commands:
            print("[llmasm] No supported compiler found (gcc, clang, or cl.exe).")
        else:
            print("[llmasm] Native C build failed:")
            for failure in failures:
                print(f"  {failure}")

    return None
