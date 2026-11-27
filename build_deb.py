"""
Generates standard Linux installation packages for NaturalGit:
1. Debian/Ubuntu .deb package: dist/naturalgit_1.0.0_amd64.deb
2. Universal Linux tarball: dist/naturalgit-linux.tar.gz

Includes:
- NaturalGit CLI & engines
- llmasm Python library & C AVX2 engine (libllmasm.so)
- Documentation (GUIDA_CONFIGURAZIONE.md, README.md)
"""
import io
import os
import struct
import tarfile
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DIST_DIR = BASE_DIR / "dist"
DIST_DIR.mkdir(parents=True, exist_ok=True)

DEB_FILE = DIST_DIR / "naturalgit_1.0.0_amd64.deb"
TAR_FILE = DIST_DIR / "naturalgit-linux.tar.gz"


def create_control_tar() -> bytes:
    control_content = """Package: naturalgit
Version: 1.0.0
Section: devel
Priority: optional
Architecture: amd64
Depends: python3 (>= 3.8), python3-numpy, git, libgomp1
Maintainer: Salvatore Romano <tatanka056@gmail.com>
Homepage: https://github.com/SalvatoreTatanka056/NaturalGit
Description: Controllo versione Git guidato da linguaggio naturale con motore AI locale llmasm
 NaturalGit consente di eseguire comandi e flussi Git usando frasi
 in italiano o inglese, con motore euristico locale ultra-veloce,
 motore di inferenza locale C/Assembly llmasm offline e
 integrazione con Google Gemini, Ollama e OpenAI.
"""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        ti = tarfile.TarInfo("control")
        ti.size = len(control_content.encode("utf-8"))
        ti.mtime = int(time.time())
        ti.mode = 0o644
        tar.addfile(ti, io.BytesIO(control_content.encode("utf-8")))
    return buf.getvalue()


def create_data_tar() -> bytes:
    buf = io.BytesIO()
    pkg_src = BASE_DIR / "naturalgit"
    llmasm_src = BASE_DIR / "llm.asm-main" / "llmasm" / "llmasm"
    src_c_dir = BASE_DIR / "llm.asm-main" / "llmasm" / "src_c"

    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        # 1. Base directories
        dirs = [
            "usr",
            "usr/bin",
            "usr/lib",
            "usr/lib/python3",
            "usr/lib/python3/dist-packages",
            "usr/lib/python3/dist-packages/naturalgit",
            "usr/lib/python3/dist-packages/llmasm",
            "usr/share",
            "usr/share/doc",
            "usr/share/doc/naturalgit",
        ]
        for d in dirs:
            ti = tarfile.TarInfo(d)
            ti.type = tarfile.DIRTYPE
            ti.mode = 0o755
            ti.mtime = int(time.time())
            tar.addfile(ti)

        # 2. Python package files (naturalgit)
        for py_file in pkg_src.glob("*.py"):
            with open(py_file, "rb") as f:
                content = f.read()
            ti = tarfile.TarInfo(f"usr/lib/python3/dist-packages/naturalgit/{py_file.name}")
            ti.size = len(content)
            ti.mtime = int(time.time())
            ti.mode = 0o644
            tar.addfile(ti, io.BytesIO(content))

        # 3. Python package files (llmasm)
        if llmasm_src.exists():
            for py_file in llmasm_src.glob("*.py"):
                with open(py_file, "rb") as f:
                    content = f.read()
                ti = tarfile.TarInfo(f"usr/lib/python3/dist-packages/llmasm/{py_file.name}")
                ti.size = len(content)
                ti.mtime = int(time.time())
                ti.mode = 0o644
                tar.addfile(ti, io.BytesIO(content))

        # 4. Compiled C library (libllmasm.so)
        so_file = None
        for candidate in [src_c_dir / "libllmasm.so", llmasm_src / "libllmasm.so"]:
            if candidate.exists():
                so_file = candidate
                break

        if so_file and so_file.exists():
            with open(so_file, "rb") as f:
                so_content = f.read()
            # In llmasm directory
            ti = tarfile.TarInfo("usr/lib/python3/dist-packages/llmasm/libllmasm.so")
            ti.size = len(so_content)
            ti.mtime = int(time.time())
            ti.mode = 0o755
            tar.addfile(ti, io.BytesIO(so_content))

            # In global /usr/lib
            ti2 = tarfile.TarInfo("usr/lib/libllmasm.so")
            ti2.size = len(so_content)
            ti2.mtime = int(time.time())
            ti2.mode = 0o755
            tar.addfile(ti2, io.BytesIO(so_content))

        # 5. Documentation
        for doc_name in ["README.md", "GUIDA_CONFIGURAZIONE.md"]:
            doc_path = BASE_DIR / doc_name
            if doc_path.exists():
                with open(doc_path, "rb") as f:
                    doc_content = f.read()
                ti = tarfile.TarInfo(f"usr/share/doc/naturalgit/{doc_name}")
                ti.size = len(doc_content)
                ti.mtime = int(time.time())
                ti.mode = 0o644
                tar.addfile(ti, io.BytesIO(doc_content))

        # 6. Executable launchers
        launcher_ngit = """#!/usr/bin/env python3
import sys
sys.path.insert(0, '/usr/lib/python3/dist-packages')
from naturalgit.cli import main
if __name__ == '__main__':
    main()
"""
        launcher_llmasm = """#!/usr/bin/env python3
import sys
sys.path.insert(0, '/usr/lib/python3/dist-packages')
from llmasm.cli import main
if __name__ == '__main__':
    main()
"""
        for bin_name in ["ngit", "naturalgit"]:
            ti = tarfile.TarInfo(f"usr/bin/{bin_name}")
            ti.size = len(launcher_ngit.encode("utf-8"))
            ti.mtime = int(time.time())
            ti.mode = 0o755
            tar.addfile(ti, io.BytesIO(launcher_ngit.encode("utf-8")))

        ti = tarfile.TarInfo("usr/bin/llmasm")
        ti.size = len(launcher_llmasm.encode("utf-8"))
        ti.mtime = int(time.time())
        ti.mode = 0o755
        tar.addfile(ti, io.BytesIO(launcher_llmasm.encode("utf-8")))

    return buf.getvalue()


def write_ar_archive(dest_path: Path, files: list):
    """Writes standard Unix ar archive format (used by .deb)."""
    with open(dest_path, "wb") as f:
        # Global magic
        f.write(b"!<arch>\n")
        for name, data in files:
            size = len(data)
            header = struct.pack(
                "16s12s6s6s8s10s2s",
                name.encode("ascii").ljust(16),
                str(int(time.time())).encode("ascii").ljust(12),
                b"0".ljust(6),
                b"0".ljust(6),
                b"100644".ljust(8),
                str(size).encode("ascii").ljust(10),
                b"`\n"
            )
            f.write(header)
            f.write(data)
            if size % 2 != 0:
                f.write(b"\n")


def build_deb():
    print("📦 Creazione pacchetto Debian/Ubuntu (.deb)...")
    deb_binary = b"2.0\n"
    control_tar = create_control_tar()
    data_tar = create_data_tar()

    files = [
        ("debian-binary", deb_binary),
        ("control.tar.gz", control_tar),
        ("data.tar.gz", data_tar)
    ]

    write_ar_archive(DEB_FILE, files)
    print(f"✔ Pacchetto Debian creato: {DEB_FILE} ({DEB_FILE.stat().st_size:,} bytes)")


def build_tarball():
    print("📦 Creazione archivio Linux universale (.tar.gz)...")
    root_name = "naturalgit-1.0.0"

    with tarfile.open(TAR_FILE, "w:gz") as tar:
        # Add root dir
        ti = tarfile.TarInfo(root_name)
        ti.type = tarfile.DIRTYPE
        ti.mode = 0o755
        ti.mtime = int(time.time())
        tar.addfile(ti)

        # Add single files
        for fname in ["install.sh", "setup.py", "README.md", "GUIDA_CONFIGURAZIONE.md"]:
            fpath = BASE_DIR / fname
            if fpath.exists():
                mode = 0o755 if fname.endswith(".sh") else 0o644
                if fname.endswith(".sh"):
                    content = fpath.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
                    ti = tar.gettarinfo(str(fpath), arcname=f"{root_name}/{fname}")
                    ti.mode = mode
                    ti.size = len(content)
                    tar.addfile(ti, io.BytesIO(content))
                else:
                    tar.add(fpath, arcname=f"{root_name}/{fname}", filter=lambda t, m=mode: _set_mode(t, m))

        # Add naturalgit directory
        tar.add(
            BASE_DIR / "naturalgit",
            arcname=f"{root_name}/naturalgit",
            filter=lambda t: None if "__pycache__" in t.name or t.name.endswith(".pyc") else t
        )

        # Add llm.asm-main/llmasm directory
        llmasm_dir = BASE_DIR / "llm.asm-main" / "llmasm"
        if llmasm_dir.exists():
            tar.add(
                llmasm_dir,
                arcname=f"{root_name}/llm.asm-main/llmasm",
                filter=lambda t: None if "__pycache__" in t.name or t.name.endswith(".pyc") else t
            )

    print(f"✔ Archivio tar.gz creato: {TAR_FILE} ({TAR_FILE.stat().st_size:,} bytes)")


def _set_mode(tarinfo, mode):
    tarinfo.mode = mode
    return tarinfo


def build():
    build_deb()
    build_tarball()


if __name__ == "__main__":
    build()
