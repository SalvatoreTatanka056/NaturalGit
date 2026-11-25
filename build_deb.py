"""
Generates a standard Debian/Ubuntu .deb package for NaturalGit in pure Python.
Output: dist/naturalgit_1.0.0_all.deb
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

DEB_FILE = DIST_DIR / "naturalgit_1.0.0_all.deb"


def create_control_tar() -> bytes:
    control_content = """Package: naturalgit
Version: 1.0.0
Section: devel
Priority: optional
Architecture: all
Depends: python3 (>= 3.8), git
Maintainer: NaturalGit Team <info@naturalgit.org>
Homepage: https://github.com/SalvatoreTatanka056/NaturalGit
Description: Controllo versione Git guidato da linguaggio naturale
 NaturalGit consente di eseguire comandi e flussi Git usando frasi
 in italiano o inglese, con motore euristico locale ultra-veloce e
 integrazione con intelligenza artificiale Google Gemini.
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

    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        # 1. Base directories
        for d in ["usr", "usr/bin", "usr/lib", "usr/lib/python3", "usr/lib/python3/dist-packages", "usr/lib/python3/dist-packages/naturalgit"]:
            ti = tarfile.TarInfo(d)
            ti.type = tarfile.DIRTYPE
            ti.mode = 0o755
            ti.mtime = int(time.time())
            tar.addfile(ti)

        # 2. Python package files
        for py_file in pkg_src.glob("*.py"):
            with open(py_file, "rb") as f:
                content = f.read()
            ti = tarfile.TarInfo(f"usr/lib/python3/dist-packages/naturalgit/{py_file.name}")
            ti.size = len(content)
            ti.mtime = int(time.time())
            ti.mode = 0o644
            tar.addfile(ti, io.BytesIO(content))

        # 3. Executable launcher /usr/bin/ngit
        launcher_content = """#!/usr/bin/env python3
import sys
sys.path.insert(0, '/usr/lib/python3/dist-packages')
from naturalgit.cli import main
if __name__ == '__main__':
    main()
"""
        for bin_name in ["ngit", "naturalgit"]:
            ti = tarfile.TarInfo(f"usr/bin/{bin_name}")
            ti.size = len(launcher_content.encode("utf-8"))
            ti.mtime = int(time.time())
            ti.mode = 0o755
            tar.addfile(ti, io.BytesIO(launcher_content.encode("utf-8")))

    return buf.getvalue()


def write_ar_archive(dest_path: Path, files: list):
    """Writes standard Unix ar archive format (used by .deb)."""
    with open(dest_path, "wb") as f:
        # Global magic
        f.write(b"!<arch>\n")
        for name, data in files:
            # Header format: 16s 12s 6s 6s 8s 10s 2s
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


def build():
    print("📦 Creazione pacchetto Debian (.deb)...")
    deb_binary = b"2.0\n"
    control_tar = create_control_tar()
    data_tar = create_data_tar()

    files = [
        ("debian-binary", deb_binary),
        ("control.tar.gz", control_tar),
        ("data.tar.gz", data_tar)
    ]

    write_ar_archive(DEB_FILE, files)
    print(f"✔ Pacchetto Debian creato: {DEB_FILE} ({DEB_FILE.stat().st_size} bytes)")


if __name__ == "__main__":
    build()
