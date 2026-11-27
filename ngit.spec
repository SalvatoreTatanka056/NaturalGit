# -*- mode: python ; coding: utf-8 -*-

import os


native_src = os.path.join(SPECPATH, 'llm.asm-main', 'llmasm', 'src_c')

a = Analysis(
    ['naturalgit/cli.py'],
    pathex=[SPECPATH],
    binaries=[],
    datas=[
        (os.path.join(native_src, 'llmasm_engine.c'), 'llmasm/src_c'),
        (os.path.join(native_src, 'llmasm_engine.h'), 'llmasm/src_c'),
    ],
    hiddenimports=['colorama', 'requests', 'llmasm'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ngit',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
