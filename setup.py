from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
import os
from pathlib import Path


NATIVE_SOURCE_DIR = Path(__file__).parent / "llm.asm-main" / "llmasm" / "src_c"
NATIVE_SOURCE_FILES = ("llmasm_engine.c", "llmasm_engine.h")


class BuildPyWithLlmasmSources(build_py):
    def run(self):
        super().run()
        destination = Path(self.build_lib) / "llmasm" / "src_c"
        self.mkpath(str(destination))
        for filename in NATIVE_SOURCE_FILES:
            self.copy_file(
                str(NATIVE_SOURCE_DIR / filename),
                str(destination / filename),
            )

    def get_outputs(self, include_bytecode=1):
        outputs = super().get_outputs(include_bytecode)
        destination = Path(self.build_lib) / "llmasm" / "src_c"
        return outputs + [str(destination / name) for name in NATIVE_SOURCE_FILES]

    def get_source_files(self):
        return super().get_source_files() + [
            os.path.relpath(NATIVE_SOURCE_DIR / name, Path(__file__).parent).replace(os.sep, "/")
            for name in NATIVE_SOURCE_FILES
        ]

setup(
    name="naturalgit",
    version="1.0.0",
    description="Git a linea di comando guidato dal linguaggio naturale con motore AI locale llmasm",
    author="Salvatore Romano",
    author_email="tatanka056@gmail.com",
    packages=["naturalgit", "llmasm"],
    package_dir={
        "naturalgit": "naturalgit",
        "llmasm": "llm.asm-main/llmasm/llmasm",
    },
    package_data={
        "llmasm": ["src_c/*.c", "src_c/*.h"],
    },
    cmdclass={"build_py": BuildPyWithLlmasmSources},
    python_requires=">=3.8",
    install_requires=[
        "colorama>=0.4.6",
        "numpy>=1.20.0",
    ],
    entry_points={
        "console_scripts": [
            "ngit=naturalgit.cli:main",
            "naturalgit=naturalgit.cli:main",
            "llmasm=llmasm.cli:main",
        ],
    },
)
