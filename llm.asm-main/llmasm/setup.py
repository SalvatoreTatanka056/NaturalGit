from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
import os
from pathlib import Path

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


NATIVE_SOURCE_DIR = Path(__file__).parent / "src_c"
NATIVE_SOURCE_FILES = ("llmasm_engine.c", "llmasm_engine.h")


class BuildPyWithNativeSources(build_py):
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
            f"src_c/{name}" for name in NATIVE_SOURCE_FILES
        ]


setup(
    name="llmasm",
    version="1.0.0",
    description="Python library and high-performance C inference engine for llm.asm models (.asmbin)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Salvatore Romano",
    author_email="tatanka056@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "llmasm": ["src_c/*.c", "src_c/*.h"],
    },
    cmdclass={"build_py": BuildPyWithNativeSources},
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "converter": ["torch", "safetensors"],
    },
    entry_points={
        "console_scripts": [
            "llmasm = llmasm.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
