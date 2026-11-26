from setuptools import setup, find_packages
import os

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

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
        "llmasm": ["*.dll", "*.so", "*.dylib"],
    },
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
