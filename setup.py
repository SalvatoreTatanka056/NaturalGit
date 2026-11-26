from setuptools import setup, find_packages

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
        "llmasm": ["*.dll", "*.so", "*.dylib"],
    },
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
