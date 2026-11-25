from setuptools import setup, find_packages

setup(
    name="naturalgit",
    version="1.0.0",
    description="Git a linea di comando guidato dal linguaggio naturale",
    author="NaturalGit Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "colorama>=0.4.6",
    ],
    entry_points={
        "console_scripts": [
            "ngit=naturalgit.cli:main",
            "naturalgit=naturalgit.cli:main",
        ],
    },
)
