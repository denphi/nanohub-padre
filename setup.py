"""
Setup file for PyPADRE - Python library for PADRE semiconductor device simulator.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read() if __import__("os").path.exists("README.md") else ""

setup(
    name="pypadre",
    version="0.1.0",
    author="",
    author_email="",
    description="Python library for PADRE semiconductor device simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[],
    extras_require={
        "dev": ["pytest", "black", "mypy"],
    },
)
