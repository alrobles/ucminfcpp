from setuptools import setup, Extension
import pybind11
import os

# Paths relative to the repository root (one level up from python/)
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir   = os.path.join(repo_root, "src")

ext = Extension(
    name="ucminf",
    sources=[
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "ucminf_pybind.cpp"),
        os.path.join(src_dir, "ucminf_core.cpp"),
    ],
    include_dirs=[
        src_dir,
        pybind11.get_include(),
    ],
    language="c++",
    extra_compile_args=["-std=c++17", "-O2"],
)

setup(
    name="ucminf",
    version="0.1.0",
    description="UCMINF unconstrained nonlinear optimization — Python bindings",
    long_description=open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md"),
        encoding="utf-8",
    ).read(),
    long_description_content_type="text/markdown",
    license="GPL-2.0-or-later",
    python_requires=">=3.8",
    ext_modules=[ext],
    zip_safe=False,
)
