import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import numpy as np
from setuptools import Extension, setup

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = Path(__file__).resolve().parent
CPP_DIR = SRC_DIR / "cpp"
CPP_BUILD_DIR = CPP_DIR / "build"

FAISS_ROOT = ROOT / "ACORN"
FAISS_INCLUDE = FAISS_ROOT
FAISS_LIB_DIR = FAISS_ROOT / "build" / "faiss"


def run_cmd(cmd, cwd=None):
    print("[setup]", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def build_cpp_tools():
    """Build build_indexes and search_filters in src/cpp/build."""
    if os.environ.get("FVS_SKIP_CPP_TOOLS") == "1":
        print("[setup] Skipping C++ tool build (FVS_SKIP_CPP_TOOLS=1)")
        return

    CPP_BUILD_DIR.mkdir(parents=True, exist_ok=True)

    cmake = shutil.which("cmake")
    if cmake:
        run_cmd([cmake, "-S", str(CPP_DIR), "-B", str(CPP_BUILD_DIR)])
        run_cmd([cmake, "--build", str(CPP_BUILD_DIR), "-j"])
        return

    print("[setup] cmake not found; falling back to g++ direct build")
    cxx = shutil.which("g++")
    if not cxx:
        raise RuntimeError("Neither cmake nor g++ found; cannot build src/cpp tools")

    common = str(CPP_DIR / "fvs_common.cpp")
    includes = ["-I", str(FAISS_INCLUDE)]
    libs = ["-L", str(FAISS_LIB_DIR), "-lfaiss", "-fopenmp"]
    std = ["-std=c++17"]
    opt = ["-O3", "-march=native"]

    run_cmd(
        [
            cxx,
            *std,
            *opt,
            *includes,
            common,
            str(CPP_DIR / "build_indexes.cpp"),
            *libs,
            "-o",
            str(CPP_BUILD_DIR / "build_indexes"),
        ]
    )

    run_cmd(
        [
            cxx,
            *std,
            *opt,
            *includes,
            common,
            str(CPP_DIR / "search_filters.cpp"),
            *libs,
            "-o",
            str(CPP_BUILD_DIR / "search_filters"),
        ]
    )


if sys.platform.startswith("win"):
    compile_args = ["/O2"]
    link_args = []
else:
    compile_args = ["-O3", "-fopenmp"]
    link_args = ["-fopenmp"]


def python_headers_available():
    include_dir = Path(sysconfig.get_paths().get("include", ""))
    return (include_dir / "Python.h").exists()


ext_modules = []
if python_headers_available():
    ext_modules = [
        Extension(
            name="filter_kernel",
            sources=[str(CPP_DIR / "filter_kernel.cpp")],
            include_dirs=[np.get_include(), str(FAISS_INCLUDE)],
            language="c++",
            library_dirs=[str(FAISS_LIB_DIR)],
            libraries=["faiss"],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        )
    ]
else:
    print("[setup] Python headers not found (Python.h). Skipping optional filter_kernel extension build.")
    print("[setup] To enable it, install python3-dev/python3.12-dev and rerun setup.")


if __name__ == "__main__":
    # Allow plain `python setup.py` as a convenient bootstrap step.
    if len(sys.argv) == 1:
        sys.argv.extend(["build_ext", "--inplace"])

    build_cpp_tools()
    setup(
        name="filter_kernel",
        version="0.1.0",
        ext_modules=ext_modules,
    )
