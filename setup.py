import os, sys
from setuptools import setup
from distutils.core import Extension

try:
    import subsets
    SUBSETS_ROOT = os.path.dirname(subsets.__file__)
    SUBSETS_SO = subsets._subsets.__file__
except ImportError:
    print("Error: Package 'subsets' must be instaled before building")
    sys.exit(1)

setup(
    ext_modules=[
        Extension(
            "divprop._lib",
            include_dirs=[
                "./src/",
                "./src/sbox/",
                "./src/divprop/",
                SUBSETS_ROOT,
            ],
            depends=[
                "./src/divprop/DivCore.hpp",
                "./src/sbox/Sbox.hpp",
                "./src/hackycpp.hpp",
                SUBSETS_SO,
            ],
            sources=[
                "./src/divprop/lib.i",
                "./src/sbox/Sbox.cpp",
            ],
            swig_opts=[
                "-c++",
                "-DSWIGWORDSIZE64",  # https://github.com/swig/swig/issues/568
                "-I./src/",
                "-I./src/sbox/",
                "-I" + SUBSETS_ROOT,
            ],
            extra_compile_args=["-std=c++2a", "-O3", "-fopenmp"],
            extra_link_args=["-fopenmp", SUBSETS_SO],
        ),
    ]
)
