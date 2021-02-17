from distutils.core import Extension

ext_modules = [
    Extension(
        "subsets._lib",
        include_dirs=["./src/subsets/"],
        sources=[
            "./src/subsets/lib.i",
            "./src/subsets/DenseSet.cpp",
            # "./src/subsets/common.cpp",
        ],
        swig_opts=["-c++", "-DSWIGWORDSIZE64"],  # https://github.com/swig/swig/issues/568
        extra_compile_args=["-std=c++2a", "-O3"],
    ),
]


def build(setup_kwargs):
    setup_kwargs.update({"ext_modules": ext_modules})
