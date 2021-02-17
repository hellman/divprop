from distutils.core import Extension

ext_modules = [
    Extension(
        "subsets._lib",
        include_dirs=["./subsets/"],
        sources=[
            "./subsets/test.i",
            "./subsets/test.cpp",
        ],
        swig_opts=["-c++"],
        extra_compile_args=["-std=c++2a -O3"],
    ),
]


def build(setup_kwargs):
    setup_kwargs.update({"ext_modules": ext_modules})
