from distutils.core import Extension

ext_modules = [
    Extension(
        "divprop._libsubsets",
        include_dirs=[
            "./src/",
            "./src/divprop/common/",
            "./src/divprop/subsets/",
        ],
        sources=[
            "./src/divprop/libsubsets.i",
            "./src/divprop/subsets/DenseSet.cpp",
        ],
        swig_opts=["-c++", "-DSWIGWORDSIZE64"],  # https://github.com/swig/swig/issues/568
        extra_compile_args=["-std=c++2a", "-O0"],
    ),
]


def build(setup_kwargs):
    setup_kwargs.update({"ext_modules": ext_modules})
