import os
from setuptools import setup
from distutils.core import Extension

try:
    import hackycpp
    HACKYCPP_ROOT = os.path.dirname(hackycpp.__file__)
except ImportError:
    print("Package 'hackycpp' must be instaled before building")
    print("Build dependencies can not be specified yet...")
    print("pip install hackycpp")
    raise

try:
    import subsets
    SUBSETS_ROOT = os.path.dirname(subsets.__file__)
except ImportError:
    print("Package 'subsets' must be instaled before building")
    print("Build dependencies can not be specified yet...")
    print("pip install subsets")
    raise

package_dir = {'': 'src'}

packages = [
    'divprop',
]

package_data = {
    # '': ['*'],
    'divprop': ['*.so'],
}

install_requires = [
    'binteger>=0.7.0',
    'coloredlogs>=15.0',
    'tqdm>=4.58.0',

    'subsets>=0.1.0',
    # 'optisolveapi>=0.1.0',
]

entry_points = {
    'console_scripts': [
        'divprop.sbox2ddt = divprop.tools:tool_sbox2ddt',
        'divprop.sbox2ptt = divprop.tools:tool_sbox2ptt',
        'divprop.sbox2divcore = divprop.tools:tool_sbox2divcore',
        'divprop.divcore2bounds = divprop.tools:tool_divcore2bounds',

        'divprop.random_sbox_benchmark = '
        + 'divprop.tool_random_sbox_benchmark:tool_RandomSboxBenchmark',
    ]
}

setup(
    name='divprop',
    version='0.2.1',
    description='Tools for cryptanalysis (division property)',
    long_description=None,
    author='Aleksei Udovenko',
    author_email="aleksei@affine.group",
    maintainer=None,
    maintainer_email=None,
    url=None,
    license="MIT",
    package_dir=package_dir,
    packages=packages,
    package_data=package_data,
    install_requires=install_requires,
    entry_points=entry_points,
    python_requires='>=3.7,<4.0',
    ext_modules=[
        # interaction between extensions is messy
        # so let's put everything in one...
        Extension(
            "subsets._lib",
            include_dirs=[
                "./src/",
                "./src/sbox/",
                "./src/divprop/divprop/",
                HACKYCPP_ROOT,
                SUBSETS_ROOT,
            ],
            depends=[
                "./src/divprop/divprop/DivCore.hpp",
                "./src/sbox/Sbox.hpp",
            ],
            sources=[
                "./src/divprop/divprop.i",
                "./src/sbox/Sbox.cpp",
            ],
            swig_opts=[
                "-c++",
                "-DSWIGWORDSIZE64",  # https://github.com/swig/swig/issues/568
                "-I" + HACKYCPP_ROOT,
                "-I" + SUBSETS_ROOT,
            ],
            extra_compile_args=["-std=c++2a", "-O2", "-fopenmp"],
            extra_link_args=["-fopenmp"],
        ),
    ]
)
