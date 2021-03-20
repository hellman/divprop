from setuptools import setup
from distutils.core import Extension

package_dir = {'': 'src'}

packages = ['divprop', 'divprop.inequalities']

package_data = {'': ['*'], 'divprop': ['common/*', 'subsets/*', '*.so']}

install_requires = [
    'binteger>=0.7.0',
    'coloredlogs>=15.0',
    'tqdm>=4.58.0',
]

entry_points = {
    'console_scripts': [
        'divprop.sbox_ineqs = divprop.sbox_ineqs:main',
        'divprop.sbox2divcore = divprop.tools:tool_sbox2divcore',
        'divprop.sbox2ddt = divprop.tools:tool_sbox2ddt',
        'divprop.setinfo = divprop.tools:tool_setinfo',
        'divprop.divcore2bounds = divprop.tools:tool_divcore2bounds',
        'divprop.mono2ineqs = divprop.inequalities.tools:tool_mono2ineqs',
        'divprop.random_sbox_benchmark = divprop.tool_random_sbox_benchmark:tool_RandomSboxBenchmark',
    ]
}

setup(
    name='divprop',
    version='0.2.1',
    description='Tools for Division Property Cryptanalysis',
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
            "divprop._libsubsets",
            include_dirs=[
                "./src/",
                "./src/divprop/common/",
                "./src/divprop/subsets/",
                "./src/divprop/divprop/",
                "./src/divprop/sbox/",
            ],
            depends=[
                "./src/divprop/common/common.hpp",
                "./src/divprop/common/Sweep.hpp",
                "./src/divprop/subsets/DenseSet.hpp",
                "./src/divprop/divprop/DivCore.hpp",
                "./src/divprop/sbox/Sbox.hpp",
            ],
            sources=[
                "./src/divprop/libsubsets.i",
                "./src/divprop/subsets/DenseSet.cpp",
                "./src/divprop/sbox/Sbox.cpp",
            ],
            swig_opts=["-c++", "-DSWIGWORDSIZE64"],  # https://github.com/swig/swig/issues/568
            extra_compile_args=["-std=c++2a", "-O2", "-fopenmp"],
            extra_link_args=["-fopenmp"],
        ),
    ]
)
