from setuptools import setup
from distutils.core import Extension

package_dir = {'': 'src'}

packages = [
    'divprop',
    'subsets',
    'optisolveapi',
    'optimodel',
]

package_data = {
    '': ['*'],
    'divprop': ['../common/*', 'subsets/*'],
    'subsets': ['../common/*', '*.so'],
}

install_requires = [
    'binteger>=0.7.0',
    'coloredlogs>=15.0',
    'tqdm>=4.58.0',
]

entry_points = {
    'console_scripts': [
        'subsets.setinfo = subsets.tools:tool_setinfo',
        'optimodel.milp = optimodel.tool_milp:main',

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
    description=''
                'Tools for Cryptanalysis '
                '(Subsets & Transforms, MILP modeling, Division property)'
                '',
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
                "./src/common/",
                "./src/subsets/",
                "./src/sbox/",
                "./src/divprop/divprop/",
            ],
            depends=[
                "./src/common/common.hpp",
                "./src/common/Sweep.hpp",
                "./src/subsets/DenseSet.hpp",
                "./src/divprop/divprop/DivCore.hpp",
                "./src/sbox/Sbox.hpp",
            ],
            sources=[
                "./src/subsets/lib.i",
                "./src/subsets/DenseSet.cpp",
                "./src/sbox/Sbox.cpp",
            ],
            swig_opts=["-c++", "-DSWIGWORDSIZE64"],  # https://github.com/swig/swig/issues/568
            extra_compile_args=["-std=c++2a", "-O2", "-fopenmp"],
            extra_link_args=["-fopenmp"],
        ),
    ]
)
