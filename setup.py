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
        Extension(
            "divprop._libsubsets",
            include_dirs=[
                "./src/",
                "./src/divprop/common/",
                "./src/divprop/subsets/",
            ],
            depends=[
                "./src/divprop/common/common.hpp",
                "./src/divprop/common/Sweep.hpp",
            ],
            sources=[
                "./src/divprop/libsubsets.i",
                "./src/divprop/subsets/DenseSet.cpp",
                "./src/divprop/subsets/SboxGraph.cpp",
            ],
            swig_opts=["-c++", "-DSWIGWORDSIZE64"],  # https://github.com/swig/swig/issues/568
            extra_compile_args=["-std=c++2a", "-O0"],
        ),
    ]
)
