# divprop - Tools for cryptanalysis using division property

This package provides C++ implementation and Python bindings (SWIG) for division property computations of S-boxes. It was developed for the [Convexity of division property transitions](https://eprint.iacr.org/2021/1285) paper ([ASIACRYPT 2021](https://link.springer.com/chapter/10.1007/978-3-030-92062-3_12)), see also the other [supporting code](https://github.com/CryptoExperts/AC21-divprop-convexity/) for the paper.

If you this library in your research, please cite

```bib
@inproceedings{AC:Udovenko21,
  author       = {Aleksei Udovenko},
  title        = {Convexity of Division Property Transitions: Theory, Algorithms and
                  Compact Models},
  booktitle    = {{ASIACRYPT} {(1)}},
  series       = {Lecture Notes in Computer Science},
  volume       = {13090},
  pages        = {332--361},
  publisher    = {Springer},
  year         = {2021}
}
```

## Installation

Requires SWIG for building the extension (both for this package and its dependency [subsets](https://github.com/hellman/subsets)). Can be installed for pure python 3 or pypy3 for faster speeds.

```
$ sudo apt install swig
$ pip install divprop
```

## Usage

DivProp is the main package related to the paper's developments on division property. The two most important classes are `Sbox` and `SboxDivision`.

- `Sbox` is a small wrapper for representing S-boxes. 
- `SboxDivision` allows to easily compute all the convex sets described in the paper.

Examples:

```py
from divprop.all_sboxes import AES
from divprop import Sbox, SboxDivision

s = Sbox(AES, 8, 8)
# <Sbox hash=3b66e44419610dd0 n=8 m=8>

sd = SboxDivision(s)
sd.divcore
# <DenseSet hash=14421c71a4b40a67 n=16 wt=122 | 2:25 3:66 4:29 8:2>
sd.min_dppt
# <DenseSet hash=3bdcec9ddb5303f2 n=16 wt=2001 | 0:1 2:64 3:224 4:448 5:560 6:428 7:173 8:54 9:42 10:6 16:1>
sd.invalid_max
# <DenseSet hash=af326bfc6e4b2f4a n=16 wt=87 | 3:30 4:41 7:16>
sd.redundant_min
# <DenseSet hash=d165309d0be60267 n=16 wt=319 | 3:137 4:168 5:6 9:8>
sd.redundant_alternative_min
# <DenseSet hash=82186fa2cffeefc6 n=16 wt=274 | 3:152 4:112 5:2 9:8>
sd.propagation_map
[[0], [1, 2, 4, 8, 16, 32, 64, 128], [1, 2, 4, 8, 16, 32, 64, 128], ..., [4, 10, 18, 24, 33, 40, 48, 65, 80, 98, 129, 144], [255]]
```

The advanced algorithm for heavy S-boxes is implemented in [divprop.divcore_peekanfs](./src/divprop/divcore_peekanfs.py):

```py
from divprop.divcore_peekanfs import SboxPeekANFs

divcore, invalid_max = SboxPeekANFs(s).compute()
assert divcore == set(sd.divcore.to_Bins())
assert invalid_max == set(sd.invalid_max.to_Bins())
```

Its variation with filesystem cache (to reduce RAM usage) is implemented in [divpop.tool_random_sbox_benchmark](./src/divprop/tool_random_sbox_benchmark.py)


Todo
