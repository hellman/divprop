from functools import reduce
from random import shuffle, randrange

from binteger import Bin
from divprop.subsets import Sbox2GI, DenseSet
from divprop.divcore import DenseDivCore
from divprop.divcore import DivCore_StrongComposition

from test_sboxes import get_sboxes


def test_DPPT():
    for name, sbox, n, m, dppt in get_sboxes():
        check_one_DPPT(sbox, n, m, dppt)


def check_one_DPPT(sbox, n, m, dppt):
    assert len(sbox) == 2**n
    assert 0 <= max(sbox) < 2**m
    if n != m or n != 4:
        return
    DCS = DivCore_StrongComposition(n, m, m, sbox, sbox)
    DCS.process()
    for lst in DCS.current:
        print(lst, lst.to_Bins())
    print()
    res = DCS.divcore
    print(res)
    print()


if __name__ == '__main__':
    test_DPPT()
