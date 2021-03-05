from functools import reduce
from random import shuffle, randrange

from binteger import Bin
from divprop.subsets import Sbox2GI, DenseSet
from divprop.divcore import DenseDivCore
from divprop.divcore import DivCore_StrongComposition
from divprop.divcore import DivCore_StrongComposition8
import divprop.logs as logging

from test_sboxes import get_sboxes

logging.setup("DEBUG")


def test_DPPT():
    for name, sbox, n, m, dppt in get_sboxes():
        check_one_DPPT(sbox, n, m, dppt)


def check_one_DPPT(sbox, n, m, dppt):
    assert len(sbox) == 2**n
    assert 0 <= max(sbox) < 2**m
    if n != m:
        return
    DCS = DivCore_StrongComposition(n, m, m, sbox, sbox)
    DCS.process_logged(64)
    # for lst in DCS.current:
    #     print(lst, lst.to_Bins())
    # print()
    res = DCS.divcore
    print(res)
    print()

    id = list(range(2**n))
    test1 = DivCore_StrongComposition(n, n, n, id, sbox)
    test2 = DivCore_StrongComposition(n, n, n, sbox, id)
    test1.process()
    test2.process()
    ans = DenseDivCore.from_sbox(sbox, n, m)
    assert test1.divcore == test2.divcore == ans.data


if __name__ == '__main__':
    test_DPPT()
