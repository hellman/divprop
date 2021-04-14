from binteger import Bin
from divprop.subsets import SparseSet
from divprop.learn import ExtraPrec_LowerSet


def test_EP():
    n = 5
    order = [SparseSet(Bin(v, n).support) for v in range(2**n)]
    lookup = {v: i for i, v in enumerate(order)}
    EP = ExtraPrec_LowerSet(order, lookup)

    vec = SparseSet({
        0b00111,
        0b00011,
        0b00001,
        0b11100,
        0b00100,
    })
    red = SparseSet({
        0b00111,
        0b11100,
    })
    exp = SparseSet({
        0b00000,
        0b00001,
        0b00010,
        0b00011,
        0b00100,
        0b00101,
        0b00110,
        0b00111,

        0b00000,
        0b00100,
        0b01000,
        0b01100,
        0b10000,
        0b10100,
        0b11000,
        0b11100,
    })
    assert EP.expand(vec) == exp
    assert EP.reduce(vec) == red
