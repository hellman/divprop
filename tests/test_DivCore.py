from divprop.subsets import Sbox2GI, DenseSet
from divprop.divcore import DivCore


def test_DivCore():
    s = [1, 2, 3, 4, 0, 7, 6, 5]
    n = m = 3
    dc = DivCore.from_sbox(s, n, m, log=True)
    assert dc.data.info("divcore") == \
        "dfa780cfc382387a:divcore n=6 wt=12 | 2:3 3:9"
    assert dc.data.get_support() == \
        (7, 11, 12, 19, 20, 25, 35, 36, 42, 49, 50, 56)


if __name__ == '__main__':
    test_DivCore()
