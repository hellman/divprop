from binteger import Bin
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

    assert dc.LB().info("LB") == \
        "9fe09c93bbcdbb87:LB n=6 wt=8 | 2:6 3:2"
    assert dc.UB().info("UB") == \
        "60f7fb1d9a638a50:UB n=6 wt=12 | 3:6 4:6"
    assert dc.FullDPPT().info("FullDPPT") == \
        "b712d2af3b433a45:FullDPPT n=6 wt=43 | 0:1 1:3 2:10 3:13 4:12 5:3 6:1"
    assert dc.MinDPPT().info("MinDPPT") == \
        "ff7ce5b30da61490:MinDPPT n=6 wt=15 | 0:1 2:7 3:3 4:3 6:1"

    assert dc.LB().get_support() == \
        (3, 5, 6, 17, 26, 34, 41, 48)
    assert dc.UB().get_support() == \
        (13, 14, 21, 22, 27, 37, 38, 43, 51, 57, 58, 60)
    assert dc.FullDPPT().get_support() == \
        (0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 27, 28, 29, 30, 31, 33, 35, 36, 37, 38, 39, 43, 44, 45, 46, 47, 51, 52, 53, 54, 55, 63)
    assert dc.MinDPPT().get_support() == \
        (0, 9, 10, 12, 18, 20, 27, 28, 33, 36, 43, 44, 51, 52, 63)


def test_Not():
    a = DenseSet(10)
    a.set(0)
    assert a.get_support() == (0,)
    a.do_Not(1)
    assert a.get_support() == (1,)
    a.do_Not(2)
    assert a.get_support() == (3,)
    a.do_Not(512)
    assert a.get_support() == (515,)


def test_DPPT():
    # ASCON
    n = m = 5
    sbox = [4, 11, 31, 20, 26, 21, 9, 2, 27, 5, 8, 18, 29, 3, 6, 28, 30, 19, 7, 14, 0, 13, 17, 24, 16, 12, 1, 25, 22, 10, 15, 23]
    dppt = [0], [1, 2, 4, 8, 16], [1, 2, 4, 8, 16], [3, 4, 9, 10, 17, 18, 24], [2, 4, 8, 16], [3, 5, 6, 9, 10, 12, 17, 18, 20, 24], [3, 5, 6, 8, 17, 18, 20], [6, 9, 10, 12, 19, 20, 24], [1, 2, 4, 8, 16], [1, 6, 10, 12, 16], [3, 5, 6, 8, 17, 18, 20], [3, 5, 6, 9, 10, 12, 18, 20, 24], [3, 5, 8, 16], [3, 5, 9, 10, 12, 17, 18, 20, 24], [7, 9, 10, 12, 17, 18, 20, 24], [7, 9, 12, 20, 24], [1, 2, 8, 16], [2, 5, 9, 12, 17, 20, 24], [2, 5, 9, 12, 17, 20, 24], [3, 5, 10, 12, 18, 20, 25], [3, 5, 6, 9, 10, 12, 18, 20, 24], [6, 10, 13, 18, 21, 25, 28], [6, 9, 10, 18, 21, 24], [10, 23, 25], [1, 6, 10, 12, 16], [3, 5, 6, 10, 18, 20, 25], [3, 5, 6, 9, 10, 18, 20, 24], [3, 5, 10, 18, 20, 25], [3, 5, 9, 10, 18, 20, 24], [10, 18, 29], [7, 9, 10, 18, 24], [31]
    check_one_DPPT(sbox, n, m, dppt)


def check_one_DPPT(sbox, n, m, dppt):
    assert len(sbox) == 2**n
    assert 0 <= max(sbox) < 2**m

    mindppt1 = set()
    for u, vs in enumerate(dppt):
        for v in vs:
            mindppt1.add((u << m) | v)
    mindppt1 = tuple(sorted(mindppt1))

    dc = DivCore.from_sbox(sbox, n, m, log=True)
    assert dc.MinDPPT().get_support() == mindppt1


if __name__ == '__main__':
    test_DivCore()
    test_DPPT()