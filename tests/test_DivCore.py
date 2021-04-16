from functools import reduce
from random import shuffle, randrange

from binteger import Bin
from subsets import DenseSet, Sbox

from divprop.divcore import DivCore

from test_sboxes import get_sboxes


def test_DivCore():
    s = [1, 2, 3, 4, 0, 7, 6, 5]
    n = m = 3
    dc = DivCore.from_sbox(s, n, m, method="dense", debug=True)
    dc2 = DivCore.from_sbox(s, n, m, method="peekanfs", debug=True)
    assert dc == dc2

    assert dc.to_dense().info() == \
        "<DenseSet hash=dfa780cfc382387a n=6 wt=12 | 2:3 3:9>"
    assert dc.to_dense().get_support() == \
        (7, 11, 12, 19, 20, 25, 35, 36, 42, 49, 50, 56)

    assert dc.LB().info() == \
        "<DenseSet hash=9fe09c93bbcdbb87 n=6 wt=8 | 2:6 3:2>"
    assert dc.UB(method="redundant").info() == \
        "<DenseSet hash=60f7fb1d9a638a50 n=6 wt=12 | 3:6 4:6>"
    assert dc.UB(method="complement").info() == \
        "<DenseSet hash=449b201e8a75f016 n=6 wt=10 | 3:8 4:2>"
    assert dc.FullDPPT().info() == \
        "<DenseSet hash=b712d2af3b433a45 n=6 wt=43 | 0:1 1:3 2:10 3:13 4:12 5:3 6:1>"
    assert dc.MinDPPT().info() == \
        "<DenseSet hash=ff7ce5b30da61490 n=6 wt=15 | 0:1 2:7 3:3 4:3 6:1>"

    assert dc.LB().get_support() == \
        (3, 5, 6, 17, 26, 34, 41, 48)
    assert dc.UB("redundant").get_support() == \
        (13, 14, 21, 22, 27, 37, 38, 43, 51, 57, 58, 60)
    assert dc.UB("complement").get_support() == \
        (13, 14, 21, 22, 26, 37, 38, 41, 51, 60)
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
    for name, sbox, n, m, dppt in get_sboxes():
        check_one_DPPT(sbox, n, m, dppt)
        check_one_relations(sbox, n, m)

    for n in range(4, 10):
        for i in range(5):
            m = n
            sbox = list(range(2**n))
            shuffle(sbox)
            check_one_relations(sbox, n, m)

    for n in range(4, 8):
        for i in range(5):
            m = n + 1
            sbox = [randrange(2**m) for _ in range(2**n)]
            check_one_relations(sbox, n, m)

            m = n + 4
            sbox = [randrange(2**m) for _ in range(2**n)]
            check_one_relations(sbox, n, m)


def check_one_DPPT(sbox, n, m, dppt):
    assert len(sbox) == 2**n
    assert 0 <= max(sbox) < 2**m

    mindppt1 = set()
    for u, vs in enumerate(dppt):
        for v in vs:
            mindppt1.add((u << m) | v)
    mindppt1 = tuple(sorted(mindppt1))

    dc = DivCore.from_sbox(sbox, n, m, debug=True)
    assert tuple(dc.MinDPPT()) == dc.MinDPPT().get_support() == mindppt1
    assert len(dc.MinDPPT()) == dc.MinDPPT().get_weight() == len(mindppt1)
    assert dc.FullDPPT() == dc.to_dense().UpperSet().Not(dc.mask_u)


def check_one_relations(sbox, n, m):
    dc = DivCore.from_sbox(sbox, n, m, debug=True)

    mid = dc.MinDPPT().Not(dc.mask_u)
    lb = dc.LB()
    ubr = dc.UB(method="redundant")
    ubc = dc.UB(method="complement")

    assert ubr.UpperSet() <= ubc.UpperSet()
    assert not (ubr & mid)
    assert not (ubc & mid)

    assert form_partition(mid, lb.LowerSet() | ubc.UpperSet())
    assert form_partition(lb.LowerSet(), mid, ubr.UpperSet())

    assert ubr.UpperSet() == (mid.LowerSet() | lb.LowerSet()).Complement()
    assert ubc.UpperSet() == (mid.LowerSet()).Complement()

    print(
        "LB", len(dc.LB()),
        "UB", len(dc.UB()),
        "MinDPPT", len(dc.MinDPPT()),
        "FullDPPT", len(dc.FullDPPT()),
    )
    print("---")


def form_partition(*sets):
    for s in sets:
        break
    return (
        reduce(lambda a, b: a | b, sets).is_full()
        and sum(map(len, sets)) == 2**s.n
    )


def test_peekanfs():

    for n in range(2, 10):
        m = n
        sbox = list(range(2**n))
    shuffle(sbox)

    sbox = Sbox(sbox, n, m)
    test1 = sorted(DivCore.from_sbox(sbox, method="dense").to_Bins())
    test2 = sorted(DivCore.from_sbox(sbox, method="peekanfs").to_Bins())
    assert test1 == test2
    print("OK")


if __name__ == '__main__':
    test_DivCore()
    test_DPPT()
    test_peekanfs()
