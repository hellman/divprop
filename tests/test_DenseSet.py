from random import randrange
from divprop.subsets import DenseSet


def test_DenseSet():
    a = DenseSet(6)  # 6-bit vectors
    a.set(1)
    a.set(5)
    a.set(7)
    assert a.get_support() == (1, 5, 7)

    b = DenseSet(6)
    b.set(2)
    b.set(3)
    b.set(4)
    b.set(5)
    b.set(4, 0)
    assert b.get_support() == (2, 3, 5)

    a |= b

    assert a.get_support() == (1, 2, 3, 5, 7)

    b = a.copy()
    b.set(0, 1)
    b.set(7, 0)
    b.set(8)
    a.set(3, 0)
    a.set(4, 1)
    assert a.get_support() == (1, 2, 4, 5, 7)
    assert b.get_support() == (0, 1, 2, 3, 5, 8)
    assert b.get_weight() == 6

    assert b.info() == 'f01a3338f076a640:? n=6 wt=6 | 0:1 1:3 2:2'
    assert b.info("SetB") == 'f01a3338f076a640:SetB n=6 wt=6 | 0:1 1:3 2:2'
    print(a.info("SetA"))
    print(b.info("SetB"))


def test_properties():
    for n in range(1, 12):
        a = DenseSet(n)
        for i in range(1000):
            a.set(randrange(2**n))
            assert a.LowerSet().Not() == a.Not().UpperSet()
            assert a.UpperSet().Not() == a.Not().LowerSet()
            assert a.MaxSet().Not() == a.Not().MinSet()
            assert a.MinSet().Not() == a.Not().MaxSet()


if __name__ == '__main__':
    test_DenseSet()
    test_properties()
