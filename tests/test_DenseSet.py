from subsets.lib import DenseSet


def test_DenseSet():
    a = DenseSet(10)
    a.set(1)
    a.set(5)
    a.set(7)
    assert a.get_support() == (1, 5, 7)

    b = DenseSet(10)
    b.set(2)
    b.set(3)
    b.set(4)
    b.set(5)
    b.set(4, 0)
    assert b.get_support() == (2, 3, 5)

    a |= b

    assert a.get_support() == (1, 2, 3, 5, 7)
