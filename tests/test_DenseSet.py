from random import randrange

from subsets import DenseSet


def assert_raises(f, err=RuntimeError):
    try:
        f()
    except err as e:
        print("exception good:", e)
    else:
        assert 0, f"exception {err} not raised"


def test_DenseSet():
    a = DenseSet(3)  # 3-bit vectors
    a.add(1)
    a.add(5)
    a.add(7)
    assert a.get_support() == (1, 5, 7)

    b = DenseSet((2, 3, 4, 5), 3)
    b.set(5)
    b.set(4, 0)
    assert b.get_support() == (2, 3, 5)

    a |= b

    assert a.get_support() == (1, 2, 3, 5, 7)

    a.resize(6)

    b = a.copy()
    b.set(0, 1)
    b.set(7, 0)
    b.set(8)
    a.set(3, 0)
    a.set(4, 1)
    assert a.get_support() == (1, 2, 4, 5, 7)
    assert b.get_support() == (0, 1, 2, 3, 5, 8)
    assert b.get_weight() == 6

    assert b.info() == '<DenseSet hash=f01a3338f076a640 n=6 wt=6 | 0:1 1:3 2:2>'
    assert DenseSet([1, 2, 4, 5, 7], 6) == a

    assert_raises(
        lambda: DenseSet([1, 2, 4, 5, 7], 2)
    )
    assert_raises(
        lambda: a.save_to_file("/NON-EXISTENT STUFF")
    )


def test_properties():
    for n in range(1, 12):
        a = DenseSet(n)
        for i in range(1000):
            a.set(randrange(2**n))
            assert a.LowerSet().Not() == a.Not().UpperSet()
            assert a.UpperSet().Not() == a.Not().LowerSet()
            assert a.MaxSet().Not() == a.Not().MinSet()
            assert a.MinSet().Not() == a.Not().MaxSet()

            assert a.LowerSet() == a.Not().UpperSet().Not()
            assert a.UpperSet() == a.Not().LowerSet().Not()
            assert a.MaxSet() == a.Not().MinSet().Not()
            assert a.MinSet() == a.Not().MaxSet().Not()


if __name__ == '__main__':
    test_DenseSet()
    test_properties()
