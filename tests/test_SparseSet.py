from binteger import Bin
from subsets import (
    DynamicLowerSet, DynamicUpperSet, DenseSet,
    GrowingUpperFrozen, GrowingLowerFrozen,
)
from random import randrange, seed


def test_Dynamic():
    seed(123)

    n = 8

    for i in range(10):
        # lower set
        sparse = DynamicLowerSet((), n)
        dense = DenseSet(n)
        for j in range(100):
            v = randrange(2**n)

            dense.add(v)
            dense.do_MaxSet()

            sparse.add_lower_singleton(v)

            assert sparse.set == set(dense)

            if i > 5:
                v = randrange(2**n)
                # print()
                # print(*sorted([Bin(v, n).str for v in sparse.set]))
                # print(*sorted([Bin(v, n).str for v in dense]))

                rem = DenseSet((v,), n).UpperSet()
                # print("remove", v, Bin(v, n), ":", *[Bin(v, n).str for v in rem])
                # print("and", *[Bin(v, n).str for v in rem.Complement()])
                dense = (dense.LowerSet() & rem.Complement()).MaxSet()

                sparse.remove_upper_singleton(v)

                # print(*[Bin(v, n).str for v in sparse.set])
                # print(*[Bin(v, n).str for v in dense])
                assert sparse.set == set(dense)

        # upper set
        sparse = DynamicUpperSet((), n)
        dense = DenseSet(n)
        for j in range(100):
            v = randrange(2**n)

            dense.add(v)
            dense.do_MinSet()

            sparse.add_upper_singleton(v)

            assert sparse.set == set(dense)

            if i > 5 and 1:
                v = randrange(2**n)

                rem = DenseSet((v,), n).LowerSet()
                dense = (dense.UpperSet() & rem.Complement()).MinSet()

                sparse.remove_lower_singleton(v)

                assert sparse.set == set(dense)


def test_GrowingExtremeFrozen():
    f01 = frozenset({0, 1})
    f02 = frozenset({0, 2})
    f012 = frozenset({0, 1, 2})

    s = GrowingUpperFrozen(10)
    s.add(f01)
    s.add(f02)
    s.add(f012)
    s.do_MinSet()
    assert set(s) == {f01, f02}

    s = GrowingLowerFrozen(10)
    s.add(f01)
    s.add(f02)
    s.add(f012)
    s.do_MaxSet()
    assert set(s) == {f012}


if __name__ == '__main__':
    test_Dynamic()
    test_GrowingExtremeFrozen()
