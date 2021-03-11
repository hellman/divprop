from binteger import Bin
from divprop.subsets import DynamicLowerSet, DenseSet
from random import randrange, seed


def test_Dynamic():
    seed(123)

    n = 8

    for i in range(10):
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
                v = 1
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


if __name__ == '__main__':
    test_Dynamic()
