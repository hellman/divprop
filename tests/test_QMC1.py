from random import randrange
from divprop.subsets import DenseSet, QMC1


def test_QMC1():
    for n in range(1, 12):
        P = DenseSet(n)
        for i in range(1000):
            P.set(randrange(2**n))

            if i < 10 or i % 10 == 0:
                S = QMC1(P)

                Sa = {}
                for a, u in S:
                    Sa.setdefault(a, []).append(u)

                assert len(Sa) == len(P)
                for a in Sa:
                    # prec(u)
                    U = DenseSet(n)
                    for u in Sa[a]:
                        U.set(a)
                    # a + prec(u)
                    U.Not(a)
                    assert U <= P


def test_QMC1_random():
    for n in (4, 8, 12, 13):
        print("random", n)
        P = DenseSet(n)
        for x in range(2**n):
            if randrange(2):
                P.set(x)

        S = QMC1(P)

        Sa = {}
        for a, u in S:
            Sa.setdefault(a, []).append(u)

        assert len(Sa) == len(P)
        for a in Sa:
            # prec(u)
            U = DenseSet(n)
            for u in Sa[a]:
                U.set(a)
            # a + prec(u)
            U.Not(a)
            assert U <= P


if __name__ == '__main__':
    test_QMC1()
    test_QMC1_random()
