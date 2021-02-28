import logging
logging.basicConfig(level=logging.DEBUG)

from random import randrange

from binteger import Bin

from divprop.subsets import DenseSet
from divprop.learn import LowerSetLearn


def test_LSL():
    for n in range(2, 12):
        a = DenseSet(n)
        for i in range(200):
            a.set(randrange(2**n))
            if i % 10 == 0:
                lower = set(a.LowerSet())
                oracle = lambda v: v.int in lower
                LSL = LowerSetLearn(n, oracle)
                answer = set(a.MaxSet())
                test = LSL.learn()
                assert {v.int for v in test} == answer


if __name__ == '__main__':
    test_LSL()
