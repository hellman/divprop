import logging
logging.basicConfig(level=logging.DEBUG)

from random import randrange, seed

from binteger import Bin

from divprop.subsets import DenseSet
from divprop.learn import LowerSetLearn


def atest_LSL():
    # seed(123)

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


def atest_DenseLowerSetLearn():
    # seed(123)

    for n in range(2, 12):
        a = DenseSet(n)
        for i in range(200):
            a.set(randrange(2**n))
            if i % 10 == 0:
                lower = set(a.LowerSet())
                answer = set(a.MaxSet())
                print("learning:", n, *[v.str for v in a.MaxSet().to_Bins()])

                class Oracle:
                    n_calls = 0

                    def query(self, v):
                        self.n_calls = self.n_calls + 1
                        return v.int in lower

                o = Oracle()
                LSL = DenseLowerSetLearn(n)
                test = LSL.learn_simple(oracle=o)

                assert {v.int for v in test} == answer
                print("learnt in", o.n_calls)
                print()


if __name__ == '__main__':
    # atest_LSL()
    # atest_DenseLowerSetLearn()
    pass
