import logging
logging.basicConfig(level=logging.DEBUG)

from random import randrange, seed

from binteger import Bin

from subsets import DenseSet

from subsets.learn import LowerSetLearn, OracleFunction
from subsets.learn import GainanovSAT


def test_LSL():
    seed(123)

    for n in range(2, 18):
        a = DenseSet(n)
        for i in range(500 // n):
            a.set(randrange(2**n))
            if i % 10 == 0:
                lower = set(a.LowerSet())

                oracle = OracleFunction(lambda v: v.as_Bin(n).int in lower)
                system = LowerSetLearn(n=n, oracle=oracle)

                g = GainanovSAT(sense="min", solver="pysat/cadical")
                g.init(system)
                g.learn()

                answer = set(a.MaxSet())
                test = {vec.as_Bin(n).int for vec in system.iter_lower()}
                assert test == answer

                answer = set(a.LowerSet().Complement().MinSet())
                test = {vec.as_Bin(n).int for vec in system.iter_upper()}
                assert test == answer


if __name__ == '__main__':
    test_LSL()
