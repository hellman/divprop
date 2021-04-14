import logging
from collections import Counter

from binteger import Bin

from .base import satisfy, MIPSolverException
from .random_group_cut import RandomGroupCut

from divprop.learn import LowerSetLearn


log = logging.getLogger(__name__)


class GemCut(RandomGroupCut):
    def _oracle(self, v):
        assert isinstance(v, Bin), v
        res = self.check_group(v.support())
        if res:
            self.sol[v] = res
        return bool(res)

    def generate(self):
        n = len(self.lo)
        self.sol = {}

        LSL = LowerSetLearn(n=n, oracle=self._oracle)
        lowerset = LSL.learn()

        tops = {}
        for v in lowerset:
            v = Bin(v, n)
            sol = self.sol[v]
            covered = [self.lo[i] for i in v.support()]
            # print("top", v.str, "%3d" % v.hw(), v.support(), "|", sol)

            # TODO: clean up asserts
            assert all(satisfy(q, sol) for q in self.hi)
            assert all(not satisfy(q, sol) for q in covered)

            if self.inverted:
                func, value_good = sol[:-1], -sol[-1]
                # x1a1 + x2a2 + x3a3 >= t
                # =>
                # x1(1-a1) + x2(1-a2) + x3(1-a3) >= t
                # -x1a1 -x2a2 -x3a3 >= t-sum(x)
                value = value_good - sum(func)
                sol = tuple(-x for x in func) + (-value,)
                covered = [tuple(1 - a for a in q) for q in covered]

                assert all(satisfy(q, sol) for q in self.orig_lo)
                for q in covered:
                    assert q in self.orig_hi
            else:
                assert all(satisfy(q, sol) for q in self.orig_hi)
                for q in covered:
                    assert q in self.orig_lo

            assert all(not satisfy(q, sol) for q in covered)
            tops[sol] = covered
        return tops

    def check_group(self, bads):
        LP = self.model.__copy__()

        for i in bads:
            q = self.lo[i]
            LP.add_constraint(self.cs_per_lo[q])

        try:
            LP.solve()
        except MIPSolverException:
            return False

        val_xs = tuple(LP.get_values(x) for x in self.xs)
        if all(abs(v - round(v)) < 0.00001 for v in val_xs):
            # is integral
            val_xs = tuple(int(v + 0.5) for v in val_xs)
            val_c = int(LP.get_values(self.c) + 0.5)
        else:
            # keep real
            val_c = LP.get_values(self.c) - 0.5
        ineq = val_xs + (-val_c,)

        assert all(satisfy(p, ineq) for p in self.hi)
        assert all(not satisfy(self.lo[i], ineq) for i in bads)
        return ineq
