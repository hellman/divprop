import os
import logging
import pickle
from math import ceil
from random import sample, randrange, choices, shuffle, choice
from collections import Counter, namedtuple, defaultdict
from itertools import combinations

from binteger import Bin

from . import (
    inner, satisfy,
    # MixedIntegerLinearProgram, MIPSolverException,
    # Polyhedron,
)

from divprop.subsets import (
    DenseSet,
    neibs_up_tuple, not_tuple, support_int_le,
    WeightedFrozenSets,
    GrowingLowerFrozen,
    GrowingUpperFrozen,
)



# TBD: polyhedron

class Generator:
    def generate(self, pool):
        raise NotImplementedError()


class Hats(Generator):
    def make_ineq(self, top, subs):
        n = len(top)

        for u in subs:
            assert sum(top) - sum(u) == 1

        large = len(subs)
        eq = [0] * n + [large]

        # 0 bits in top
        for i in range(n):
            if top[i] == 0:
                eq[i] = large

        # single 0 bits in each relevant u
        for u in subs:
            for i in range(n):
                if u[i] == 0 and top[i] == 1:
                    eq[i] = 1
                    break
            else:
                assert 0, "something wrong"
        return eq

    def generate(self, pool):
        hats = defaultdict(list)
        for u in pool.bad:
            for supu in neibs_up_tuple(u):
                hats[supu].append(u)

        for top, subs in hats.items():
            ineq = self.make_ineq(top, subs)
            fset = pool.system.encode_bad_subset(
                pool.bad2i[q] for q in subs
            )
            pool.system.add_feasible(
                fset, sol=IneqInfo(ineq=ineq, source="hat"),
            )


class RandomPlaneCut(Generator):
    def __init__(self, max_coef=100, exp=-0.5):
        self.max_coef = int(max_coef)
        self.exp = float(exp)

    def generate(self, pool):
        lst = list(range(self.max_coef))
        wts = [1] + [i**self.exp for i in range(1, self.max_coef)]
        lin = choices(lst, wts, k=pool.n)
        # lin = [randrange(max_coef+1) for _ in range(pool.n)]
        ev_good = min(inner(p, lin) for p in pool.good)
        fset = pool.system.encode_bad_subset(
            i for i, q in enumerate(pool.i2bad) if inner(q, lin) < ev_good
        )
        # lin. comb. >= ev_good
        ineq = tuple(lin) + (-ev_good,)
        pool.system.add_feasible(
            fset,
            sol=IneqInfo(ineq=ineq, source="random_ineq")
        )


def tuple_xor(t, mask):
    assert len(t) == len(mask)
    return tuple(a ^ b for a, b in zip(t, mask))


def tuple_preceq(a, b):
    return all(aa <= bb for aa, bb in zip(a, b))



def gen_basic_ineq_monotone(self, pi, p):
    fset = self.system.encode_bad_subset((pi,))
    # does not belong to LowerSet({p})
    # <=>
    # sum coords with p_i = 0 >= 1
    ineq = tuple(1 if x == 0 else 0 for x in p) + (-1,)

    self.system.add_feasible(
        fset, sol=IneqInfo(ineq=ineq, source="basic-mono"),
    )

def gen_basic_ineq_single(self, pi, p):
    fset = self.system.encode_bad_subset((pi,))
    # x != 0011 iff x0 + x1 + (1-x2) + (1-x3) >= 1
    #           iff x0 + x1 - x2 - x3 + sum() - 1 >= 0
    ineq = tuple(1 if x == 0 else -1 for x in p) + (sum(p) - 1,)

    self.system.add_feasible(
        fset, sol=IneqInfo(ineq=ineq, source="basic-single"),
    )

def gen_random_inequality(self, max_coef=100, exp=-0.5):
    RandomPlaneCut(max_coef, exp).generate(self)

def gen_hats(self):
    Hats().generate(self)


# def init_system(self):
        # if self.is_monotone:
        #     for pi, p in enumerate(self.i2bad):
        #         if self.use_point_prec:
        #             if any(tuple_preceq(p, q) for q in self.i2bad if q != p):
        #                 continue
        #         self.gen_basic_ineq_monotone(pi, p)
        # else:
        #     for pi, p in enumerate(self.i2bad):
        #         self.gen_basic_ineq_single(pi, p)


class Polyhedron(Generator):
    def generate_from_polyhedron(self):
        """
        Note: SageMath uses PPL library for this.
        """

        self.log_algo("InequalitiesPool.generate_from_polyhedron")

        if self.type_good is None:
            good = self.points_good
        elif self.type_good == "upper":
            good = upper_set(self.points_good, self.n)
        elif self.type_good == "lower":
            good = lower_set(self.points_good, self.n)
        else:
            assert 0

        p = Polyhedron(vertices=good)

        L = sorted(map(tuple, p.inequalities()))
        # https://twitter.com/SiweiSun2/status/1327973545666891777
        E = sorted(map(tuple, p.equations()))
        for eq in E:
            # >= 0
            L.append(eq)
            # <= 0
            L.append(tuple(-v for v in eq))
        # Sage outputs constant term first, rotate
        L = [tuple(eq[1:] + eq[:1]) for eq in L]

        # keep/force monotone useful eqs only
        if 1 and self.type_good == "upper":
            L2 = []
            for ineq in L:
                ineq2 = [v if v >= 0 else 0 for v in ineq[:-1]]
                ineq2.append(ineq[-1])
                if ineq2[-1] < 0:
                    ineq2 = tuple(ineq2)
                    L2.append(ineq2)
                    if ineq != ineq2:
                        log.warning(f"forcing monotone {ineq} -> {ineq2}")
            L = L2
        if 1 and self.type_good == "lower":
            L2 = []
            # a1*x1 -a2*x2 -a3*x3+ c >= 0

            for ineq in L:
                res = ineq[-1]
                ineq2 = []
                for v in ineq[:-1]:
                    if v <= 0:
                        ineq2.append(v)
                    else:
                        ineq2.append(0)
                        # res += v
                ineq2.append(res)
                if ineq2[-1] > 0:
                    ineq2 = tuple(ineq2)
                    L2.append(ineq2)
                    if ineq != ineq2:
                        log.warning(f"forcing monotone {ineq} -> {ineq2}")
            L = L2
        return self.pool_update(L, "sage.polyhedron")

