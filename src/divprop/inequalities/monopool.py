import logging
from random import sample, randrange, choices
from collections import Counter, namedtuple, defaultdict
from math import gcd

from tqdm import tqdm
from binteger import Bin

from .base import (
    inner, satisfy,
    MixedIntegerLinearProgram, MIPSolverException,
    Polyhedron,
)
from .random_group_cut import RandomGroupCut
from .gem_cut import GemCut


log = logging.getLogger(__name__)


def notpoint(p):
    assert 0 <= min(p) <= max(p) <= 1
    return tuple(1 ^ v for v in p)


def neibs_up_tuple(p):
    p = list(p)
    for i in range(len(p)):
        if p[i] == 0:
            p[i] = 1
            yield tuple(p)
            p[i] = 0


IneqInfo = namedtuple("SourcedIneq", ("ineq", "source", "state"))


class Generator:
    def generate(self, pool):
        pass


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
        for u in pool.lo:
            for supu in neibs_up_tuple(u):
                hats[supu].append(u)

        for top, subs in hats.items():
            ineq = self.make_ineq(top, subs)
            fset = pool.LSL.encode_fset(
                pool.lo2i[q] for q in subs
            )
            pool.LSL.add_feasible(
                fset,
                ineq=ineq,
                source="hat",
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
        ev_good = min(inner(p, lin) for p in pool.hi)
        fset = pool.LSL.encode_fset(
            i for i, q in enumerate(pool.i2lo) if inner(q, lin) < ev_good
        )
        if fset:
            print("lin", lin)
            print("good", min(inner(p, lin) for p in pool.hi), max(inner(p, lin) for p in pool.hi))
            print(" bad", min(inner(p, lin) for p in pool.lo), max(inner(p, lin) for p in pool.lo))
            print(fset)
        # lin. comb. >= ev_good
        ineq = tuple(lin) + (-ev_good,)
        pool.LSL.add_feasible(fset, ineq=ineq, source="random_ineq")


class DFS(Generator):
    def __init__(self):
        pass

    def generate(self, pool):
        while pool.LSL.feasible_open:
            for fset in pool.LSL.feasible_open:
                break
            fset2 = pool.LSL.get_next_unknown_neighbour(fset)
            if not fset2:
                continue
            ineq = pool.oracle.query(fset2)
            # print("visit", fset, "->", fset2, ":", ineq)
            if not ineq:
                pool.LSL.add_infeasible(fset2)
            else:
                pool.LSL.add_feasible(
                    fset2,
                    ineq=ineq,
                    source="DFS",
                )


class LPbasedOracle:
    def __init__(self, solver=None):
        self.solver = solver
        self.n_calls = 0

    def attach_to_pool(self, pool):
        self.pool = pool
        self._prepare_constraints()

    def _prepare_constraints(self):
        self.model = MixedIntegerLinearProgram(solver=self.solver)
        self.var = self.model.new_variable(real=True, nonnegative=True)
        self.xs = [self.var["x%d" % i] for i in range(self.pool.n)]
        self.c = self.var["c"]

        for p in self.pool.hi:
            self.model.add_constraint(inner(p, self.xs) >= self.c)

        cs = {}
        for q in self.pool.lo:
            cs[q] = inner(q, self.xs) <= self.c - 1

        self.cs_per_lo = cs

    def query(self, bads):
        self.n_calls += 1
        LP = self.model.__copy__()

        for i in bads:
            q = self.pool.lo[i]
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

        assert all(satisfy(p, ineq) for p in self.pool.hi)
        assert all(not satisfy(self.pool.lo[i], ineq) for i in bads)
        return ineq


class MonotoneInequalitiesPool:
    def __init__(self, points_good, points_bad, type_good=None):
        for p in points_bad:
            self.n = len(p)
            break

        assert type_good in ("lower", "upper")
        self.type_good = type_good
        if type_good == "upper":
            self.lo = sorted(map(tuple, points_bad))
            self.hi = sorted(map(tuple, points_good))
        else:
            self.lo = sorted(map(notpoint, points_good))
            self.hi = sorted(map(notpoint, points_bad))

        self.i2lo = list(self.lo)
        self.lo2i = {p: i for i, p in enumerate(self.i2lo)}
        self.N = len(self.lo)

        self.LSL = SparseLowerSetLearn(self.N)

        # initialize
        for pi, p in enumerate(self.i2lo):
            self.gen_basic_ineq(pi, p)

        self._oracle = None

    @property
    def oracle(self):
        if not self._oracle:
            self.oracle = LPbasedOracle()
        return self._oracle

    @oracle.setter
    def oracle(self, oracle):
        self._oracle = oracle
        oracle.attach_to_pool(self)

    # python impl goal: smallish s-boxes
    # + maybe OK random ineqs / pure hats

    def gen_basic_ineq(self, pi, p):
        fset = self.LSL.encode_fset((pi,))
        # does not belong to LowerSet({p})
        # <=>
        # sum coords with p_i = 0 >= 1
        ineq = tuple(1 if x == 0 else 0 for x in p) + (-1,)
        self.LSL.add_feasible(fset, ineq=ineq, source="basic")

    def gen_random_inequality(self, max_coef=100, exp=-0.5):
        RandomPlaneCut(max_coef, exp).generate(self)

    def gen_hats(self):
        Hats().generate(self)

    def gen_dfs(self):
        DFS().generate(self)

    # tbd:
    # port polyhedron
    # port subset greedy

    def try_extend_highest(self):
        '''
        extend

        extending top improves FEASIBLE
        but extending bot improves INFEASIBLE
        both are useful

        infeasible seems cutting off more checks
        BFS
        '''

    def try_extend_lowest(self):
        dunno

    def choose_subset_milp(self, solver=None):
        """
        [SecITC:SasTod17]
        Choose subset optimally by optimizing MILP system.
        """
        log.info(f"InequalitiesPool.choose_subset_milp(solver={solver})")
        log.info(f"{len(self.LSL.feasible)} ineqs {len(self.lo)} bad points")

        L = list(self.LSL.feasible)
        self.check_good(L)  # to avoid surprises
        eq2i = {eq: i for i, eq in enumerate(L)}

        model = MixedIntegerLinearProgram(maximization=False, solver=solver)
        var = model.new_variable(binary=True, nonnegative=True)
        n = len(L)

        # xi = take i-th inequality?
        take_eq = [var["take_eq%d" % i] for i in range(n)]

        by_bad = {q: [] for q in self.points_bad}
        for eq, (source, covered) in self.pool.items():
            v_take_eq = take_eq[eq2i[eq]]
            for q in covered:
                by_bad[q].append(v_take_eq)

        # each bad point is removed by at least one ineq
        for q, lst in by_bad.items():
            assert lst, "no solutions"
            model.add_constraint(sum(lst) >= 1)

        # minimize number of ineqs
        model.set_objective(sum(take_eq))
        log.info(
            f"solving model with {n} variables, "
            f"{len(self.points_bad)} constraints"
        )

        # show log for large problems
        model.solve(log=(n >= 10000))

        Lstar = []
        for take, eq in zip(take_eq, L):
            if model.get_values(take):
                Lstar.append(eq)
        self.check(Lstar)
        return Lstar


# TBD: base class interface, dense class for smallish N (<100? < 1000?)
class SparseLowerSetLearn:
    def __init__(self, N):
        self.N = int(N)
        self.coprimes = [
            i for i in range(1, self.N)
            if gcd(i, self.N) == 1
        ]

        # TBD: optimization by hw?
        # { set of indexes of covered bad points (hi)
        #   :
        #   ineq, source, state }
        # state = int index of last unchecked bit up?
        self.feasible = {}
        self.feasible_open = set()

        # { set of indexes of infeasible to cover bad points }
        self.infeasible = set()

        self._order_sbox = sample(range(self.N), self.N)

    def log_info(self):
        log.info(
            "stat:"
            f" good max-set {len(self.feasible)}"
            f" ({len(self.feasible) - len(self.feasible_open)} final)"
            f" bad min-set {len(self.infeasible)}"
        )
        freq = Counter(map(len, self.feasible))
        log.info(
            "freq: "
            + " ".join(f"{sz}:{cnt}" for sz, cnt in sorted(freq.items()))
        )

    def encode_fset(self, fset):
        return frozenset(map(int, fset))

    def _hash(self, fset):
        mask = 2**64-1
        res = 0x9c994e7c9068e947
        for v in fset:
            res ^= v
            res *= 0xf7ace5e55fd1c1ad
            res &= mask
            res ^= res >> 17
            res &= mask
        return res

    def _get_real_index(self, h, i):
        return i
        i += h + 0x28a5e1f1
        i %= self.N
        i = self._order_sbox[i]
        i += h + 0xb5520e03
        i %= self.N
        i = self._order_sbox[i]
        i += h + 0xb12dcbaa
        i %= self.N
        return i

    def is_already_feasible(self, fset):
        # quick check
        if fset in self.feasible:
            return True
        # is in feasible lowerset?
        for fset2 in self.feasible:
            if fset <= fset2:
                return True
        return False

    def is_already_infeasible(self, fset):
        # quick check
        if fset in self.infeasible:
            return True
        # is in infeasible upperset?
        for fset2 in self.infeasible:
            if fset2 <= fset:
                return True
        return False

    def add_feasible(self, fset, ineq, source, check=True):
        if check and self.is_already_feasible(fset):
            return
        # remove existing redundant
        self.feasible = {
            fset2: info2
            for fset2, info2 in self.feasible.items()
            if not (fset2 <= fset)
        }
        self.feasible_open = {
            fset2
            for fset2 in self.feasible_open
            if not (fset2 <= fset)
        }
        self.feasible[fset] = IneqInfo(
            ineq,
            source="basic",
            state=(self._hash(fset), 0)
        )
        self.feasible_open.add(fset)

    def add_infeasible(self, fset, check=True):
        if check and self.is_already_infeasible(fset):
            return
        # remove existing redundant
        self.infeasible = {
            fset2
            for fset2 in self.infeasible
            if not (fset2 >= fset)
        }
        self.infeasible.add(fset)

    def get_next_unknown_neighbour(self, fset):
        assert fset in self.feasible_open
        h, i = self.feasible[fset].state
        good = 0
        while i < self.N:
            ii = self._get_real_index(h, i)
            i += 1
            if ii not in fset:
                fset2 = fset | {ii}
                if fset2 in self.infeasible:
                    continue
                if self.is_already_feasible(fset2):
                    continue
                if self.is_already_infeasible(fset2):
                    continue
                good = 1
                break

        if i >= self.N:
            i = None
            self.feasible_open.remove(fset)
        self.feasible[fset] = self.feasible[fset]._replace(state=(h, i))

        if good:
            return fset2
