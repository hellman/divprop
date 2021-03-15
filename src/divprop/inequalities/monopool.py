import logging
from random import sample, randrange, choices, shuffle
from collections import Counter, namedtuple, defaultdict
from itertools import combinations

from binteger import Bin

from .base import (
    inner, satisfy,
    MixedIntegerLinearProgram, MIPSolverException,
    Polyhedron,
)

from divprop.subsets import (
    DenseSet,
    neibs_up_tuple, not_tuple, support_int_le,
    WeightedFrozenSets,
    GrowingLowerFrozen,
    GrowingUpperFrozen,
)

from divprop.milp import MILP


log = logging.getLogger(__name__)


IneqInfo = namedtuple("IneqInfo", ("ineq", "source"))


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


class LPbasedOracle:
    def __init__(self, solver=None):
        self.solver = solver
        self.n_calls = 0

    def attach_to_pool(self, pool):
        self.pool = pool
        self._prepare_constraints()

    def _prepare_constraints(self):
        self.milp = MILP.maximization(solver=self.solver)

        if self.pool.type_good is None:  # not monotone
            lb = None
        else:
            lb = 0  # monotone => nonnegative

        # set ub = 1000+ ? ...
        self.xs = []
        for i in range(self.pool.n):
            self.xs.append(self.milp.var_real("x%d" % i, lb=lb, ub=None))
        self.c = self.milp.var_real("c", lb=lb, ub=None)

        for p in self.pool.good:
            self.milp.add_constraint(inner(p, self.xs) >= self.c)

        self.i2cs = []
        for q in self.pool.i2bad:
            self.i2cs.append(inner(q, self.xs) <= self.c - 1)

    def query(self, bads: Bin):
        assert isinstance(bads, Bin)
        bads = bads.support()

        self.n_calls += 1

        LP = self.milp
        cs = [LP.add_constraint(self.i2cs[i]) for i in bads]
        res = LP.optimize()
        LP.remove_constraints(cs)

        if res is None:
            return False

        sol = LP.solutions[0]
        val_xs = tuple(sol[x] for x in self.xs)
        val_c = sol[self.c]

        if not all(isinstance(v, int) for v in val_xs + (val_c,)):
            # keep real ineq, put the separator in the middle
            val_c -= 0.5

        ineq = val_xs + (-val_c,)
        assert all(satisfy(p, ineq) for p in self.pool.good)
        assert all(not satisfy(self.pool.i2bad[i], ineq) for i in bads)
        return ineq


class InequalitiesPool:
    log = logging.getLogger(f"{__name__}:InequalitiesPool")

    @classmethod
    def from_DenseSet_files(cls, fileprefix, system=None, checks=False):
        points_good = DenseSet.load_from_file(fileprefix + ".good.set")
        points_bad = DenseSet.load_from_file(fileprefix + ".bad.set")
        with open(fileprefix + ".type_good") as f:
            type_good = f.read().strip()
            assert type_good in ("upper", "lower", None)

        cls.log.info(f"points_good: {points_good}")
        cls.log.info(f" points_bad: {points_bad}")
        cls.log.info(f"  type_good: {type_good}")

        if checks:
            if type_good == "lower":
                assert points_bad <= points_good.LowerSet().Complement()
            elif type_good == "upper":
                assert points_bad <= points_good.UpperSet().Complement()

        pool = cls(
            points_good=points_good.to_Bins(),
            points_bad=points_bad.to_Bins(),
            type_good=type_good,
            system=system,
        )
        assert pool.N == len(points_bad)
        return pool

    def __init__(self, points_good, points_bad, type_good=None, system=None):
        for p in points_bad:
            self.n = len(p)
            break

        assert type_good in ("lower", "upper", None)
        self.type_good = type_good
        if type_good == "upper":
            self.bad = set(map(tuple, points_bad))
            self.good = set(map(tuple, points_good))
        else:
            # ensure good is an upper set
            self.bad = set(map(not_tuple, points_bad))
            self.good = set(map(not_tuple, points_good))

        self.i2bad = sorted(self.bad)
        self.bad2i = {p: i for i, p in enumerate(self.i2bad)}
        self.N = len(self.bad)

        self.system = system if system else LazySparseSystem()
        self.system.init(pool=self)

        if self.type_good in ("lower", "upper"):
            # initialize
            for pi, p in enumerate(self.i2bad):
                self.gen_basic_ineq(pi, p)

        self._oracle = None

    @property
    def oracle(self):
        if not self._oracle:
            self.oracle = LPbasedOracle()
        return self._oracle

    def set_oracle(self, oracle):
        self._oracle = oracle
        oracle.attach_to_pool(self)

    def gen_basic_ineq(self, pi, p):
        fset = self.system.encode_bad_subset((pi,))
        # does not belong to LowerSet({p})
        # <=>
        # sum coords with p_i = 0 >= 1
        ineq = tuple(1 if x == 0 else 0 for x in p) + (-1,)

        self.system.add_feasible(
            fset,
            sol=IneqInfo(ineq=ineq, source="basic-mono"),
        )

    def gen_random_inequality(self, max_coef=100, exp=-0.5):
        RandomPlaneCut(max_coef, exp).generate(self)

    def gen_hats(self):
        Hats().generate(self)

    # tbd:
    # port polyhedron
    # port subset greedy

    def choose_subset_milp(self, solver=None):
        """
        [SecITC:SasTod17]
        Choose subset optimally by optimizing MILP system.
        """
        self.log.info(f"InequalitiesPool.choose_subset_milp(solver={solver})")
        self.log.info(f"{len(self.system.feasible)} ineqs {len(self.bad)} bad points")

        L = list(self.system.feasible)
        self.check_good(L)  # to avoid surprises
        eq2i = {eq: i for i, eq in enumerate(L)}

        milp = MILP.minimization(solver=solver)
        n = len(L)

        # xi = take i-th inequality?
        take_eq = [milp.var_binary("take_eq%d" % i) for i in range(n)]

        by_bad = {q: [] for q in self.points_bad}
        for eq, (source, covered) in self.pool.items():
            v_take_eq = take_eq[eq2i[eq]]
            for q in covered:
                by_bad[q].append(v_take_eq)

        # each bad point is removed by at least one ineq
        for q, lst in by_bad.items():
            assert lst, "no solutions"
            milp.add_constraint(sum(lst) >= 1)

        # minimize number of ineqs
        milp.set_objective(sum(take_eq))

        self.log.info(
            f"solving milp with {n} variables, "
            f"{len(self.points_bad)} constraints"
        )

        # show log for large problems
        milp.debug = 1
        res = milp.optimize(log=(n >= 10000))
        assert res is not None
        sol = milp.solutions[0]
        print("RES", res, "SOL", sol)

        Lstar = []
        for take, eq in zip(take_eq, L):
            if sol[take]:
                Lstar.append(eq)
        self.check(Lstar)
        return Lstar


class LazySparseSystem:
    log = logging.getLogger(f"{__name__}:LazySparseSystem")

    def init(self, pool):
        self.pool = pool
        self.N = int(self.pool.N)
        self.feasible = GrowingLowerFrozen(self.N)
        self.infeasible = GrowingUpperFrozen(self.N)
        self.solution = {}

    # NAIVE
    def is_already_feasible(self, v):
        # quick check
        if v in self.feasible_cache:
            return True
        # is in feasible lowerset?
        for u in self.feasible:
            # v <= u
            if v & u == v:
                return True
        return False

    def is_already_infeasible(self, v):
        # quick check
        if v in self.infeasible_cache:
            return True
        # is in infeasible upperset?
        for u in self.infeasible:
            # v >= u
            if v | u == v:
                return True
        return False

    def clean_solution(self):
        todel = [k for k in self.solution if k not in self.feasible]
        for k in todel:
            del self.solution[k]

    def encode_bad_subset(self, indexes):
        indexes = frozenset(indexes)
        assert all(0 <= i < self.N for i in indexes)
        return indexes

    def add_feasible(self, fset, sol=None):
        assert isinstance(fset, frozenset)
        self.feasible.add(fset)

        if sol is not None:
            self.solution[fset] = sol

    def add_infeasible(self, fset):
        assert isinstance(fset, frozenset)
        self.infeasible.add(fset)

    def remove_redundant(self):
        self.feasible.do_MaxSet()

    def log_info(self):
        self.log.info("stat:")
        for (name, s) in [
            ("feasible", self.feasible),
            ("infeasible", self.infeasible),
        ]:
            freq = Counter(len(v) for v in s)
            freqstr = " ".join(
                f"{sz}:{cnt}" for sz, cnt in sorted(freq.items())
            )
            log.info(f"   {name}: {len(s)}: {freqstr}")
