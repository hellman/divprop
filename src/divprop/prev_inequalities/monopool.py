import os
import logging
import pickle
from math import ceil
from random import sample, randrange, choices, shuffle, choice
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
from divprop.milp.symbase import LPwriter

TYPE_GOOD_SHIFTED = "shifted"
TYPE_GOOD_GENERIC = "-"

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

        if self.pool.is_monotone:
            lb = 0  # monotone => nonnegative
        else:
            lb = None

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
        bads = bads.support

        self.n_calls += 1

        LP = self.milp
        cs = [LP.add_constraint(self.i2cs[i]) for i in bads]
        res = LP.optimize(log=0)
        LP.remove_constraints(cs)

        if res is None:
            return False

        sol = LP.solutions[0]
        val_xs = tuple(sol[x] for x in self.xs)
        val_c = sol[self.c]

        # print("res", res, "sol", sol, "val_c", val_c)
        if not all(isinstance(v, int) for v in val_xs + (val_c,)):
            # keep real ineq, put the separator in the middle
            val_c -= 0.5
            pass

        ineq = val_xs + (-val_c,)
        assert all(satisfy(p, ineq) for p in self.pool.good)
        assert all(not satisfy(self.pool.i2bad[i], ineq) for i in bads)
        return ineq


class InequalitiesPool:
    log = logging.getLogger(f"{__name__}:InequalitiesPool")

    @classmethod
    def from_DenseSet_files(cls, fileprefix, checks=False):
        points_good = DenseSet.load_from_file(fileprefix + ".good.set")
        points_bad = DenseSet.load_from_file(fileprefix + ".bad.set")
        with open(fileprefix + ".type_good") as f:
            type_good = f.read().strip()
            assert type_good in ("upper", "lower", TYPE_GOOD_GENERIC)

        cls.log.info(f"points_good: {points_good}")
        cls.log.info(f" points_bad: {points_bad}")
        cls.log.info(f"  type_good: {type_good}")

        if checks:
            if type_good == "lower":
                assert points_bad <= points_good.LowerSet().Complement()
            elif type_good == "upper":
                assert points_bad <= points_good.UpperSet().Complement()
            elif type_good == TYPE_GOOD_GENERIC:
                assert (points_good & points_bad).is_empty()
                assert (points_good | points_bad).is_full()

        pool = cls(
            points_good=points_good.to_Bins(),
            points_bad=points_bad.to_Bins(),
            type_good=type_good,
        )
        assert pool.N == len(points_bad)
        return pool

    def __init__(
        self,
        points_good, points_bad, type_good=TYPE_GOOD_GENERIC, shift=None,
        use_point_prec=False,
    ):
        for p in points_bad:
            self.n = len(p)
            break

        assert type_good in ("lower", "upper", TYPE_GOOD_GENERIC)
        if type_good != TYPE_GOOD_GENERIC:
            if shift is None:
                shift = (0,) * self.n
            if type_good == "lower":
                shift = tuple_xor(shift, (1,) * self.n)
            type_good = "upper"

        if type_good == TYPE_GOOD_GENERIC:
            self.is_monotone = False
            self.shift = None
            assert shift is None

            self.bad = set(map(tuple, points_bad))
            self.good = set(map(tuple, points_good))
        else:
            self.is_monotone = True
            self.shift = shift
            assert shift is not None

            self.bad = {tuple_xor(p, shift) for p in points_bad}
            self.good = {tuple_xor(p, shift) for p in points_good}

        self._good_orig = points_good
        self._bad_orig = points_bad

        self.i2bad = sorted(self.bad)
        self.bad2i = {p: i for i, p in enumerate(self.i2bad)}
        self.N = len(self.bad)

        if use_point_prec:
            assert self.is_monotone
        self.use_point_prec = use_point_prec

        self._oracle = None
        self._system = None

    def init_system(self):
        if self.is_monotone:
            for pi, p in enumerate(self.i2bad):
                if self.use_point_prec:
                    if any(tuple_preceq(p, q) for q in self.i2bad if q != p):
                        continue
                self.gen_basic_ineq_monotone(pi, p)
        else:
            for pi, p in enumerate(self.i2bad):
                self.gen_basic_ineq_single(pi, p)

    @property
    def oracle(self):
        if not self._oracle:
            self.set_oracle(LPbasedOracle())
        return self._oracle

    def set_oracle(self, oracle):
        self._oracle = oracle
        oracle.attach_to_pool(self)

    @property
    def system(self):
        if not self._system:
            self.set_system(LazySparseSystem())
        return self._system

    def set_system(self, system):
        self._system = system
        system.attach_to_pool(self)
        self.init_system()

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

    # tbd:
    # port polyhedron
    # port subset greedy

    def _output_fsets(self, fsets):
        self.log.info(
            f"sanity checking {len(fsets)} ineqs on "
            f"{len(self.good)} good and "
            f"{len(self.bad)} bad points..."
        )
        ineqs = [self.system.solution[fset].ineq for fset in fsets]
        for q in self.good:
            assert all(satisfy(q, ineq) for ineq in ineqs)
        for q in self.bad:
            assert any(not satisfy(q, ineq) for ineq in ineqs)

        self.log.info(f"processing ineqs (shifting by {self.shift})...")
        return list(map(self._output_ineq, ineqs))

    def _output_ineq(self, ineq):
        if self.shift:
            ineq = shift_ineq(ineq, self.shift)
        return ineq

    def choose_all(self):
        fsets = list(self.system.feasible)
        return self._output_fsets(fsets)

    def choose_subset_milp(self, solver=None):
        """
        [SecITC:SasTod17]
        Choose subset optimally by optimizing MILP system.
        """
        self.log.info(f"InequalitiesPool.choose_subset_milp(solver={solver})")
        self.log.info(f"{len(self.system.feasible)} ineqs {len(self.bad)} bad points")

        i2fset = list(self.system.feasible)

        milp = MILP.minimization(solver=solver)
        n = len(i2fset)

        # xi = take i-th inequality?
        v_take_ineq = [milp.var_binary("v_take_ineq%d" % i) for i in range(n)]

        by_bad = [[] for _ in range(self.N)]
        for i, fset in enumerate(i2fset):
            for q in fset:
                by_bad[q].append(v_take_ineq[i])

        # each bad point is removed by at least one ineq
        for lst in by_bad:
            assert lst, "no solutions"
            milp.add_constraint(sum(lst) >= 1)

        # minimize number of ineqs
        milp.set_objective(sum(v_take_ineq))

        self.log.info(f"solving milp with {n} variables, {self.N} constraints")

        # show log for large problems
        res = milp.optimize(log=(n >= 10000))
        assert res is not None
        milpsol = milp.solutions[0]
        self.log.info(f"objective {res}")

        # print("obj", res, "milpsol", milpsol)

        ineqs = [
            i2fset[i] for i, take in enumerate(v_take_ineq) if milpsol[take]
        ]
        return self._output_fsets(ineqs)

    def choose_subset_greedy_once(
            self, eps=0,
            lp_snapshot_step=None,
            lp_snapshot_format=None,
        ):
        self.log.debug("preparing greedy")

        order = list(self.system.feasible)
        M = len(order)

        by_fset = {j: set(order[j]) for j in range(M)}
        by_point = {i: [] for i in range(self.N)}
        for j, fset in enumerate(order):
            for i in fset:
                by_point[i].append(j)

        self.log.debug("running greedy")

        n_removed = 0
        Lstar = set()
        while by_fset:
            max_remove = max(map(len, by_fset.values()))
            assert max_remove >= 1

            cands = [
                j for j, rem in by_fset.items()
                if len(rem) >= max_remove - eps
            ]
            j = choice(cands)

            Lstar.add(order[j])
            n_removed += max_remove

            for i in order[j]:
                js = by_point.get(i, ())
                if js:
                    for j2 in js:
                        s = by_fset.get(j2)
                        if s:
                            s.discard(i)
                            if not s:
                                del by_fset[j2]
                    del by_point[i]
            assert j not in by_fset

            lb = len(Lstar) + ceil(self.N / max_remove)
            self.log.debug(
                f"removing {max_remove} points: "
                f"cur {len(Lstar)} ineqs, left {len(by_fset)} ineqs"
                f"removed {n_removed}/{self.N} points; "
                f"bound {lb} ineqs"
            )

            if lp_snapshot_step and len(Lstar) % lp_snapshot_step == 0:
                self.do_greedy_snapshot(
                    order, Lstar, by_fset, by_point,
                    lp_snapshot_format
                )

        self.log.debug(f"greedy result: {len(Lstar)}")
        return self._output_fsets(Lstar)

    def do_greedy_snapshot(
            self, fset_order, Lstar, by_fset, by_point,
            lp_snapshot_format,
        ):
        prefix = lp_snapshot_format % dict(
            selected=len(Lstar),
            remaining=len(fset_order)
        )

        self.log.info(
            f"snapshot to {prefix} "
            f"(selected={len(Lstar)}, points_left={len(by_point)})"
        )

        with open(prefix + ".meta", "w") as f:
            for i, fset in enumerate(fset_order):
                if fset not in Lstar and i not in by_fset:
                    continue
                ineq = self._output_ineq(self.system.solution[fset].ineq)
                print(
                    i,
                    ":".join(map(str, fset)),
                    ":".join(map(str, ineq)),
                    int(fset in Lstar),
                    file=f
                )

        lp = LPwriter(filename=prefix + ".lp")

        var_fset = {}
        for j in by_fset:
            var_fset[j] = "x%d" % j  # take ineq j

        lp.objective(
            objective=lp.sum(var_fset.values()),
            sense="minimize",
        )

        for i, js in by_point.items():
            lp.constraint(lp.sum(var_fset[j] for j in js) + " >= 1")

        lp.binaries(var_fset.values())
        lp.close()

    def choose_subset_greedy(self, iterations=10, eps=0):
        self.log.info(
            f"InequalitiesPool.choose_subset_greedy("
            f"iterations={iterations},eps={eps}"
            ")"
        )
        self.log.info(
            f"{len(self.system.feasible)} ineqs {len(self.bad)} bad points"
        )

        best = len(self.system.feasible) + 1, None
        for itr in range(iterations):
            Lstar = self.choose_subset_greedy_once(eps=eps)

            cur = len(Lstar), Lstar
            self.log.info(f"itr #{itr}: {cur[0]} ineqs")
            if cur < best:
                best = cur
        log.info(f"best: {best[0]} inequalities")
        assert best[1] is not None
        return best[1]

    def point_prec_max_set(self, fset):
        res = set()
        qs = [self.i2bad[i] for i in fset]
        for q in qs:
            if not any(tuple_preceq(q, p) and q != p for p in qs):
                res.add(self.bad2i[q])
        return self.system.encode_bad_subset(res)

    def point_prec_lower_set(self, fset):
        res = set()
        qs = [self.i2bad[i] for i in fset]
        for pi, p in enumerate(self.i2bad):
            if any(tuple_preceq(p, q) for q in qs):
                res.add(pi)
        return self.system.encode_bad_subset(res)


class LazySparseSystem:
    log = logging.getLogger(f"{__name__}:LazySparseSystem")

    def __init__(self, sysfile=None):
        self.sysfile = sysfile

    def attach_to_pool(self, pool):
        self.pool = pool
        self.N = int(self.pool.N)
        self.feasible = GrowingLowerFrozen(self.N)
        self.infeasible = GrowingUpperFrozen(self.N)
        self.solution = {}
        if self.sysfile and os.path.exists(self.sysfile):
            self.load()

    def refresh(self, extremize=True):
        self.log.info(f"refreshing system, extremize={extremize}")
        if extremize:
            self.log_info()
            self.feasible.do_MaxSet()
            self.infeasible.do_MinSet()
        try:
            self.save()
        except KeyboardInterrupt:
            log.error("interrupted saving! trying again, please be patient")
            self.save()
            quit()
        # self.log_info()  # save triggers log info

    def save(self):
        if self.sysfile:
            self.save_to_file(self.sysfile)

    def load(self):
        if self.sysfile:
            self.load_from_file(self.sysfile)

    def load_from_file(self, filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        self.__dict__.update(data)
        self.log.info(f"loaded state from file {filename}")
        self.log_info()

    def save_to_file(self, filename):
        data = self.__dict__.copy()
        del data["pool"]
        del data["sysfile"]
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        self.log.info(f"saved state to file {filename}")
        self.log_info()

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

        if self.pool.use_point_prec:
            fset = self.pool.point_prec_lower_set(fset)

        self.feasible.add(fset)

        if sol is not None:
            self.solution[fset] = sol

    def add_infeasible(self, fset):
        assert isinstance(fset, frozenset)

        if self.pool.use_point_prec:
            fset = self.pool.point_prec_max_set(fset)

        self.infeasible.add(fset)

    def remove_redundant(self):
        self.feasible.do_MaxSet()

    def log_info(self):
        for (name, s) in [
            ("feasible", self.feasible),
            ("infeasible", self.infeasible),
        ]:
            freq = Counter(len(v) for v in s)
            freqstr = " ".join(
                f"{sz}:{cnt}" for sz, cnt in sorted(freq.items())
            )
            cachestr = len(s.cache) if s.cache is not None else None
            log.info(f"   {name}: {len(s)}: {freqstr}, cache {cachestr}")


def tuple_xor(t, mask):
    assert len(t) == len(mask)
    return tuple(a ^ b for a, b in zip(t, mask))


def tuple_preceq(a, b):
    return all(aa <= bb for aa, bb in zip(a, b))


def invert_ineq(ineq: tuple):
    # a1x1 + a2x2 + ... + c >= 0
    # a1(1-x1) + a2(1-x2) + ... + c >= 0
    # -a1x1 -a2x2 - ... + c + a1 + a2 - ... >= 0
    val = ineq[-1] + sum(ineq[:-1])
    ineq = tuple(-ai for ai in ineq[:-1]) + (val,)
    return ineq


def shift_ineq(ineq: tuple, shift: tuple):
    assert len(ineq) == len(shift) + 1
    val = ineq[-1]
    ineq2 = []
    for a, s in zip(ineq, shift):
        if s:
            ineq2.append(-a)
            val += a
        else:
            ineq2.append(a)
    ineq2.append(val)
    return tuple(ineq2)