from random import choice
from math import ceil
from enum import Enum

from binteger import Bin

from subsets import DenseSet, SparseSet
from subsets.learn import LowerSetLearn, Oracle, ExtraPrec_LowerSet

from optisolveapi.milp import MILP
from optisolveapi.milp.symbase import LPwriter

from divprop.logs import logging

from .base import inner, satisfy


class TypeGood(Enum):
    LOWER = "lower"
    UPPER = "upper"
    GENERIC = "-"


class InequalitiesPool:
    log = logging.getLogger(f"{__name__}:InequalitiesPool")

    @classmethod
    def from_DenseSet_files(cls, fileprefix, checks=True, **opts):
        sysfile = fileprefix + ".system"

        points_good = DenseSet.load_from_file(fileprefix + ".good.set")
        points_bad = DenseSet.load_from_file(fileprefix + ".bad.set")

        with open(fileprefix + ".type_good") as f:
            type_good = TypeGood(f.read().strip())

        cls.log.info(f"points_good: {points_good}")
        cls.log.info(f" points_bad: {points_bad}")
        cls.log.info(f"  type_good: {type_good}")

        if checks:
            if type_good == TypeGood.LOWER:
                assert points_bad <= points_good.LowerSet().Complement()
            elif type_good == TypeGood.UPPER:
                assert points_bad <= points_good.UpperSet().Complement()
            elif type_good == TypeGood.GENERIC:
                assert (points_good & points_bad).is_empty()
                # not necessary ("don't care" points)
                # assert (points_good | points_bad).is_full()

        opts.setdefault("sysfile", sysfile)
        pool = cls(
            points_good=points_good.to_Bins(),
            points_bad=points_bad.to_Bins(),
            type_good=type_good,
            **opts
        )
        assert pool.N == len(points_bad)
        return pool

    def __init__(
        self,
        points_good: tuple,
        points_bad: tuple,
        type_good: TypeGood = TypeGood.GENERIC,
        use_point_prec=False,
        sysfile=None,
        oracle=None,
        pre_shift=0,
    ):
        for p in points_bad:
            self.n = len(p)
            break

        self.bad = {Bin(v, self.n) for v in points_bad}
        self.good = {Bin(v, self.n) for v in points_good}
        self._good_orig = self.good
        self._bad_orig = self.bad

        assert pre_shift is None or isinstance(pre_shift, (int, Bin))
        pre_shift = Bin(pre_shift, self.n)

        if type_good == TypeGood.GENERIC:
            self.is_monotone = False
            self.shift = None
            assert pre_shift in (0, None)

        elif type_good == TypeGood.LOWER:
            self.is_monotone = True
            self.shift = pre_shift ^ ~Bin(0, self.n)
            self.bad = {~v for v in self.bad}
            self.good = {~v for v in self.good}

        elif type_good == TypeGood.UPPER:
            self.is_monotone = True
            self.shift = pre_shift

        self.i2bad = sorted(self.bad)
        self.bad2i = {p: i for i, p in enumerate(self.i2bad)}
        self.N = len(self.bad)

        if use_point_prec:
            assert self.is_monotone
            ep = ExtraPrec_LowerSet(
                int2point=self.i2bad,
                point2int=self.bad2i,
            )
        else:
            ep = None
        self.use_point_prec = use_point_prec

        if oracle is None:
            oracle = LPbasedOracle()
        self.oracle = oracle
        self.oracle.set_pool(self)

        self.system = LowerSetLearn(
            n=self.N,
            file=sysfile,
            oracle=self.oracle,
            extra_prec=ep,
        )

    # tbd:
    # port polyhedron

    def _output_results(self, vecs):
        self.log.info(
            f"sanity checking {len(vecs)} ineqs on "
            f"{len(self.good)} good and "
            f"{len(self.bad)} bad points..."
        )
        ineqs = [self.system.meta[vec] for vec in vecs]
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
        self.log.info(
            "InequalitiesPool.choose_all()"
        )
        return self._output_results(list(self.system.iter_lower()))

    def create_subset_milp(self, solver=None):
        """
        [SecITC:SasTod17]
        Choose subset optimally by optimizing MILP system.
        """
        self.log.info(
            f"InequalitiesPool.create_subset_milp(solver={solver})"
        )
        self.log.info(
            f"{self.system.n_lower()} ineqs {len(self.bad)} bad points"
        )

        vec_order = list(self.system.iter_lower())

        milp = MILP.minimization(solver=solver)
        n = len(vec_order)

        # xi = take i-th inequality?
        v_take_ineq = [milp.var_binary("v_take_ineq%d" % i) for i in range(n)]

        by_bad = [[] for _ in range(self.N)]
        for i, vec in enumerate(vec_order):
            for q in vec:
                by_bad[q].append(v_take_ineq[i])

        # each bad point is removed by at least one ineq
        for lst in by_bad:
            assert lst, "no solutions"
            milp.add_constraint(sum(lst) >= 1)

        # minimize number of ineqs
        milp.set_objective(sum(v_take_ineq))
        return v_take_ineq, vec_order, milp

    def write_subset_milp(self, filename, solver=None):
        v_take_ineq, vec_order, milp = self.create_subset_milp(solver=solver)
        self.log.info(
            f"saving LP with {self.N} variables (per ineq), "
            f"{self.n} constraints (per bad point) to {filename}"
        )
        milp.write_lp(filename)

    def choose_subset_milp(self, lp_output=None, solver=None):
        v_take_ineq, vec_order, milp = self.create_subset_milp(solver=solver)

        if lp_output:
            self.log.info(
                f"saving LP with {self.N} variables (per ineq), "
                f"{self.n} constraints (per bad point) to {lp_output}"
            )
            milp.write_lp(lp_output)

        self.log.info(
            f"solving milp with {self.n} variables, {self.N} constraints"
        )

        # show log for large problems
        res = milp.optimize(log=(self.N >= 10000))
        assert res is not None, "insufficient inequalities pool?"
        milpsol = milp.solutions[0]
        self.log.info(f"objective {res}")

        ineqs = [
            vec_order[i] for i, take in enumerate(v_take_ineq) if milpsol[take]
        ]
        return self._output_results(ineqs)

    def choose_subset_greedy_once(
            self, eps=0,
            lp_snapshot_step=None,
            lp_snapshot_format=None,
        ):
        self.log.debug("preparing greedy")

        # tbd update for non-prime option (clean up or ... ?)
        vec_order = list(self.system.iter_lower())
        M = len(vec_order)

        by_vec = {j: set(vec_order[j]) for j in range(M)}
        by_point = {i: [] for i in range(self.N)}
        for j, fset in enumerate(vec_order):
            for i in fset:
                by_point[i].append(j)

        self.log.debug("running greedy")

        n_removed = 0
        Lstar = set()
        while by_vec:
            max_remove = max(map(len, by_vec.values()))
            assert max_remove >= 1

            cands = [
                j for j, rem in by_vec.items()
                if len(rem) >= max_remove - eps
            ]
            j = choice(cands)

            Lstar.add(vec_order[j])
            n_removed += max_remove

            for i in vec_order[j]:
                js = by_point.get(i, ())
                if js:
                    for j2 in js:
                        s = by_vec.get(j2)
                        if s:
                            s.discard(i)
                            if not s:
                                del by_vec[j2]
                    del by_point[i]
            assert j not in by_vec

            lb = len(Lstar) + ceil(self.N / max_remove)
            self.log.debug(
                f"removing {max_remove} points: "
                f"cur {len(Lstar)} ineqs, left {len(by_vec)} ineqs"
                f"removed {n_removed}/{self.N} points; "
                f"bound {lb} ineqs"
            )

            if lp_snapshot_step and len(Lstar) % lp_snapshot_step == 0:
                self.do_greedy_snapshot(
                    vec_order, Lstar, by_vec, by_point,
                    lp_snapshot_format
                )

        self.log.debug(f"greedy result: {len(Lstar)}")
        return self._output_results(Lstar)

    def do_greedy_snapshot(
            self, vec_order, Lstar, by_vec, by_point,
            lp_snapshot_format,
        ):
        prefix = lp_snapshot_format % dict(
            selected=len(Lstar),
            remaining=len(vec_order)
        )

        self.log.info(
            f"snapshot to {prefix} "
            f"(pre-selected={len(Lstar)}, points_left={len(by_point)})"
        )

        with open(prefix + ".meta", "w") as f:
            for i, fset in enumerate(vec_order):
                if fset not in Lstar and i not in by_vec:
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
        for j in by_vec:
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
            f"{self.system.n_lower()} ineqs {len(self.bad)} bad points"
        )

        best = float("+inf"), None
        for itr in range(iterations):
            Lstar = self.choose_subset_greedy_once(eps=eps)

            cur = len(Lstar), Lstar
            self.log.info(f"itr #{itr}: {cur[0]} ineqs")
            if cur < best:
                best = cur
        self.log.info(f"best: {best[0]} inequalities")
        assert best[1] is not None
        return best[1]


class LPbasedOracle(Oracle):
    def __init__(self, solver=None):
        super().__init__()

        self.solver = solver
        self.n_calls = 0
        self.milp = None

    def set_pool(self, pool):
        self.pool = pool

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

    def _query(self, bads: SparseSet):
        assert isinstance(bads, SparseSet)
        if not bads:
            # trivial inequality
            ineq = (0,) * self.pool.n + (0,)
            return True, ineq

        if self.milp is None:
            self._prepare_constraints()

        self.n_calls += 1

        LP = self.milp
        cs = [LP.add_constraint(self.i2cs[i]) for i in bads]
        res = LP.optimize(log=0)
        LP.remove_constraints(cs)

        if res is None:
            return False, None

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
        return True, ineq


def shift_ineq(ineq: tuple, shift: Bin):
    shift = shift.tuple
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
