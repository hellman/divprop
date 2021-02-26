import logging
from binteger import Bin
from random import randint, choice, shuffle
from collections import Counter

# sage/pure python compatibility
try:
    import sage.all
    from sage.numerical.mip import MixedIntegerLinearProgram
    from sage.numerical.mip import MIPSolverException
    from sage.all import Polyhedron
    is_sage = True
except ImportError:
    is_sage = False

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def inner(a, b):
    return sum(aa * bb for aa, bb in zip(a, b))


def satisfy(pt, eq):
    """
    Inequality format:
    (a0, a1, a2, ..., a_{n-1}, c)
    a0*x0 + a1*x1 + ... + a_{n-1}*x_{n-1} + c >= 0
    """
    assert len(pt) + 1 == len(eq)
    return inner(pt, eq) + eq[-1] >= 0


class LinearSeparator:
    """
    Algorithm to separate a lower set from an upper set (must be disjoint)
    by linear inequalities ("cuts").
    Sets may be given by their extremes (maxset and minset).

    Algorithm:
        1. choose incrementally a set of bad points
        2. try to find a linear functional separating them from good points
           (solving linear inequalities over reals)

    Algorithm is implemented for removing 'lo' while keeping 'hi'.
    For the other way around, vectors are flipped and swapped and
    output inequalities are adapted.
    """
    def __init__(self, lo, hi, inverted=False, solver="GLPK"):
        """
        pairwise is an old deprecated method
        where instead of having explicit separator constant c,
        the inequalities were <p,x> >= <q,x> for each p in good, q in bad
        (quadratic number).
        Seems not be useful, as introducing 1 variable for c reduces the number
        of inequalities significantly.
        """
        self.inverted = inverted
        if self.inverted:
            self.lo = [tuple(a ^ 1 for a in p) for p in hi]
            self.hi = [tuple(a ^ 1 for a in p) for p in lo]
        else:
            self.lo = [tuple(p) for p in lo]
            self.hi = [tuple(p) for p in hi]
        assert self.lo and self.hi
        self.n = len(self.lo[0])
        self.solver = solver
        self._prepare_constraints()
        self.seen_sorts = set()

    def _prepare_constraints(self):
        self.model = MixedIntegerLinearProgram(solver=self.solver)
        self.var = self.model.new_variable(real=True, nonnegative=True)
        self.xs = [self.var["x%d" % i] for i in range(self.n)]
        self.c = self.var["c"]

        for p in self.hi:
            self.model.add_constraint(inner(p, self.xs) >= self.c)

        cs = {}
        for q in self.lo:
            cs[q] = inner(q, self.xs) <= self.c - 1

        self.cs_per_lo = cs

    def generate_inequality(self):
        LP = self.model.__copy__()
        covered_lo = []

        lstq = list(self.lo)
        shuffle(lstq)

        itr = 5
        while tuple(lstq) in self.seen_sorts:
            shuffle(lstq)
            itr -= 1
            if itr == 0:
                raise EOFError("exhausted")
        self.seen_sorts.add(tuple(lstq))

        for i, q in enumerate(lstq):
            constr_id = LP.number_of_constraints()
            LP.add_constraint(self.cs_per_lo[q])

            try:
                LP.solve()
            except MIPSolverException:
                assert i != 0
                LP.remove_constraint(constr_id)
            else:
                # print(f"covering #{i}/{len(lstq)}: {q}")
                covered_lo.append(q)
        LP.solve()

        func = tuple(LP.get_values(x) for x in self.xs)
        for v in func:
            # dunno why this hold, the vars are real
            assert abs(v - round(v)) < 0.01, func
        func = tuple(int(0.5 + v) for v in func)

        value_good = min(inner(p, func) for p in self.hi)
        value_bad = max(inner(p, func) for p in covered_lo)
        assert value_bad < value_good

        if self.inverted:
            # x1a1 + x2a2 + x3a3 >= t
            # =>
            # x1(1-a1) + x2(1-a2) + x3(1-a3) >= t
            # -x1a1 -x2a2 -x3a3 >= t-sum(x)
            value = value_good - sum(func)
            sol = tuple(-x for x in func) + (-value,)
            ret_covered = [tuple(1 - a for a in q) for q in covered_lo]
        else:
            # x1a1 + x2a2 + x3a3 >= t
            sol = func + (-value_good,)
            ret_covered = covered_lo

        # print(
        #     f"inequality coveres {len(covered_lo)}/{len(self.hi)}:",
        #     f"{func} >= {value_good}"
        # )
        for q in covered_lo:
            self.stat_covered[q] += 1
            self.stat_maxsize[q] = max(self.stat_maxsize[q], len(covered_lo))
        return sol, ret_covered


class InequalitiesPool:
    def __init__(self, points_good, points_bad, type_good=None):
        """
        type_good:
            lower:
                - points_good define a lower set to be kept,
                - points_bad define an upper set to be removed.
            upper:
                - points_good define an upper set to be kept,
                - points_bad define a lower set to be removed.
            None:
                - unstructured

        """
        for p in points_bad:
            self.n = len(p)
            break
        self.points_good = set(map(tuple, points_good))
        self.points_bad = set(map(tuple, points_bad))
        self.pool = {}  # ineq: (source, list of covered bad points)

        assert type_good in ("lower", "upper", None)
        self.type_good = type_good

    def check_good(self, ineqs=None):
        if ineqs is None:
            ineqs = self.pool
        if self.type_good == "upper":
            for ineq in ineqs:
                assert min(ineq[:-1]) >= 0
                assert ineq[-1] <= 0
        elif self.type_good == "upper":
            for ineq in ineqs:
                assert max(ineq[:-1]) <= 0
                assert ineq[-1] >= 0
        for p in self.points_good:
            assert all(satisfy(p, ineq) for ineq in ineqs)

    def check_bad(self, ineqs=None):
        if ineqs is None:
            ineqs = self.pool
        for p in self.points_bad:
            assert any(not satisfy(p, ineq) for ineq in ineqs)

    def check(self, ineqs=None):
        self.check_good(ineqs)
        self.check_bad(ineqs)

    def log_stat(self, ineqs=None):
        if ineqs is None:
            ineqs = self.pool

        log.info("-----------------------------------------------")
        log.info(f"total: {len(ineqs)} ineqs")
        cnt = Counter()
        for ineq in ineqs:
            source, covered = self.pool[ineq]
            cnt[source] += 1

        for source in sorted(cnt):
            log.info(f"- {source:20}: {cnt[source]}")
        log.info("-----------------------------------------------")

    def log_algo(self, s):
        log.info("")
        log.info("===============================================")
        log.info(s)
        log.info("===============================================")

    def get_covered_by_ineq(self, ineq):
        return [p for p in self.points_bad if not satisfy(p, ineq)]

    def pool_update(self, inequalities, source):
        num_new = 0
        covered_stat = Counter()
        if isinstance(inequalities, dict):
            # ineq: list of covered
            for ineq, covered in inequalities.items():
                if ineq not in self.pool:
                    self.pool[ineq] = (
                        source,
                        set(covered),
                    )
                    num_new += 1
                    covered_stat[len(covered)] += 1

        else:
            # iterable of ineqs
            for ineq in inequalities:
                if ineq not in self.pool:
                    covered = self.get_covered_by_ineq(ineq)
                    self.pool[ineq] = (
                        source,
                        set(covered),
                    )
                    num_new += 1
                    covered_stat[len(covered)] += 1
        log.info(f"generated {num_new} ineqs from {source}")
        log.info(f"covered stat: {sorted(covered_stat.items(), reverse=True)}")
        return num_new

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

    def generate_random(
        self, num=1000, max_coef=100, take_best_ratio=None, take_best_num=None):
        """
        sign: -1 for lower set,
               1 for upper set
            None for random signs
        """
        assert take_best_ratio is None or take_best_num is None
        if take_best_ratio is None and take_best_num is None:
            take_best_ratio = 0.2

        self.log_algo(
            "InequalitiesPool.generate_random("
            f"num={num}, max_coef={max_coef}, "
            f"take_best_ratio={take_best_ratio} "
            f"take_best_num={take_best_num}"
            ")"
        )

        L = []
        log.info(
            f"generating {num} random inequalities"
            f"(type_good={self.type_good})"
        )
        for _ in tqdm(range(num)):
            if self.type_good is None:
                eq = [randint(-max_coef, max_coef) for i in range(self.n)]
            else:
                eq = [randint(0, max_coef) for i in range(self.n)]

            ev_good = [inner(p, eq) for p in self.points_good]
            vmin = min(ev_good)
            vmax = max(ev_good)

            covmin = [q for q in self.points_bad if inner(q, eq) < vmin]
            covmax = [q for q in self.points_bad if inner(q, eq) > vmax]

            # covering 1 bad point is not interesting
            if self.type_good is None:
                if len(covmin) >= max(2, len(covmax)):
                    ineq = tuple(eq + [-vmin])
                    L.append((ineq, covmin))
                elif len(covmax) >= max(2, len(covmin)):
                    ineq = tuple([-a for a in eq] + [vmax])
                    L.append((ineq, covmax))
            elif self.type_good == "upper":
                if len(covmin) >= 2:
                    ineq = tuple(eq + [-vmin])
                    L.append((ineq, covmin))
            elif self.type_good == "lower":
                if len(covmax) >= 2:
                    ineq = tuple([-a for a in eq] + [vmax])
                    L.append((ineq, covmax))

        L.sort(reverse=True, key=lambda item: len(item[1]))
        if take_best_ratio is not None:
            L = L[:int(take_best_ratio * len(L) + 1)]
        else:
            L = L[:take_best_num]
        return self.pool_update(dict(L), source="random")

    def generate_linsep(self, num, solver="GLPK"):
        assert self.type_good in ("lower", "upper")
        self.log_algo(f"InequalitiesPool.generate_linsep(num={num})")

        if self.type_good == "lower":
            gen = self.linsep = LinearSeparator(
                lo=self.points_good,
                hi=self.points_bad,
                inverted=True,
                solver=solver,
            )
        elif self.type_good == "upper":
            gen = self.linsep = LinearSeparator(
                lo=self.points_bad,
                hi=self.points_good,
                inverted=False,
                solver=solver,
            )

        source = "linsep:random"

        L = {}
        for i in tqdm(range(num)):
            try:
                ineq, covered = gen.generate_inequality()
            except EOFError:
                break
            L[ineq] = covered
        return self.pool_update(L, source=source)

    def choose_subset_milp(self, solver=None):
        """
        [SecITC:SasTod17]
        Choose subset optimally by optimizing MILP system.

        Lcov: dict {ineq: covered bad points}
        """
        self.log_algo(f"InequalitiesPool.choose_subset_milp(solver={solver})")
        log.info(f"{len(self.pool)} ineqs {len(self.points_bad)} bad points")

        L = list(self.pool)
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

    def choose_subset_greedy_once(self):
        Lstar = set()
        Lover = set(self.pool)
        B = {self.points_bad}
        while B:
            lst = [
                (len(B & self.pool[ineq][1]), ineq)
                for ineq in Lover
            ]
            max_remov, _ = max(lst)
            lst = [
                ineq for num_remov, ineq in lst
                if num_remov == max_remov
            ]
            ineq = choice(lst)
            B -= self.pool[ineq][1]
            Lstar.add(ineq)
            Lover.remove(ineq)
        return Lstar

    def choose_subset_greedy(self, iterations=10):
        """
        Algorithm 1 [AC:XZBL16]
        https://eprint.iacr.org/2016/857
        """
        self.log_algo(
            f"InequalitiesPool.choose_subset_greedy(iterations={iterations})"
        )
        log.info(f"{len(self.pool)} ineqs {len(self.points_bad)} bad points")

        self.check_good()  # to avoid surprises

        best = len(self.pool), set(self.pool)
        for itr in tqdm(range(iterations)):
            Lstar = self._choose_subset_greedy_iter()

            cur = len(Lstar), Lstar
            if cur < best:
                best = cur
                log.info(f"new best: {cur[0]} inequalities")

                self.check(Lstar)
        return best[1]


def upper_set(s, n):
    from divprop.subsets import DenseSet
    ds = DenseSet(n)
    for c in s:
        ds.set(Bin(c).int)
    ds.do_UpperSet()
    return set(Bin(v, n).tuple for v in ds)


def lower_set(s, n):
    from divprop.subsets import DenseSet
    ds = DenseSet(n)
    for c in s:
        ds.set(Bin(c).int)
    ds.do_LowerSet()
    return set(Bin(v, n).tuple for v in ds)
