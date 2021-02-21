import sys
import logging
from random import randint, choice, sample, shuffle, seed, randrange
from collections import Counter

# sage/pure python compatibility
try:
    import sage.all
    from sage.numerical.mip import MixedIntegerLinearProgram
    from sage.numerical.mip import MIPSolverException
except ImportError:
    pass

from tqdm import tqdm

from binteger import Bin

from divprop.divcore import DenseDivCore


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
    Algorithm is implemented for removing 'lo' while keeping 'hi'.
    For the other way around, vectors are flipped and swapped and
    output inequalities are adapted.
    """
    def __init__(self, lo, hi, inverted=False, solver="GLPK"):
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

    def _prepare_constraints(self):
        self.model = MixedIntegerLinearProgram(solver=self.solver)
        self.var = self.model.new_variable(real=True, nonnegative=True)
        self.xs = [self.var["x%d" % i] for i in range(self.n)]
        cs = {}
        for q in self.lo:
            csq = set()
            for p in self.hi:
                eq = (inner(p, self.xs) - inner(q, self.xs)) >= 1  # > 0
                csq.add(eq)
            cs[q] = csq
        self.cs_per_lo = cs
        self.stat_covered = {q: 0 for q in self.lo}
        self.stat_maxsize = {q: 0 for q in self.lo}

    def generate_inequality(self, by_covered=False, by_maxsize=False):
        LP = self.model.__copy__()
        covered_lo = []

        lstq = list(self.lo)
        if by_covered:
            lstq.sort(key=self.stat_covered.__getitem__)
        elif by_maxsize:
            lstq.sort(key=self.stat_maxsize.__getitem__)
        else:
            shuffle(lstq)

        for i, q in enumerate(lstq):
            n_start = LP.number_of_constraints()
            for c in self.cs_per_lo[q]:
                LP.add_constraint(c)
            n_end = LP.number_of_constraints()

            try:
                LP.solve()
            except MIPSolverException:
                assert i != 0
                LP.remove_constraints(range(n_start, n_end))
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


def get_n(points):
    for p in points:
        return len(p)


def check(points_good, points_bad, L):
    for p in points_good:
        assert all(satisfy(p, eq) for eq in L)
    for p in points_bad:
        assert any(not satisfy(p, eq) for eq in L)


def maxes(itr):
    cur_mx = -1
    eqs = []
    for val, eq in itr:
        if val > cur_mx:
            eqs = [eq]
            cur_mx = val
        elif val == cur_mx:
            eqs.append(eq)
    return eqs


def choose_subset_greedy_iter(points_bad, L):
    Lstar = set()
    Lover = set(L)
    itr = 0
    B = points_bad
    while B:
        itr += 1
        # print("itr", itr, len(B), "Lstar", len(Lstar), "Lover", len(Lover))
        eqs = maxes(
            (sum(1 for pt in B if not satisfy(pt, eq)), eq)
            for eq in Lover
        )
        eq = choice(eqs)
        B = [p for p in B if satisfy(p, eq)]
        Lstar.add(eq)
        Lover.remove(eq)
    return Lstar


def choose_subset_greedy(points_good, points_bad, L, iterations=5, output_file=None):
    """
    Algorithm 1 [AC:XZBL16]
    https://eprint.iacr.org/2016/857
    """
    check(points_good, (), L)  # to avoid surprises
    best = len(L), set(L)
    for itr in tqdm(range(iterations)):
        Lstar = choose_subset_greedy_iter(points_bad, L)

        cur = len(Lstar), Lstar
        if cur < best:
            best = cur
            print(f"new best: {cur[0]} inequalities")

            check((), points_bad, Lstar)
    return best[1]


def choose_subset_milp(points_good, points_bad, Lcov, solver=None, output_file=None):
    """
    [SecITC:SasTod17]
    Lcov: dict {ineq: covered bad points}
    """

    L = list(Lcov)
    check(points_good, (), L)  # to avoid surprises
    eq2i = {eq: i for i, eq in enumerate(L)}

    model = MixedIntegerLinearProgram(maximization=False, solver=solver)
    var = model.new_variable(binary=True, nonnegative=True)
    n = len(L)

    # xi = take i-th inequality?
    take_eq = [var["take_eq%d" % i] for i in range(n)]

    by_bad = {q: [] for q in points_bad}
    for eq, (source, covered) in Lcov.items():
        i = take_eq[eq2i[eq]]
        for q in covered:
            by_bad[q].append(i)

    for q, lst in by_bad.items():
        assert lst
        model.add_constraint(sum(lst) >= 1)

    model.set_objective(sum(take_eq))
    print("solving model...", n, "variables", len(points_bad), "constraints")
    model.solve(log=(n >= 10000))

    Lstar = []
    for take, eq in zip(take_eq, L):
        if model.get_values(take):
            Lstar.append(eq)
    check(points_good, points_bad, Lstar)
    return Lstar


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
                        covered,
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
                        covered,
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
        from sage.all import Polyhedron
        p = Polyhedron(vertices=points_good)
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
        return self.pool_update(L, "sage.polyhedron")

    def generate_random(self, num=1000, max_coef=100, take_best_ratio=None, take_best_num=None):
        """
        sign: -1 for lower set,
               1 for upper set
            None for random signs
        """
        assert take_best_ratio is None or take_best_num is None
        if take_best_ratio is None and take_best_num is None:
            take_best_ratio = 0.2

        L = []
        sign = None
        if self.type_good == "lower":
            sign = -1
        elif self.type_good == "upper":
            sign = 1

        print(f"generating {num} random inequalities with sign {sign}")
        for _ in tqdm(range(num)):
            if sign is None:
                eq = [randint(-max_coef, max_coef) for i in range(n)]
            else:
                eq = [sign * randint(0, max_coef) for i in range(n)]

            ev_good = [inner(p, eq) for p in points_good]
            vmin = min(ev_good)
            vmax = max(ev_good)

            covmin = [q for q in points_bad if inner(q, eq) < vmin]
            covmax = [q for q in points_bad if inner(q, eq) > vmax]

            # covering 1 bad point is not interesting
            if len(covmin) >= max(2, len(covmax)):
                ineq = tuple(eq + [-vmin])
                L.append((ineq, covmin))
            elif len(covmax) >= max(2, len(covmin)):
                ineq = tuple([-a for a in eq] + [vmax])
                L.append((ineq, covmax))
        L.sort(reverse=True, key=lambda item: len(item[1]))
        if take_best_ratio is not None:
            L = L[:int(take_best_ratio * len(L))]
        else:
            L = L[:take_best_num]
        return self.pool_update(dict(L), source="random")

    def generate_linsep(self, num, by_maxsize=False, by_covered=False):
        assert self.type_good in ("lower", "upper")

        if type_good == "lower":
            gen = self.linsep = LinearSeparator(
                lo=self.points_good,
                hi=self.points_bad,
                inverted=True,
            )
        elif type_good == "upper":
            gen = self.linsep = LinearSeparator(
                lo=self.points_bad,
                hi=self.points_good,
                inverted=False,
            )

        assert not (by_maxsize and by_covered)
        source = "linsep:"
        if by_maxsize:
            source += "by_maxsize"
        elif by_covered:
            source += "by_covered"
        else:
            source += "random"

        L = {}
        for i in tqdm(range(num)):
            ineq, covered = gen.generate_inequality(
                by_maxsize=by_maxsize,
                by_covered=by_covered,
            )
            L[ineq] = covered
        return self.pool_update(L, source=source)

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


SEED = randrange(10**10)
seed(SEED)

print("SEED =", SEED)

name = sys.argv[1].lower()

num_random = {
    4: 2_000_000,
    5: 100_000,
    8: 1_000,
    9: 123,
}
num_random = {
    4: 100_000,
    5: 25_000,
    8: 1_000,
    9: 123,
}

from sage.crypto.sboxes import sboxes

def get_sbox(name):
    for k, v in sboxes.items():
        if k.lower() == name.lower():
            return v
    raise KeyError()

sbox = get_sbox(name)
n = int(sbox.input_size())
m = int(sbox.output_size())
sbox = tuple(map(int, sbox))
num_random = num_random[(n+m)//2]

def get_UB(dc):
    ret = dc.data.copy()
    # mindppt
    ret.do_UpperSet(dc.mask_u)
    ret.do_MinSet(dc.mask_v)
    # get upper
    ret.do_LowerSet()
    ret.do_Complement()
    ret.do_MinSet()
    return ret

def get_UB_old(dc):
    ret = dc.data.copy()
    ret.do_UpperSet_Up1(True, dc.mask_v)  # is_minset=true
    ret.do_MinSet()
    return ret

def prset(name, s):
    print(name)
    for a in s:
        print("   ", Bin(a, n+m))

print("name", name)

dc = DenseDivCore.from_sbox(sbox, n, m)
mid = dc.MinDPPT().Not(dc.mask_u)

dclo = mid.MinSet()
dcup = mid.MaxSet()

lo = dc.LB().LowerSet()
up = get_UB(dc).UpperSet()
assert (lo & mid).is_empty()
assert (up & mid).is_empty()
# assert (lo & up).is_empty()
print("lo & up complem", (lo & up))
assert (lo | mid | up).is_full()

lo = dc.LB().LowerSet()
up = get_UB_old(dc).UpperSet()
assert (lo & mid).is_empty()
assert (up & mid).is_empty()
print("lo & up    old", (lo & up))
assert (lo | mid | up).is_full()

lb = dc.LB()
ub_compl = get_UB(dc)
ub_old = get_UB_old(dc)

print("dc    ", dc)
print("lb    ", lb)
print("mid   ", mid)
print("ub_old", get_UB_old(dc))
print("ub_com", get_UB(dc))
# assert get_UB(dc) == get_UB_old(dc)
# assert get_UB(dc) == get_UB_old(dc) == ub

test = mid.UpperSet()
test2 = lb.LowerSet()
assert (test & test2).is_empty(), "mid + lb"
assert (test | test2).is_full(), "mid + lb"

test = mid.LowerSet()
test2 = ub_compl.UpperSet()
assert (test & test2).is_empty(), "mid + ub"
assert (test | test2).is_full(), "mid + ub"

test = mid.LowerSet()
test2 = ub_old.UpperSet()
assert (test & test2).is_empty(), "mid + ub"
# assert (test | test2).is_full(), "mid + ub"


def topts(st):
    return [Bin(v, n+m).tuple for v in st]


def section(name):
    print()
    print("=" * 50)
    print("Section:", name)
    print("=" * 50)


for typ in "lb", "ubc", "ubo":
    log.info("\n\n\n")
    log.info(f"Starting type {typ}")
    extremize = True
    if typ == "lb":
        points_good = dclo.UpperSet()
        points_bad = points_good.Complement()

        if extremize:
            points_good = points_good.MinSet()
            points_bad = points_bad.MaxSet()
        type_good = "upper"

    elif typ == "ubo":
        selected_pts_ub = ub_old
        points_good = dcup.LowerSet()
        points_bad = points_good.Complement() - lb.LowerSet()

        if extremize:
            points_good = points_good.MaxSet()
            points_bad = points_bad.MinSet()
        type_good = "lower"
    elif typ == "ubc":
        selected_pts_ub = ub_compl
        points_good = dcup.LowerSet()
        points_bad = points_good.Complement()

        if extremize:
            points_good = points_good.MaxSet()
            points_bad = points_bad.MinSet()
        type_good = "lower"
    else:
        assert 0

    print("points_good", points_good)
    print("points_bad", points_bad)
    print("inter", points_bad & points_good)

    points_good = set(topts(points_good))
    points_bad = set(topts(points_bad))

    pool = InequalitiesPool(
        points_good=points_good,
        points_bad=points_bad,
        type_good=type_good,
    )

    if 1:
        section("Linear Separator")

        pool.generate_linsep(num=250, by_covered=True)
        pool.generate_linsep(num=50, by_maxsize=True)

        check(points_good, points_bad, pool.pool)

    if n + m < 16 and 1:
        section("Sage Polyhedron")

        pool.generate_from_polyhedron()

        check(points_good, points_bad, pool.pool)

    if 1:
        section("Random")
        pool.generate_random(
            num=10000,
            max_coef=100,
            take_best_num=2500,
        )

        check(points_good, points_bad, pool.pool)

    pool.log_stat()

    if 0:
        section("Choose Subset Greedy")
        Lstar = choose_subset_greedy(
            points_good,
            points_bad,
            pool.pool,
            iterations=10,
        )
    if 1:
        section("Choose Subset OPT")
        Lstar = choose_subset_milp(
            points_good,
            points_bad,
            pool.pool,
        )

    pool.log_stat(Lstar)

    if 1:
        filename = f"results/{name}_sbox.{typ}.%d_ineq" % len(Lstar)
        with open(filename, "w") as f:
            print(len(Lstar), file=f)
            for eq in Lstar:
                print(*eq, file=f)
