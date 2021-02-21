# sage/pure python compatibility
try:
    import sage.all
    from sage.numerical.mip import MixedIntegerLinearProgram
    from sage.numerical.mip import MIPSolverException
except ImportError:
    pass

from random import randint, choice, sample, shuffle, seed, randrange

from tqdm import tqdm

from binteger import Bin

from divprop.divcore import DenseDivCore

SEED = randrange(10**10)
seed(SEED)


def satisfy(pt, eq):
    assert len(pt) + 1 == len(eq)
    res = sum(a * b for a, b in zip(pt, eq)) + eq[-1]
    return res >= 0


def sage_gen_inequalities(points_good, points_bad):
    from sage.all import Polyhedron
    p = Polyhedron(vertices=points_good)
    L = sorted(map(tuple, p.inequalities()))
    E = sorted(map(tuple, p.equations()))
    # https://twitter.com/SiweiSun2/status/1327973545666891777
    for eq in E:
        L.append(eq)
        L.append(tuple(-v for v in eq))
    # xs + ys + coef >= 0
    L = [tuple(eq[1:] + eq[:1]) for eq in L]
    return L


def inner(a, b):
    return sum(aa * bb for aa, bb in zip(a, b))


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


def gen_random_inequalities(points_good, points_bad, num=1000, max_coef=4, sign=None, take_best=0.2):
    """
    sign: -1 for lower set,
           1 for upper set
        None for random signs
    """
    n = get_n(points_good)
    evaleq = lambda pt, eq: sum(a * b for a, b in zip(pt, eq))
    L = []
    min_remove = None
    n_samples = 0
    best_remove = []
    print(f"generating {num} random inequalities")
    with tqdm(total=num) as tq:
        while len(L) < num:
            if sign is None:
                eq = [randint(-max_coef, max_coef) for i in range(n)]
            else:
                eq = [sign * randint(0, max_coef) for i in range(n)]

            ev_good = [evaleq(p, eq) for p in points_good]
            ev_bad = [evaleq(p, eq) for p in points_bad]

            vmin = min(ev_good)
            vmax = max(ev_good)

            nbadmin = sum(1 for v in ev_bad if v < vmin)
            nbadmax = sum(1 for v in ev_bad if v > vmax)
            if min_remove is not None:
                if nbadmin >= min_remove:
                    L.append(tuple(eq + [-vmin]))
                    tq.update(1)
                    assert all(satisfy(p, L[-1]) for p in points_good)
                elif nbadmax >= min_remove:
                    L.append(tuple([-a for a in eq] + [vmax]))
                    tq.update(1)
                    assert all(satisfy(p, L[-1]) for p in points_good)
            else:
                n_samples += 1
                best_remove.append(max(nbadmin, nbadmax))
                if n_samples >= 100:
                    min_remove = sorted(best_remove, reverse=True)[int(len(best_remove) * take_best)]
                    print(f"determined min_remove={min_remove} in {n_samples} samples")
    return L


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
    best = len(L), set(L)
    check(points_good, (), L)
    for itr in tqdm(range(iterations)):
        Lstar = choose_subset_greedy_iter(points_bad, L)

        cur = len(Lstar), Lstar
        if cur < best:
            best = cur
            print(f"new best: {cur[0]} inequalities")

            check((), points_bad, Lstar)
            if output_file:
                with open(output_file % len(Lstar), "w") as f:
                    for eq in Lstar:
                        print(*eq, file=f)

    return best[1]


def choose_subset_milp(points_good, points_bad, L, solver=None, output_file=None):
    n = get_n(points_bad)
    model = MixedIntegerLinearProgram(solver=solver)
    var = model.new_variable(real=True, nonnegative=True)
    xs = [var["x%d" % i] for i in range(n)]


import sys

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
    5: 100_000,
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


for typ in "lb", "ubc", "ubo":
    if typ == "lb":
        gen = LinearSeparator(topts(lb), topts(dclo))
        points_good = dclo.UpperSet()
        points_bad = points_good.Complement()
    elif typ == "ubo":
        selected_pts_ub = ub_old
        gen = LinearSeparator(topts(dcup), topts(selected_pts_ub), inverted=True)
        points_good = dcup.LowerSet()
        points_bad = points_good.Complement() - lb.LowerSet()
    elif typ == "ubc":
        selected_pts_ub = ub_compl
        gen = LinearSeparator(topts(dcup), topts(selected_pts_ub), inverted=True)
        points_good = dcup.LowerSet()
        points_bad = points_good.Complement()
    else:
        assert 0

    print("points_good", points_good)
    print("points_bad", points_bad)
    print("inter", points_bad & points_good)

    points_good = set(topts(points_good))
    points_bad = set(topts(points_bad))

    L = []

    if 1:
        todo = 250
        print(f"generating {todo} separator inequalities")
        cur = []
        allcov = set()
        for i in tqdm(range(todo)):
            if i > todo * 0.9:
                eq, covered = gen.generate_inequality(by_maxsize=True)
            elif i > todo * 0.8:
                eq, covered = gen.generate_inequality(by_covered=True)
            else:
                eq, covered = gen.generate_inequality()
            for q in covered:
                assert not satisfy(q, eq)
            cur.append(eq)
            allcov |= set(covered)
            # print(eq, ":", len(set(covered)), "->", len(allcov), "/", len(selected_pts_ub))

        todo = list(gen.stat_covered.values()).count(0)
        print(f"generating {todo} missing separator inequalities")
        for i in tqdm(range(todo)):
            eq, covered = gen.generate_inequality(by_covered=True)
            cur.append(eq)
            allcov |= set(covered)
            # print(eq, ":", len(set(covered)), "->", len(allcov), "/", len(selected_pts_ub))
        print(len(cur), "inequalities from separator")
        L += cur

    check(points_good, points_bad, L)

    if 0:
        cur = sage_gen_inequalities(points_good, points_bad)
        print(len(cur), "inequalities from sage")
        check(points_good, points_bad, cur)
        L += cur

    # There's a ~tradeoff between
    # #ineq to generate,
    # best percentage of inequalities to keep.
    # The main algorithm depends on the #ineq linearly (times #iter).
    # So removing bad ineqs at prepro does not improve performance a lot.
    if 0:
        cur = gen_random_inequalities(
            points_good,
            points_bad,
            num=num_random,
            max_coef=100,
            sign=-1 if typ == "lb" else 1,
            take_best=0.4,  # percentage of inequalities kept
        )
        L += cur
        print(len(cur), "inequalities random")
        check(points_good, [], L)

    if 0:
        cur = []
        for p in edge_bad:
            # print("p bad  ", p)
            if typ == "lb":
                # sum(xi with xi=0) >= 1
                r = 1
                cur.append(tuple(int(xi == 0) for xi in p) + (-r,))
            else:
                # sum(1-xi with xi=1) >= 1
                w = sum(p)
                r = 1 - w
                cur.append(tuple(-int(xi == 1) for xi in p) + (-r,))
            # print("eq     ", cur[-1])
            # for p in points_good:
            #     if not satisfy(p, cur[-1]):
            #         print("p good ", p)
            #         break
            check(points_good, [], cur[-1:])
        L += cur
        print(len(cur), "inequalities basic")
        check(points_good, points_bad, L)

        for i in range(1000):
            l = randint(2, 3)
            eqs = sample(L, l)
            cs = [randint(1, 1) for _ in range(l)]
            L.append(tuple(
                sum(c * coef for c, coef in zip(cs, coefs))
                for coefs in zip(*eqs)
            ))

    check(points_good, points_bad, L)

    print("Total", len(L), "ineqs")
    choose_subset_greedy(
        points_good,
        points_bad,
        L,
        iterations=10,
        output_file=f"results/{name}_sbox.{typ}.%d_ineq"
    )
