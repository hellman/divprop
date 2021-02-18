import ast
import subprocess
from itertools import combinations, product
from random import randint, randrange, choice

from tqdm import tqdm

from binteger import Bin

from divprop.divcore import DenseDivCore


def satisfy(pt, eq):
    assert len(pt) + 1 == len(eq)
    res = sum(a * b for a, b in zip(pt, eq)) + eq[-1]
    return res >= 0


def sage_gen_inequalities(points_good, points_bad):
    code = f"""
from sage.all import Polyhedron
p = Polyhedron(vertices={points_good})
print(sorted(map(tuple, p.inequalities())))
print(sorted(map(tuple, p.equations())))
quit()
    """.lstrip()
    out = subprocess.check_output(
        ["sage", "-python"], input=code.encode()
    ).splitlines()
    L = ast.literal_eval(out[0].strip().decode())
    E = ast.literal_eval(out[1].strip().decode())
    # https://twitter.com/SiweiSun2/status/1327973545666891777
    for eq in E:
        L.append(eq)
        L.append(tuple(-v for v in eq))
    # xs + ys + coef >= 0
    L = [tuple(eq[1:] + eq[:1]) for eq in L]
    return L


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


def optimize_inequalities_greedy_iter(points_good, points_bad, L):
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


def optimize_inequalities_greedy(points_good, points_bad, L, iterations=5, output_file=None):
    best = len(L), L
    check(points_good, points_bad, L)
    for itr in tqdm(range(iterations)):
        Lstar = optimize_inequalities_greedy_iter(points_good, points_bad, L)

        cur = len(Lstar), Lstar
        if cur < best:
            best = cur
            print(f"new best: {cur[0]} equations")

            check(points_good, points_bad, cur[1])
            if output_file:
                with open(output_file % len(Lstar), "w") as f:
                    for eq in Lstar:
                        print(*eq, file=f)

    return best[1]


if 0:
    name = "rectangle"
    n = m = 4
    sbox = 6, 5, 12, 10, 1, 14, 7, 9, 11, 0, 3, 13, 8, 15, 4, 2
    num_random = 2_000_000

if 0:
    name = "ascon"
    n = m = 5
    sbox = 4, 11, 31, 20, 26, 21, 9, 2, 27, 5, 8, 18, 29, 3, 6, 28, 30, 19, 7, 14, 0, 13, 17, 24, 16, 12, 1, 25, 22, 10, 15, 23
    num_random = 100_000

if 1:
    name = "present"
    n = m = 4
    sbox = 0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2
    num_random = 2_000_000

print("name", name)

dc = DenseDivCore.from_sbox(sbox, n, m)
lb = dc.LB()
lb.do_LowerSet()
lb.do_Complement()
ub = dc.UB()
ub.do_UpperSet()
ub.do_Complement()

for typ in "ub", "lb":
    print("GO TYP", typ)

    target = lb if typ == "lb" else ub
    points_good = {Bin(p, n+m).tuple for p in target}
    points_bad = {Bin(v, n+m).tuple for v in range(2**(n+m))} - points_good
    print(len(points_good), "points_good")
    print(len(points_bad), "points_bad")

    L = sage_gen_inequalities(points_good, points_bad)
    print(len(L), "equations from sage")
    check(points_good, points_bad, L)

    # There's a ~tradeoff between
    # #ineq to generate,
    # best percentage of inequalities to keep.
    # The main algorithm depends on the #ineq linearly (times #iter).
    # So removing bad ineqs at prepro does not improve performance a lot.
    L2 = gen_random_inequalities(
        points_good,
        points_bad,
        num=num_random,
        max_coef=1_000,
        sign=-1 if typ == "lb" else 1,
        take_best=0.4,  # percentage of inequalities kept
    )
    print(len(L2), "equations random")
    check(points_good, points_bad, L+L2)

    optimize_inequalities_greedy(
        points_good,
        points_bad,
        L + L2,
        iterations=10,
        output_file=f"results/{name}_sbox.{typ}.%d_ineq"
    )
