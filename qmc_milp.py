import os, pickle
from itertools import combinations
from collections import Counter

from binteger import Bin

from divprop.inequalities.monopool import (
    InequalitiesPool, LPbasedOracle, LazySparseSystem,
    tuple_xor, IneqInfo, shift_ineq,
)
from divprop.system_learn import (
    SupportLearner, RandomMaxFeasible,
    UnknownFillSAT, UnknownFillMILP,
    SATVerifier, Verifier,
)

from divprop.inequalities.base import satisfy, inner
from divprop.subsets import DenseSet, QMC1
from divprop import logging
from divprop.tools import get_sbox, get_sbox_sizes

logging.setup(level="WARNING")
log = logging.getLogger()

# name = "present"
name = "aes"
sbox = get_sbox(name)
n, m = get_sbox_sizes(sbox)
assert n == m
del m

file = f"data/qmc_{name}"
try:
    ddt, Sau = pickle.load(open(file, "rb"))
    print("load ok")
except Exception as err:
    Sau = None

if Sau is None:
    ddt = DenseSet(2*n)
    for x in range(2**n):
        for dx in range(2**n):
            dy = sbox[x] ^ sbox[x ^ dx]
            ddt.set((dx << n) | dy)
    nddt = ddt.Complement()
    print("ddt compl", nddt)

    Sau = QMC1(nddt)
    ddt = ddt.to_Bins()
    nddt = nddt.to_Bins()
    with open(file, "wb") as f:
        pickle.dump((ddt, nddt, Sau), f)

Sa = {}
for a, u in Sau:
    Sa.setdefault(a, []).append(u)

cnt = Counter(map(len, Sa.values()))
for la, cnta in sorted(cnt.items()):
    print(la, cnta)


sysfile = f"data/cache/qmc_{name}_main"
mainpool = InequalitiesPool(
    points_good=ddt,
    points_bad=nddt,
    type_good="-",
)
mainpool.set_system(LazySparseSystem(sysfile=sysfile))

logging.setLevel("INFO")
mainpool.system.refresh(extremize=False)

if 0:
    ineqs = mainpool.choose_subset_milp(solver="gurobi")

    print("minimum:", len(ineqs))

    file = f"data/qmc_{name}" + ".ineqs.%d" % len(ineqs)
    with open(file, "w") as f:
        print(len(ineqs), file=f)
        for ineq in ineqs:
            print(*ineq, file=f)

    print("written to", file)

    quit()

logging.setLevel("WARNING")

itr = 0
nddt = DenseSet([v.int for v in nddt], 2*n)
for a, us in sorted(Sa.items(), key=lambda aus: len(aus[1]), reverse=True):
    itr += 1
    if len(us) <= 1:
        break

    # if a != 0b00001111:  # (9, 6, 7, 7, -3, -1, -2, -3, 0) : 22
    #     continue

    # d = DenseSet(2*n)
    # for q in mainpool.bad:
    #     if not satisfy(q, (9, 6, 7, 7, -3, -1, -2, -3, 0)):
    #         d.set(Bin(q).int)
    # d.do_MaxSet()
    # for v in d.to_Bins():
    #     print(v)

    d = DenseSet(2*n)
    for u in us:
        d.set(u)
    d.do_LowerSet()
    full = d.copy()
    d.do_Complement()
    d.do_MinSet()

    ss = DenseSet(list(map(int, ddt)), 2*n)
    ss.do_Not(a)
    d = ss.MinSet()
    ss.do_UpperSet()
    ss.do_Complement()
    full = ss.copy()
    ss.do_MaxSet()
    us = ss

    upper = {Bin(v^a, 2*n).tuple for v in d}
    lower = {Bin(v^a, 2*n).tuple for v in us}
    shift = Bin(a, 2*n).tuple
    pool = InequalitiesPool(
        points_good=upper,
        points_bad=lower,
        type_good="upper",
        shift=shift,
    )
    sysfile = f"data/cache/qmc_{name}_a{a:x}"
    pool.set_oracle(LPbasedOracle(solver="glpk"))
    pool.set_system(LazySparseSystem(sysfile=sysfile))

    print(
        f"#{itr}/{len(Sa)}:",
        "a =", Bin(a, 2*n).str, "us", len(us), "|", "bad", len(upper), "good", len(lower))
    print("upper", len(upper))#*[Bin(v).str for v in upper])
    print("lower", len(lower))#*[Bin(v).str for v in lower])
    print("lower full", len(full))
    # print(us)

    # for fset, sol in pool.system.solution.items():
    #     for q in pool.good:
    #         assert satisfy(q, sol.ineq)
    #     for i in fset:
    #         assert not satisfy(pool.i2bad[i], sol.ineq)

    # for fset, sol in pool.system.solution.items():
    #     for q in mainpool.good:
    #         assert satisfy(q, shift_ineq(sol.ineq, shift))

    try:
        sat = UnknownFillSAT(
            minimization=True,
            refresh_rate=1000,
            solver="lingeling",
        )
        sat.init(system=pool.system, oracle=pool.oracle)
        sat.learn(num=10**6)
    except EOFError:
        print("solved, best:", len(pool.system.feasible))
    else:
        print("ouch??")
        quit()
    # print()
    # print("all bad")
    # for v in lower:
    #     print(v)

    d = DenseSet(2*n)
    for fset in pool.system.feasible:
        # print("removable", fset)
        sol = pool.system.solution[fset]
        sol = IneqInfo(shift_ineq(sol.ineq, shift), source=sol.source)

        # for q in mainpool.good:
        #     assert satisfy(q, sol.ineq)

        qs = [pool.i2bad[i] for i in fset]
        d.clear()
        for q in qs:
            d.set(Bin(q).int)
        d.do_LowerSet()
        d.do_Not(a)

        mainfset = frozenset(mainpool.bad2i[q.tuple] for q in d.to_Bins())
        mainpool.system.add_feasible(mainfset, sol=sol)

        # d.do_MaxSet()
        # for v in d.to_Bins():
        #     print(v)
        # print()

    # ineqs = mainpool.choose_subset_milp(solver="gurobi")

    # if itr % 5000 == 0:
    #     logging.setLevel("INFO")
    #     mainpool.system.refresh()
    #     logging.setLevel("WARNING")

logging.setLevel("INFO")
mainpool.system.refresh(extremize=False)
mainpool.system.refresh()

if 1:
    ineqs = mainpool.choose_subset_milp(solver="gurobi")

    print("minimum:", len(ineqs))

    file = f"data/qmc_{name}" + ".ineqs.%d" % len(ineqs)
    with open(file, "w") as f:
        print(len(ineqs), file=f)
        for ineq in ineqs:
            print(*ineq, file=f)

    print("written to", file)

if len(ineqs) < 50:
    for ineq in ineqs:
        cnt = sum(1 for q in mainpool._bad_orig if not satisfy(q, ineq))
        print(ineq, ":", cnt)
