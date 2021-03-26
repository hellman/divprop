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

name = "present"
name = "aes"
sbox = get_sbox(name)
n, m = get_sbox_sizes(sbox)
assert n == m
del m

file = f"data/qmc_{name}"

ddt = DenseSet(2*n)
for x in range(2**n):
    for dx in range(2**n):
        dy = sbox[x] ^ sbox[x ^ dx]
        ddt.set((dx << n) | dy)
nddt = ddt.Complement()
print("ddt compl", nddt)

sysfile = f"data/cache/qmc_{name}_main"
sysfile = None
mainpool = InequalitiesPool(
    points_good=ddt.to_Bins(),
    points_bad=nddt.to_Bins(),
    type_good="-",
)
mainpool.set_system(LazySparseSystem(sysfile=sysfile))

logging.setLevel("INFO")
mainpool.system.refresh(extremize=False)

if 0:
    ineqs = mainpool.choose_subset_greedy(iter=1)

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
for a in range(2**(2*n)):
    itr += 1
    if a in ddt:
        continue

    d = ddt.Not(a)
    good = d.MinSet()
    d.do_UpperSet()
    d.do_Complement()
    fullsz = len(d)
    # d.do_MaxSet()
    bad = d

    shift = Bin(a, 2*n)
    pool = InequalitiesPool(
        points_good=good.to_Bins(),
        points_bad=bad.to_Bins(),
        type_good="upper",
    )
    sysfile = f"data/cache/qmc_{name}_a{a:x}"
    sysfile = None

    pool.set_oracle(LPbasedOracle(solver="glpk"))
    pool.set_system(LazySparseSystem(sysfile=sysfile))

    print(
        f"#{itr}/{4**n}:", "a =", Bin(a, 2*n).str, "|",
        "up", len(good), "lo", len(bad), "full", fullsz
    )

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
            solver="cadical",
        )
        sat.init(system=pool.system, oracle=pool.oracle)
        sat.learn(num=10**6)
    except EOFError:
        print("solved, best:", len(pool.system.feasible))
    else:
        print("ouch??")
        quit()

    d = DenseSet(2*n)
    for fset in pool.system.feasible:
        qs = [pool.i2bad[i] for i in fset]
        d.clear()
        for q in qs:
            d.set(Bin(q).int)
        d.do_LowerSet()
        d.do_Not(a)

        sol = pool.system.solution[fset]
        sol = sol._replace(ineq=shift_ineq(sol.ineq, shift))

        mainfset = frozenset(mainpool.bad2i[q.tuple] for q in d.to_Bins())
        mainpool.system.add_feasible(mainfset, sol=sol)

    if itr % 1000 == 0:
        logging.setLevel("INFO")
        # ineqs = mainpool.choose_subset_greedy(1)
        # log.info(f"cur best greedy: {len(ineqs)}")
        mainpool.system.refresh(extremize=False)
        logging.setLevel("WARNING")

logging.setLevel("INFO")
mainpool.system.refresh(extremize=False)
mainpool.system.refresh()

if 1:
    ineqs = mainpool.choose_subset_greedy(1)

    print("greedy:", len(ineqs))

    file = f"data/qmc_{name}" + ".ineqs.%d" % len(ineqs)
    with open(file, "w") as f:
        print(len(ineqs), file=f)
        for ineq in ineqs:
            print(*ineq, file=f)

    print("written to", file)

    # ==================================================

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
