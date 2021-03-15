from divprop.inequalities.monopool import (
    InequalitiesPool, LPbasedOracle,
)
from itertools import combinations
from divprop.learn import CliqueMountainHills

from divprop.inequalities.base import satisfy, inner
from divprop.subsets import DenseSet
from divprop import logging
from binteger import Bin


logging.setup(level="DEBUG")
log = logging.getLogger()

fileprefix = "/work/division/workspace/data/sbox_skinny_4/divcore.lb"

fileprefix = "/work/division/workspace/data/sbox_present/divcore.lb"
fileprefix = "/work/division/workspace/data/sbox_present/divcore.ubc"
fileprefix = "/work/division/workspace/data/sbox_present/divcore.full"

fileprefix = "/work/division/workspace/data/sbox_present/divcore.lb"
fileprefix = "/work/division/workspace/data/sbox_present/divcore.ubc"
fileprefix = "/work/division/workspace/data/sbox_present/divcore.full"

fileprefix = "/work/division/workspace/data/sbox_present/ddt"

# fileprefix = "/work/division/workspace/data/sbox_aes/divcore.lb"
# fileprefix = "/work/division/workspace/data/sbox_aes/divcore.ubc"

pool = InequalitiesPool.from_DenseSet_files(fileprefix)

# pool.gen_hats()
# for i in range(1000):
#     pool.gen_random_inequality()

pool.set_oracle(LPbasedOracle(solver="glpk"))

CliqueMountainHills(
    base_level=3,
    max_mountains=0,
    n_random=2500,
    max_exclusion_size=10,
    solver="gurobi",
).learn_system(
    system=pool.system,
    oracle=pool.oracle,
)

print("calls", pool.oracle.n_calls)

pool.system.log_info()

print("minimizing...")

ineqs = pool.choose_subset_milp(solver="gurobi")

print("minimum:", len(ineqs))

file = fileprefix + ".ineqs.%d" % len(ineqs)
with open(file, "w") as f:
    print(len(ineqs), file=f)
    for ineq in ineqs:
        print(*ineq, file=f)

print("written to", file)

for q in pool._good_orig:
    assert all(satisfy(q.tuple, ineq) for ineq in ineqs)

for q in pool._bad_orig:
    assert any(not satisfy(q.tuple, ineq) for ineq in ineqs)

print("checks ok!")

if pool.N <= 18:
    for l in range(pool.N+1):
        for t in combinations(range(pool.N), l):
            t = pool.system.encode_fset(t)
            test1 = pool.system.is_already_feasible(t)
            test2 = pool.system.is_already_infeasible(t)
            # print(Bin(t, pool.N), Bin(t, pool.N).support(), test1, test2)
            assert test1 ^ test2

for fset in pool.system.infeasible:
    assert not pool.oracle.query(Bin(fset, pool.N))

for v in pool.system.feasible:
    ineq = pool.system.solution[v].ineq
    badinds = Bin(v, pool.N).support()
    # print("v", Bin(v, pool.N), "badinds", badinds, "ineq", ineq)
    res = pool.oracle.query(Bin(v, pool.N))
    # print("query", res)
    assert all(satisfy(q, ineq) for q in pool.good)
    assert all(not satisfy(pool.i2bad[i], ineq) for i in badinds)

# fset [3, 8, 10, 11, 13, 14] ineq (5, 2, 1, 7, 3, 5, 4, 8, -11)
# fset [8, 9, 11, 14, 15, 16] ineq (2, 1, 4, 3, 3, 5, 7, 8, -10)
# fset [0, 2, 3, 4, 5, 6, 8] ineq (4, 3, 1, 4, 2, 0, 3, 2, -6)
# fset [1, 2, 3, 4, 7, 8, 14] ineq (7, 5, 2, 8, 1, 3, 5, 4, -11)
# fset [6, 7, 8, 9, 11] ineq (3, 0, 3, 3, 2, 2, 5, 4, -7)

# fset [3, 5, 6, 8, 9, 11, 17] ineq (5, 3, 2, 6, 6, 1, 8, 7, -12)
# fset [0, 1, 2, 3, 7, 8, 12] ineq (8, 6, 5, 10, 1, 4, 3, 5, -13)
