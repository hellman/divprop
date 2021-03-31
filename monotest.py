import os
from itertools import combinations

from binteger import Bin

from divprop.inequalities.monopool import (
    InequalitiesPool, LPbasedOracle, LazySparseSystem,
)
from divprop.system_learn import (
    SupportLearner, RandomMaxFeasible,
    UnknownFillSAT, UnknownFillMILP,
    SATVerifier, Verifier,
)

from divprop.inequalities.base import satisfy, inner
from divprop.subsets import DenseSet
from divprop import logging

logging.setup(level="DEBUG")
log = logging.getLogger()

fileprefix = "/work/division/workspace/data/sbox_skinny_4/divcore.lb"

fileprefix = "/work/division/workspace/data/sbox_present/divcore.lb"
fileprefix = "/work/division/workspace/data/sbox_present/divcore.ubc"
fileprefix = "/work/division/workspace/data/sbox_present/divcore.full"

fileprefix = "/work/division/workspace/data/sbox_present/divcore.lb"
fileprefix = "/work/division/workspace/data/sbox_present/divcore.ubc"
fileprefix = "/work/division/workspace/data/sbox_present/divcore.full"

# fileprefix = "/work/division/workspace/data/sbox_present/ddt"
fileprefix = "/work/division/workspace/data/sbox_presentmod/ptt"
# fileprefix = "/work/division/workspace/data/sbox_present/ptt"

fileprefix = "/work/division/workspace/data/sbox_aes/divcore.lb"
# fileprefix = "/work/division/workspace/data/sbox_aes/divcore.ubc"
#fileprefix = "/work/division/workspace/data/sbox_aes/divcore.ubo"

#fileprefix = "/work/division/workspace/data/sbox_present/ddt"
# fileprefix = "/work/division/workspace/data/sbox_klein/ddt"
# fileprefix = "/work/division/workspace/data/sbox_klein/ddt"
# fileprefix = "/work/division/workspace/data/sbox_twine/ddt"
# # fileprefix = "/work/division/workspace/data/sbox_prince/ddt"
# fileprefix = "/work/division/workspace/data/sbox_piccolo/ddt"
# fileprefix = "/work/division/workspace/data/sbox_mibs/ddt"
# fileprefix = "/work/division/workspace/data/sbox_lilliput_ae/ddt"
fileprefix = "/work/division/workspace/data/sbox_ascon/ddt"

sysfile = fileprefix + ".system"

if 0:
    try:
        os.unlink(sysfile)
    except Exception as err:
        pass

pool = InequalitiesPool.from_DenseSet_files(fileprefix)
pool.set_oracle(LPbasedOracle(solver="sage/glpk"))
pool.set_system(LazySparseSystem(sysfile=sysfile))

# SL = SupportLearner(level=3)
# SL.init(system=pool.system, oracle=pool.oracle)
# SL.learn()

if 0:
    RandMax = RandomMaxFeasible(base_level=2, extremize_rate=100)
    RandMax.init(system=pool.system, oracle=pool.oracle)
    RandMax.learn(num=5_000)

if 0:
    Comp = UnknownFillMILP(extremize_rate=25, solver="sage/gurobi", batch_size=10)
    Comp.init(system=pool.system, oracle=pool.oracle)
    Comp.learn(level=4, num=50)

if 0:
    while True:
        try:
            Comp = UnknownFillMILP(extremize_rate=1, solver="sage/gurobi", batch_size=1)
            Comp.init(system=pool.system, oracle=pool.oracle)
            Comp.learn(maximization=True, num=1)

            Comp = UnknownFillMILP(extremize_rate=1, solver="sage/gurobi", batch_size=1)
            Comp.init(system=pool.system, oracle=pool.oracle)
            Comp.learn(maximization=False, num=1)
        except EOFError:
            break

if 1:
    try:
        sat = UnknownFillSAT(
            minimization=True,
            save_rate=100,
            solver="cadical",
        )
        sat.init(system=pool.system, oracle=pool.oracle)
        sat.learn(num=10**9)
    except EOFError:
        print("solved?!")
        pass

if 1:
    # Ver = Verifier(solver="gurobi")
    Ver = SATVerifier(solver="cadical")
    Ver.init(system=pool.system, oracle=pool.oracle)
    Ver.learn(clean=False)

pool.system.log_info()

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

print("minimizing...")

ineqs = pool.choose_subset_milp(solver="gurobi")

print("minimum:", len(ineqs))

file = fileprefix + ".ineqs.%d" % len(ineqs)
with open(file, "w") as f:
    print(len(ineqs), file=f)
    for ineq in ineqs:
        print(*ineq, file=f)

print("written to", file)

if len(ineqs) < 50:
    for ineq in ineqs:
        cnt = sum(1 for q in pool._bad_orig if not satisfy(q, ineq))
        print(ineq, ":", cnt)
