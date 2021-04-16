import os

from binteger import Bin

from divprop.inequalities import (
    InequalitiesPool, LPbasedOracle, satisfy,
)
from divprop.learn import (
    GainanovSAT, RandomLower, LevelLearn,
    # SATVerifier, Verifier,
)

from divprop.subsets import DenseSet
from divprop import logging

logging.setup(level="DEBUG")
log = logging.getLogger()

# fileprefix = "/work/division/workspace/data/sbox_skinny_4/divcore.lb"

# fileprefix = "/work/division/workspace/data/sbox_present/divcore.lb"
# fileprefix = "/work/division/workspace/data/sbox_present/divcore.ubc"
# fileprefix = "/work/division/workspace/data/sbox_present/divcore.full"

# fileprefix = "/work/division/workspace/data/sbox_present/divcore.lb"
# fileprefix = "/work/division/workspace/data/sbox_present/divcore.ubc"
# fileprefix = "/work/division/workspace/data/sbox_present/divcore.full"

# fileprefix = "/work/division/workspace/data/sbox_present/ddt"
# fileprefix = "/work/division/workspace/data/sbox_presentmod/ptt"
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
# fileprefix = "/work/division/workspace/data/sbox_ascon/ddt"

sysfile = fileprefix + ".system"

if 0:
    try:
        os.unlink(sysfile)
    except Exception as err:
        pass

pool = InequalitiesPool.from_DenseSet_files(
    fileprefix=fileprefix,
    oracle=LPbasedOracle(solver="sage/glpk"),
    # sysfile=None,
)

# SL = SupportLearner(level=3)
# SL.init(system=pool.system, oracle=pool.oracle)
# SL.learn()

if 1:
    lev = LevelLearn(levels_lower=3)  # 0, 1, 2
    lev.init(system=pool.system)
    print(lev.__dict__)
    lev.learn()

if 0:
    low = RandomLower(max_repeat_rate=0.9)
    low.init(system=pool.system)
    low.learn()

if 1:
    sat = GainanovSAT(
        sense="min",
        save_rate=100,
        solver="cadical",
    )
    sat.init(system=pool.system)
    ret = sat.learn()
    print("GainanovSAT returned", ret)

if 0:
    Ver = SATVerifier(solver="cadical")
    Ver.init(system=pool.system)
    Ver.learn()

pool.write_subset_milp("/tmp/test.lp")

if 1:
    print("test 1")
    for vec in pool.system.iter_upper():
        assert not pool.oracle(vec)[0]

    print("test 2")
    for vec in pool.system.iter_lower():
        ineq = pool.system.meta[vec]
        assert pool.oracle(vec)[0]
        assert all(satisfy(q, ineq) for q in pool.good)
        assert all(not satisfy(pool.i2bad[i], ineq) for i in vec)

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
