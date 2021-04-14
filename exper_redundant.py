from random import randint, seed, random
from divprop.subsets import DenseSet
from collections import defaultdict
from functools import reduce

from binteger import Bin

from divprop.inequalities.monopool import (
    InequalitiesPool, LPbasedOracle, LazySparseSystem,
    tuple_xor, IneqInfo, shift_ineq,
    GrowingLowerFrozen,
)
from divprop.system_learn import (
    SupportLearner, RandomMaxFeasible,
    UnknownFillSAT, UnknownFillMILP,
    SATVerifier, Verifier,
)

from divprop import logging

logging.setup(level="INFO")
log = logging.getLogger()

# seed(1)

n = 8

prob = 0.2
d = DenseSet(n)
for x in range(2**n):
    if random() < prob:
        d.add(x)

ddt = d
print(d)

hs1 = ""
hs2 = ""
itr = 0
itreal = 0

sources = defaultdict(list)
sources = defaultdict(list)
cnt = defaultdict(int)

G = GrowingLowerFrozen(n=2**n)

for a in range(2**n):
    itr += 1
    if a in ddt:
        continue

    itreal += 1

    d = ddt.Not(a)
    good = d.MinSet()
    d.do_UpperSet()
    d.do_Complement()
    fullsz = len(d)
    # d.do_MaxSet()
    bad = d

    log.info(
        f"#{itr}/{2**n} ({itreal}): a = {Bin(a, n).str} = {Bin(a, n).hex} | "
        f"up {len(good)} lo {len(bad)} full {fullsz}"
    )

    shift = Bin(a, n)

    def get_pool(use_prec):
        pool = InequalitiesPool(
            points_good=good.to_Bins(),
            points_bad=bad.to_Bins(),
            type_good="upper",
            use_point_prec=use_prec,
        )
        sysfile = None

        pool.set_oracle(LPbasedOracle(solver="sage/glpk"))
        pool.set_system(LazySparseSystem(sysfile=sysfile))

        try:
            sat = UnknownFillSAT(
                minimization=False,
                save_rate=100,
                solver="cadical",
            )
            sat.init(system=pool.system, oracle=pool.oracle)
            sat.learn(num=10**6)
        except EOFError:
            log.info(f"solved, feasible set: {len(pool.system.feasible)}")
        else:
            print("ouch??")
            quit()
        return pool

    pool = get_pool(use_prec=True)
    tmp = DenseSet(n)
    # print("a:", Bin(a, n).str)
    for fset in pool.system.feasible:
        qs = [pool.i2bad[i] for i in fset]

        tmp.clear()
        for q in qs:
            tmp.set(Bin(q).int)
        mx = list(tmp.MaxSet())
        tmp.do_LowerSet()
        tmp.do_Not(a)

        fset = frozenset(tmp)
        sources[fset].append(a)

        vand = reduce(lambda a, b: a & b, mx)
        # if a & vand:
        #     continue
        cnt[fset, a & (~vand), vand] += 1
        # G.add(fset)

        # print(*[Bin(v, n).str for v in fset], "&:", Bin(vand, n))

for key, num in cnt.items():
    fset, abase, vand = key
    if num == 2**Bin(vand).weight:
        G.add(fset)


import hashlib

# for fset, alst in sources.items():
#     print(*[Bin(v, n).str for v in fset])
#     print(*[Bin(a, n).str for a in alst])
#     print()

G0 = list(G)
G.do_MaxSet()
G1 = list(G)
hs0 = sorted(map(str, G0))
hs1 = sorted(map(str, G1))
h0 = hashlib.sha1(str(hs0).encode()).hexdigest()
h1 = hashlib.sha1(str(hs1).encode()).hexdigest()
assert h0 == h1

for g0 in G0:
    for g1 in G1:
        if g0 < g1:
            print("redundant", g0, "<", g1)
            print(*[Bin(v, n).str for v in g0], "<")
            print(*[Bin(v, n).str for v in g1])
            print(*[Bin(a, n).str for a in sources[g0]])
            print(*[Bin(a, n).str for a in sources[g1]])
            print()


'''

*1*100
<
*1*100
11110*

011100 a1
110100 a2

111100 a



0101**
*10100
01*100

010100 a

'''
