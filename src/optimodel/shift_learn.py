import os, sys, pickle
from itertools import combinations
from collections import Counter

from binteger import Bin

from divprop.inequalities.monopool import (
    InequalitiesPool, LPbasedOracle, LazySparseSystem,
    tuple_xor, IneqInfo, shift_ineq,
)
from divprop.system_learn import (
    RandomMaxFeasible, GainanovSAT
)

from subsets import DenseSet
from divprop import logging
from divprop.tools import get_sbox, get_sbox_sizes

logging.setup(level="DEBUG")


class ShiftLearn:
    log = logging.getLogger(__name__)

    def __init__(self, pool, path, learn_chain):
        self.pool = pool
        if self.pool.is_monotone or self.pool.shift is not None:
            # convert to generic? tool
            raise ValueError(
                "ShiftLearn is only applicable to generic non-shifted sets"
            )

        self.good = DenseSet(list(map(int, self.pool.good)), self.pool.n)
        self.bad = DenseSet(list(map(int, self.pool.bad)), self.pool.n)
        if self.good.Complement() != self.bad:
            self.log.warning(
                "the implementation for don't care points"
                "was not carefully checked and tested"
            )

        self.path = path
        assert os.path.isdir(self.path)

    def process_shift(self, shift: Bin):
        # xor
        assert shift.n == self.pool.n
        s = self.good.copy()
        s.do_Not(shift.int)
        s.do_UpperSet()
        good = s.MinSet()
        s.do_Complement()
        removable = s

        bad = self.bad.copy()
        bad.do_Not(shift.int)
        bad &= removable
        # bad.do_LowerSet()  # unnecessary?! optimization

        # good is MinSet of the upper closure
        # bad is what can be removed within this shift
        #          (subset of the removable lower set)
        subpool = InequalitiesPool(
            points_good=good.to_Bins(),
            points_bad=bad.to_Bins(),
            type_good="upper",
            system=os.path.join(self.path, f"shift_{shift.hex}"),

        )



ddt = DenseSet(2*n)
for x in range(2**n):
    for dx in range(2**n):
        dy = sbox[x] ^ sbox[x ^ dx]
        ddt.set((dx << n) | dy)
nddt = ddt.Complement()
log.info(f"ddt compl {nddt}")

itr = 0
itreal = 0
for a in range(2**(2*n)):
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
        f"#{itr}/{4**n} ({itreal}): a = {Bin(a, 2*n).str} = {Bin(a, 2*n).hex} | "
        f"up {len(good)} lo {len(bad)} full {fullsz}"
    )

    shift = Bin(a, 2*n)
    pool = InequalitiesPool(
        points_good=good.to_Bins(),
        points_bad=bad.to_Bins(),
        type_good="upper",
    )
    sysfile = f"data/cache/qmc_{name}_a{a:x}"
    # sysfile = None

    pool.set_oracle(LPbasedOracle(solver="sage/glpk"))
    pool.set_system(LazySparseSystem(sysfile=sysfile))

    # for fset, sol in pool.system.solution.items():
    #     for q in pool.good:
    #         assert satisfy(q, sol.ineq)
    #     for i in fset:
    #         assert not satisfy(pool.i2bad[i], sol.ineq)

    # for fset, sol in pool.system.solution.items():
    #     for q in mainpool.good:
    #         assert satisfy(q, shift_ineq(sol.ineq, shift))

    good = 1
    try:
        Ver = SATVerifier(solver="cadical")
        Ver.init(system=pool.system, oracle=pool.oracle)
        Ver.learn(clean=False, correctness=False)
        log.info("existing system verify ok")
    except AssertionError:
        log.info("existing system verify fail, solving")
        good = 0

    if not good:
        try:
            sat = UnknownFillSAT(
                minimization=True,
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

    if itreal % 256 == 0:
        # ineqs = mainpool.choose_subset_greedy(1)
        # log.info(f"cur best greedy: {len(ineqs)}")
        mainpool.system.refresh(extremize=False)


mainpool.system.log_info()
mainpool.system.refresh(extremize=False)
# mainpool.system.refresh()

if 0:
    ineqs = mainpool.choose_subset_greedy(1)

    log.info(f"greedy: {len(ineqs)}")

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

# if len(ineqs) < 50:
#     for ineq in ineqs:
#         cnt = sum(1 for q in mainpool._bad_orig if not satisfy(q, ineq))
#         print(ineq, ":", cnt)
