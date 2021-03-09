from divprop.inequalities.monopool import MonotoneInequalitiesPool
from divprop.inequalities.base import satisfy, inner
from divprop.subsets import DenseSet
from divprop import logging

logging.setup()
log = logging.getLogger()

fileprefix = "/work/division/workspace/data/sbox_skinny_4/divcore.lb"

points_good = DenseSet.load_from_file(fileprefix + ".good.set")
points_bad = DenseSet.load_from_file(fileprefix + ".bad.set")

with open(fileprefix + ".type_good") as f:
    type_good = f.read().strip()
    assert type_good in ("upper", "lower")

log.info(f"points_good: {points_good}")
if len(points_good) < 100:
    log.info(f"{list(points_good)}")
log.info(f" points_bad: {points_bad}")
if len(points_bad) < 100:
    log.info(f"{list(points_bad)}")
log.info(f"  type_good: {type_good}")

if type_good == "lower":
    assert points_bad <= points_good.LowerSet().Complement()
else:
    assert points_bad <= points_good.UpperSet().Complement()

points_good
pool = MonotoneInequalitiesPool(
    points_good=points_good.to_Bins(),
    points_bad=points_bad.to_Bins(),
    type_good=type_good,
)
print(pool.oracle.n_calls)
pool.LSL.log_info()

# pool.gen_hats()
# for i in range(1000):
#     pool.gen_random_inequality()

pool.gen_dfs()

print("calls", pool.oracle.n_calls)
pool.LSL.log_info()


for fset in pool.LSL.infeasible:
    assert not pool.oracle.query(fset)

for fset, data in pool.LSL.feasible.items():
    print("fset", sorted(fset), "ineq", data.ineq)
    assert all(satisfy(q, data.ineq) for q in pool.hi)
    assert all(not satisfy(pool.i2lo[i], data.ineq) for i in fset)

# fset [3, 8, 10, 11, 13, 14] ineq (5, 2, 1, 7, 3, 5, 4, 8, -11)
# fset [8, 9, 11, 14, 15, 16] ineq (2, 1, 4, 3, 3, 5, 7, 8, -10)
# fset [0, 2, 3, 4, 5, 6, 8] ineq (4, 3, 1, 4, 2, 0, 3, 2, -6)
# fset [1, 2, 3, 4, 7, 8, 14] ineq (7, 5, 2, 8, 1, 3, 5, 4, -11)
# fset [6, 7, 8, 9, 11] ineq (3, 0, 3, 3, 2, 2, 5, 4, -7)

# fset [3, 5, 6, 8, 9, 11, 17] ineq (5, 3, 2, 6, 6, 1, 8, 7, -12)
# fset [0, 1, 2, 3, 7, 8, 12] ineq (8, 6, 5, 10, 1, 4, 3, 5, -13)
