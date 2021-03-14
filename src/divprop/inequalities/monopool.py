import logging
from random import sample, randrange, choices, shuffle
from collections import Counter, namedtuple, defaultdict
from itertools import combinations

from tqdm import tqdm
from binteger import Bin

from .base import (
    inner, satisfy,
    MixedIntegerLinearProgram, MIPSolverException,
    Polyhedron,
)
from .random_group_cut import RandomGroupCut
from .gem_cut import GemCut

from divprop.learn import DenseLowerSetLearn
from divprop.subsets import (
    neibs_up_tuple, not_tuple, support_int_le,
    WeightedFrozenSets,
    GrowingLowerFrozen,
    GrowingUpperFrozen,
)

from divprop.milp import MILP


log = logging.getLogger(__name__)


IneqInfo = namedtuple("IneqInfo", ("ineq", "source"))


class Generator:
    def generate(self, pool):
        pass


class Hats(Generator):
    def make_ineq(self, top, subs):
        n = len(top)

        for u in subs:
            assert sum(top) - sum(u) == 1

        large = len(subs)
        eq = [0] * n + [large]

        # 0 bits in top
        for i in range(n):
            if top[i] == 0:
                eq[i] = large

        # single 0 bits in each relevant u
        for u in subs:
            for i in range(n):
                if u[i] == 0 and top[i] == 1:
                    eq[i] = 1
                    break
            else:
                assert 0, "something wrong"
        return eq

    def generate(self, pool):
        hats = defaultdict(list)
        for u in pool.bad:
            for supu in neibs_up_tuple(u):
                hats[supu].append(u)

        for top, subs in hats.items():
            ineq = self.make_ineq(top, subs)
            fset = pool.syatem.encode_bad_subset(
                pool.bad2i[q] for q in subs
            )
            pool.syatem.add_feasible(
                fset, sol=IneqInfo(ineq=ineq, source="hat"),
            )


class RandomPlaneCut(Generator):
    def __init__(self, max_coef=100, exp=-0.5):
        self.max_coef = int(max_coef)
        self.exp = float(exp)

    def generate(self, pool):
        lst = list(range(self.max_coef))
        wts = [1] + [i**self.exp for i in range(1, self.max_coef)]
        lin = choices(lst, wts, k=pool.n)
        # lin = [randrange(max_coef+1) for _ in range(pool.n)]
        ev_good = min(inner(p, lin) for p in pool.good)
        fset = pool.LSL.encode_bad_subset(
            i for i, q in enumerate(pool.i2bad) if inner(q, lin) < ev_good
        )
        # if fset:
        #     print("lin", lin)
        #     print(
        #         "good",
        #         min(inner(p, lin) for p in pool.good),
        #         max(inner(p, lin) for p in pool.good),
        #     )
        #     print(
        #         " bad",
        #         min(inner(p, lin) for p in pool.bad),
        #         max(inner(p, lin) for p in pool.bad),
        #     )
        #     print(fset)
        # lin. comb. >= ev_good
        ineq = tuple(lin) + (-ev_good,)
        pool.LSL.add_feasible(
            fset,
            sol=IneqInfo(ineq=ineq, source="random_ineq")
        )


class DFS(Generator):
    def __init__(self):
        pass

    def generate(self, pool):
        return pool.system.learn_simple(
            oracle=pool.oracle,
            sol_encoder=lambda ineq: IneqInfo(ineq, "DFS")
        )


class LPbasedOracle:
    def __init__(self, solver=None):
        self.solver = solver
        self.n_calls = 0

    def attach_to_pool(self, pool):
        self.pool = pool
        self._prepare_constraints()

    def _prepare_constraints(self):
        self.milp = MILP.maximization(solver=self.solver)

        if self.pool.type_good is None:  # not monotone
            lb = None
        else:
            lb = 0  # monotone => nonnegative

        # set ub = 1000+ ? ...
        self.xs = []
        for i in range(self.pool.n):
            self.xs.append(self.milp.var_real("x%d" % i, lb=lb, ub=None))
        self.c = self.milp.var_real("c", lb=lb, ub=None)

        for p in self.pool.good:
            self.milp.add_constraint(inner(p, self.xs) >= self.c)

        self.i2cs = []
        for q in self.pool.i2bad:
            self.i2cs.append(inner(q, self.xs) <= self.c - 1)

    def query(self, bads: Bin):
        assert isinstance(bads, Bin)
        bads = bads.support()

        self.n_calls += 1

        LP = self.milp
        cs = [LP.add_constraint(self.i2cs[i]) for i in bads]
        res = LP.optimize()
        LP.remove_constraints(cs)

        if res is None:
            return False

        sol = LP.solutions[0]
        val_xs = tuple(sol[x] for x in self.xs)
        val_c = sol[self.c]

        if not all(isinstance(v, int) for v in val_xs + (val_c,)):
            # keep real ineq, put the separator in the middle
            val_c -= 0.5

        ineq = val_xs + (-val_c,)
        assert all(satisfy(p, ineq) for p in self.pool.good)
        assert all(not satisfy(self.pool.i2bad[i], ineq) for i in bads)
        return ineq


class InequalitiesPool:
    def __init__(self, points_good, points_bad, type_good=None, system=None):
        for p in points_bad:
            self.n = len(p)
            break

        assert type_good in ("lower", "upper", None)
        self.type_good = type_good
        if type_good == "upper":
            self.bad = set(map(tuple, points_bad))
            self.good = set(map(tuple, points_good))
        else:
            # ensure good is an upper set
            self.bad = set(map(not_tuple, points_bad))
            self.good = set(map(not_tuple, points_good))

        self.i2bad = sorted(self.bad)
        self.bad2i = {p: i for i, p in enumerate(self.i2bad)}
        self.N = len(self.bad)

        self.system = system if system else LazySparseSystem()
        self.system.init(pool=self)

        if self.type_good in ("lower", "upper"):
            # initialize
            for pi, p in enumerate(self.i2bad):
                self.gen_basic_ineq(pi, p)

        self._oracle = None

    @property
    def oracle(self):
        if not self._oracle:
            self.oracle = LPbasedOracle()
        return self._oracle

    def set_oracle(self, oracle):
        self._oracle = oracle
        oracle.attach_to_pool(self)

    def gen_basic_ineq(self, pi, p):
        fset = self.system.encode_bad_subset((pi,))
        # does not belong to LowerSet({p})
        # <=>
        # sum coords with p_i = 0 >= 1
        ineq = tuple(1 if x == 0 else 0 for x in p) + (-1,)

        self.system.add_feasible(
            fset,
            sol=IneqInfo(ineq=ineq, source="basic-mono"),
        )

    def gen_random_inequality(self, max_coef=100, exp=-0.5):
        RandomPlaneCut(max_coef, exp).generate(self)

    def gen_hats(self):
        Hats().generate(self)

    def gen_dfs(self):
        DFS().generate(self)

    # tbd:
    # port polyhedron
    # port subset greedy

    def choose_subset_milp(self, solver=None):
        """
        [SecITC:SasTod17]
        Choose subset optimally by optimizing MILP system.
        """
        log.info(f"InequalitiesPool.choose_subset_milp(solver={solver})")
        log.info(f"{len(self.system.feasible)} ineqs {len(self.bad)} bad points")

        L = list(self.system.feasible)
        self.check_good(L)  # to avoid surprises
        eq2i = {eq: i for i, eq in enumerate(L)}

        milp = MILP.minimization(solver=solver)
        n = len(L)

        # xi = take i-th inequality?
        take_eq = [milp.var_binary("take_eq%d" % i) for i in range(n)]

        by_bad = {q: [] for q in self.points_bad}
        for eq, (source, covered) in self.pool.items():
            v_take_eq = take_eq[eq2i[eq]]
            for q in covered:
                by_bad[q].append(v_take_eq)

        # each bad point is removed by at least one ineq
        for q, lst in by_bad.items():
            assert lst, "no solutions"
            milp.add_constraint(sum(lst) >= 1)

        # minimize number of ineqs
        milp.set_objective(sum(take_eq))

        log.info(
            f"solving milp with {n} variables, "
            f"{len(self.points_bad)} constraints"
        )

        # show log for large problems
        milp.debug = 1
        res = milp.optimize(log=(n >= 10000))
        assert res is not None
        sol = milp.solutions[0]
        print("RES", res, "SOL", sol)

        Lstar = []
        for take, eq in zip(take_eq, L):
            if sol[take]:
                Lstar.append(eq)
        self.check(Lstar)
        return Lstar


class CliqueMountainHills:
    def __init__(
        self,
        base_level=3,
        max_mountains=0,
        min_height=10,
        max_repeated_streak=5,
        solver="scip",
    ):
        assert base_level >= 2
        self.base_level = int(base_level)
        self.max_mountains = int(max_mountains)
        self.min_height = int(min_height)
        self.max_repeated_streak = int(max_repeated_streak)
        self.solver = solver
        self.log = logging.getLogger(f"{__name__}:{type(self).__name__}")

    def learn_system(self, system, oracle, sol_encoder):
        self.sys = system
        self.oracle = oracle
        self.N = system.N
        self.sol_encoder = sol_encoder

        n_calls0 = self.oracle.n_calls

        # =================================

        self.generate_base()
        self.generate_cliques()

        self.log.info("final statistics:")
        self.log.info(
            f"    {self.n_cliques} cliques enumerated: "
            f"{self.n_good} good, {self.n_bad} bad (sub)cliques"
        )
        self.log.info(f"    {self.oracle.n_calls - n_calls0} oracle calls")
        self.sys.log_info()

        self.sys.feasible.do_MaxSet()
        self.sys.clean_solution()

        self.log.info("after MaxSet")
        self.sys.log_info()

    def generate_base(self):
        for i in range(self.N):
            assert self.sys.encode_bad_subset([i]) in self.sys.feasible, \
                "single-point removal inequalities must be present"
            assert isinstance(self.sys.encode_bad_subset([i]), frozenset), \
                "frozenset assumed"

        for l in range(2, self.base_level+1):
            log.info(
                f"generating exhaustive base, height={l}/{self.base_level}"
            )
            n_good = 0
            n_total = 0
            if l == 2:
                # exhaust all pairs
                for inds in combinations(range(self.N), l):
                    fset = self.sys.encode_bad_subset(inds)

                    ineq = self.oracle.query(Bin(fset, self.N))
                    if ineq:
                        self.sys.add_feasible(
                            fset, sol=self.sol_encoder(ineq)
                        )
                        n_good += 1
                    else:
                        self.sys.add_infeasible(fset)
                    n_total += 1
            else:
                # only extend feasible pairs/triples/etc.
                for prev_fset in self.sys.feasible.cache[l-1]:
                    for k in range(max(prev_fset)+1, self.N):
                        fset = prev_fset | {k}

                        ineq = self.oracle.query(Bin(fset, self.N))
                        if ineq:
                            self.sys.add_feasible(
                                fset, sol=self.sol_encoder(ineq)
                            )
                            n_good += 1
                        else:
                            self.sys.add_infeasible(fset)
                        n_total += 1
            log.info(
                f"exhaustive base, height={l}/{self.base_level}: "
                f"feasible {n_good}/{n_total} "
                f"(frac. {(n_good+1)/(n_total+1):.3f})"
            )

    def exclude_subcliques(self, fset):
        self.milp.add_constraint(
            sum(self.xs[i] for i in range(self.N) if i not in fset)
            >= 1
        )

    def exclude_supercliques(self, fset):
        self.milp.add_constraint(
            sum(self.xs[i] for i in fset)
            <= len(fset) - 1
        )

    def generate_cliques(self):
        self.milp = MILP.maximization(solver=self.solver)
        try:
            self.milp.set_reopt()
        except AttributeError:
            pass

        self.xs = [self.milp.var_binary("x%d" % i) for i in range(self.N)]
        self.xsum = self.milp.var_int("xsum", lb=self.base_level+1, ub=self.N)
        self.milp.add_constraint(sum(self.xs) == self.xsum)
        self.milp.set_objective(self.xsum)

        # exclude super-cliques of known infeasible ones
        for l in range(2, self.base_level+1):
            for fset in self.sys.infeasible.iter_wt(l):
                self.exclude_supercliques(fset)

        self.n_cliques = 0
        self.n_bad = 0
        self.n_good = 0
        while True:
            size = self.milp.optimize(solution_limit=100, only_best=True)
            if size is None:
                log.info(f"no new cliques, milp.err: {self.milp.err}")
                break

            assert isinstance(size, int), size
            assert size > self.base_level + 0.5

            assert self.milp.solutions
            for sol in self.milp.solutions:
                log.info(
                    f"clique #{self.n_cliques} "
                    f"(batch {len(self.milp.solutions)}): "
                    f"{size} (bads: {self.n_bad})"
                )
                fset = self.sys.encode_bad_subset(
                    i for i, x in enumerate(self.xs) if sol[x] > 0.5
                )
                assert fset

                self.n_cliques += 1

                ineq = self.oracle.query(Bin(fset, self.N))

                log.info(f"ineq: {ineq}")
                if ineq:
                    self.process_good_clique(fset, ineq)
                else:
                    self.process_bad_clique(fset, ineq)

                self.milp.set_ub(self.xsum, size)

    def process_good_clique(self, fset, ineq):
        self.sys.add_feasible(fset, sol=self.sol_encoder(ineq))

        for i in fset:
            log.debug(f"{i:4d} {Bin(self.oracle.pool.i2bad[i]).str}")
        log.debug("")

        self.exclude_subcliques(fset)

        self.n_good += 1

        # take mountains
        # then hunt for the hills
        if self.n_good <= self.max_mountains and len(fset) >= self.min_height:
            log.info(
                f"mountain #{self.n_good}/{self.max_mountains}, "
                f"height {len(fset)}>={self.min_height}"
            )
            # exclude this variables (assume this points are already covered)
            for i in fset:
                self.milp.set_ub(self.xs[i], 0)

    def process_bad_clique(self, fset, ineq):
        # exclude this clique (& super-cliques)
        orig = fset
        # hard limit
        repeated_streak = 0
        for itr in range(200):
            inds = list(orig)
            shuffle(inds)
            # print("exclude itr", itr)
            for i in inds:
                and_fset = fset - {i}

                ineq = self.oracle.query(Bin(and_fset, self.N))
                # print("exclude itr", itr, i, "ineq", ineq)
                if ineq:
                    self.sys.add_feasible(and_fset, sol=self.sol_encoder(ineq))
                    break
                else:
                    fset = and_fset

            if fset not in self.sys.infeasible.cache:
                self.n_bad += 1
                log.info(f"exclude wt={len(fset)}: {fset}")
                # exclude this clique (&super-cliques since it's reduced)
                self.exclude_subcliques(fset)
                self.sys.add_infeasible(fset)
                repeated_streak = 0
            else:
                repeated_streak += 1
                if repeated_streak >= self.max_repeated_streak:
                    break


class LazySparseSystem:
    def init(self, pool):
        self.pool = pool
        self.N = int(self.pool.N)
        self.feasible = GrowingLowerFrozen(self.N)
        self.infeasible = GrowingUpperFrozen(self.N)
        self.solution = {}

    # NAIVE
    def is_already_feasible(self, v):
        # quick check
        if v in self.feasible_cache:
            return True
        # is in feasible lowerset?
        for u in self.feasible:
            # v <= u
            if v & u == v:
                return True
        return False

    def is_already_infeasible(self, v):
        # quick check
        if v in self.infeasible_cache:
            return True
        # is in infeasible upperset?
        for u in self.infeasible:
            # v >= u
            if v | u == v:
                return True
        return False

    def clean_solution(self):
        todel = [k for k in self.solution if k not in self.feasible]
        for k in todel:
            del self.solution[k]

    def encode_bad_subset(self, indexes):
        indexes = frozenset(indexes)
        assert all(0 <= i < self.N for i in indexes)
        return indexes

    def add_feasible(self, fset, sol=None):
        assert isinstance(fset, frozenset)
        self.feasible.add(fset)

        if sol is not None:
            self.solution[fset] = sol

    def add_infeasible(self, fset):
        assert isinstance(fset, frozenset)
        self.infeasible.add(fset)

    def remove_redundant(self):
        self.feasible.do_MaxSet()

    def log_info(self):
        log.info("stat:")
        for (name, s) in [
            ("feasible", self.feasible),
            ("infeasible", self.infeasible),
        ]:
            freq = Counter(len(v) for v in s)
            freqstr = " ".join(
                f"{sz}:{cnt}" for sz, cnt in sorted(freq.items())
            )
            log.info(f"   {name}: {len(s)}: {freqstr}")

    def learn_simple(self, oracle, sol_encoder=lambda v: v):
        self.oracle = oracle

        print("starting pairs")
        good_pairs = []
        bad_pairs = []
        for i, j in combinations(range(self.N), 2):
            fset = self.encode_bad_subset((i, j))
            ineq = oracle.query(Bin(fset, self.N))
            # print("visit", Bin(fset, self.N).str, ":", ineq)
            if not ineq:
                self.infeasible.add(fset)
                bad_pairs.append((i, j))
            else:
                self.feasible.add(fset)
                self.solution[fset] = sol_encoder(ineq)
                good_pairs.append((i, j))

        print("pairs done")
        print("n_calls", oracle.n_calls)
        self.log_info()
        print()

        known = 2

        if 0:
            bad_triples = set()
            for i, j in good_pairs:
                for k in range(j+1, self.N):
                    assert (i < j < k)

                    fset_ik = self.encode_bad_subset((i, k))
                    if fset_ik not in self.feasible:
                        continue
                    fset_jk = self.encode_bad_subset((j, k))
                    if fset_jk not in self.feasible:
                        continue

                    fset = self.encode_bad_subset((i, j, k))
                    ineq = oracle.query(Bin(fset, self.N))
                    if not ineq:
                        self.infeasible.add(fset)
                        bad_triples.add((i, j, k))
                    else:
                        self.feasible.add(fset)
                        self.solution[fset] = sol_encoder(ineq)

            print("triples done", len(bad_triples), "bad triples")
            print("n_calls", oracle.n_calls)
            self.log_info()
            print()
            known = 3
        print("===================================")

        # find cliques
        solver = "scip"
        solver = "gurobi"
        print("clique solver:", solver)
        if solver == "scip":
            from pyscipopt import Model
            model = Model()
            model.hideOutput()

            xs = [model.addVar("x%d" % i, vtype="B") for i in range(self.N)]
            xsum = model.addVar("xsum", vtype="I", lb=known+1, ub=self.N)
            model.addCons(xsum == sum(xs))
            model.setObjective(xsum)
            model.setMaximize()
            m_add_cons = model.addCons
            m_set_max = model.tightenVarUbGlobal
            def m_solve():
                model.optimize()
                status = model.getStatus()
                nsols = model.getNSols()
                print("nsols", nsols)
                assert status in ("optimal", "infeasible"), status
                if status == "optimal":
                    return model.getObjVal()
                raise MIPSolverException()
            m_get_val = model.getVal
        else:
            model = MixedIntegerLinearProgram(maximization=True, solver=solver)

            var = model.new_variable(binary=True)
            xs = [var["x%d" % i] for i in range(self.N)]
            xsum = model.new_variable(integer=True, nonnegative=True)["xsum"]
            model.add_constraint(xsum == sum(xs))
            model.set_min(xsum, known+1)
            model.set_objective(xsum)
            m_add_cons = model.add_constraint
            m_set_max = model.set_max
            m_solve = model.solve
            m_get_val = model.get_values

        for i, j in bad_pairs:
            m_add_cons(xs[i] + xs[j] <= 1)
        if known == 3:
            for i, j, k in bad_triples:
                m_add_cons(xs[i] + xs[j] + xs[k] <= 2)

        # vs = 0, 1, 3, 13
        # for i, j in combinations(vs, 2):
        #     assert (i, j) in good_pairs
        #     assert (i, j) not in bad_pairs
        # for i, j, k in combinations(vs, 3):
        #     assert (i, j, k) not in bad_triples

        # # for v in vs:
        # #     add_cons(xs[v] == 1)

        bads = set()
        goods = set()
        n_cliques = 0
        while True:
            try:
                obj = m_solve()
            except MIPSolverException as err:
                print("exception (no solution?):", err)
                break

            n_cliques += 1

            log.info(
                f"clique #{n_cliques}: {obj} "
                f"(bads: {len(bads)}, queries: {self.oracle.n_calls})"
            )

            assert obj > known + 0.5

            val_xs = tuple(m_get_val(x) for x in xs)
            assert all(abs(v - round(v)) < 0.00001 for v in val_xs)
            val_xs = tuple(int(v + 0.5) for v in val_xs)

            if solver == "scip":
                model.freeTransform()

            indic = Bin(val_xs, self.N)
            fset = frozenset(indic.support())

            ineq = oracle.query(indic)
            print("".join(map(str, val_xs)), ineq)
            if ineq:

                self.feasible.add(fset)
                self.solution[fset] = sol_encoder(ineq)
                for i in indic.support():
                    print("%3d" % i, Bin(oracle.pool.i2bad[i]).str)
                print()

                # exclude all subcliques
                m_add_cons(sum(
                    xs[i] for i, x in enumerate(val_xs) if x == 0
                ) >= 1)
                goods.add(indic)

                # take one mountain
                # then hunt for the hills
                if indic.hw() > 18:
                    print("taken the mountain!", len(goods))
                    for i in indic.support():
                        m_set_max(xs[i], 0)
            else:
                # exclude this clique (&overcliques)
                orig = indic
                for itr in range(100):
                    indic = orig

                    # print("removal itr", itr)
                    inds = list(indic.support())
                    shuffle(inds)

                    for i in inds:
                        and_fset = fset - {i}
                        and_indic = Bin(and_fset, self.N)

                        ineq = oracle.query(and_indic)
                        if ineq:
                            # print("degraded to GOOD", and_indic.hw(), and_fset)
                            self.feasible.add(and_fset)
                            self.solution[and_fset] = sol_encoder(ineq)
                        else:
                            # print("degraded to  BAD", and_indic.hw(), and_fset)
                            fset = and_fset
                            indic = and_indic

                    if indic not in bads:
                        print("exclude", indic.hw(), indic.support())
                        # exclude this clique (&overcliques since it's reduced)
                        m_add_cons(sum(
                            xs[i] for i, x in enumerate(indic.tuple) if x == 1
                        ) <= sum(indic.tuple) - 1)
                        bads.add(indic)
                    elif itr > 50:
                        break

            m_set_max(xsum, int(obj + 0.5))

        print("cliques enumerated", n_cliques)
        print("n_calls", oracle.n_calls)
        self.log_info()
        print()

        self.feasible.do_MaxSet()
        self.clean_solution()

        print("clean")
        print("n_calls", oracle.n_calls)
        self.log_info()
        print()
