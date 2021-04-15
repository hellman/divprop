import time

from random import shuffle
from itertools import combinations

from collections import Counter, defaultdict

from binteger import Bin

from divprop.subsets import SparseSet

from divprop.logs import logging

from divprop.milp import MILP
from divprop.sat import CNF

log = logging.getLogger(__name__)


class LearnModule:
    use_point_prec = True

    def init(self, system, oracle):
        self._options = self.__dict__.copy()

        self.N = system.N
        self.system = system
        self.oracle = oracle

        self.milp = None
        self.sat = None

        self.itr = 0
        self.n_upper = 0
        self.n_lower = 0

        self.vec_full = SparseSet(range(self.N))

        if self.system.extra_prec is None:
            self.use_point_prec = False

    def query(self, vec):
        if self.use_point_prec:
            vec = self.system.extra_prec.reduce(vec)
        return self.oracle(vec)

    def milp_init(self, maximization=True, init=True):
        if maximization:
            self.milp = MILP.maximization(solver=self.solver)
        else:
            self.milp = MILP.minimization(solver=self.solver)

        self.xs = [self.milp.var_binary("x%d" % i) for i in range(self.N)]
        self.xsum = self.milp.var_int("xsum", lb=2, ub=self.N)
        self.milp.add_constraint(sum(self.xs) == self.xsum)

        if maximization is not None:
            self.milp.set_objective(self.xsum)

        if init:
            self.log.info(
                "milp: initializing "
                f"feasible constraints: {len(self.system.lower)}"
            )
            for vec in self.system.upper:
                self.model_exclude_super(vec)

            self.log.info(
                "milp: initializing "
                f"infeasible constraints: {len(self.system.upper)}"
            )
            for vec in self.system.lower:
                self.model_exclude_sub(vec)

            self.log.info("milp: initialization done")

    def sat_init(self, init_sum=True, init=True):
        self.sat = CNF(solver=self.solver)

        self.xs = [self.sat.var() for i in range(self.N)]
        if init_sum:
            self.xsum = self.sat.SeqAddMany(*[[x] for x in self.xs])
            self.xsum.append(self.sat.ZERO)  # padding

        if init:
            self.log.info(
                "sat: initializing "
                f"feasible constraints: {len(self.system.lower)}"
            )
            for vec in self.system.upper:
                self.model_exclude_super(vec)

            self.log.info(
                "sat: initializing "
                f"infeasible constraints: {len(self.system.upper)}"
            )
            for vec in self.system.lower:
                self.model_exclude_sub(vec)

            self.log.info("sat: initialization done")

    def model_exclude_sub(self, vec):
        if self.use_point_prec:
            vec = self.system.extra_prec.expand(vec)

        if self.milp:
            self.milp.add_constraint(
                sum(self.xs[i] for i in range(self.N) if i not in vec) >= 1
            )

        if self.sat:
            self.sat.add_clause(tuple(
                self.xs[i] for i in self.fset_full - vec
            ))

    def model_exclude_super(self, vec):
        if self.use_point_prec:
            vec = self.system.extra_prec.reduce(vec)

        if self.milp:
            self.milp.add_constraint(
                sum(self.xs[i] for i in vec) <= len(vec) - 1
            )

        if self.sat:
            self.sat.add_clause(tuple(
                -self.xs[i] for i in vec
            ))

    def learn_down(self, vec: SparseSet, meta=None):
        """reduce given upper element to minimal one"""
        if vec in self.system.upper:
            return

        log.debug(
            f"itr #{self.itr}: learning down from upper "
            f" wt {len(vec)}: {truncrepr(vec)}"
        )

        inds = list(vec)
        shuffle(inds)
        for i in inds:
            subvec = vec - i
            assert subvec not in self.system.upper
            if subvec in self.system.lower:
                continue

            is_lower, new_meta = self.query(subvec)
            if is_lower:
                continue

            vec = subvec
            meta = new_meta

        assert vec not in self.system.lower
        assert vec not in self.system.upper

        self.system.add_upper(vec, meta=meta, is_prime=True)
        self.model_exclude_super(vec)
        self.log.debug(
            f"learnt minimal upper vec wt {len(vec)}: {truncrepr(vec)}"
        )

    def learn_up(self, vec: SparseSet, meta=None):
        """lift given lower element to a maximal one"""
        if vec in self.system.lower:
            return

        log.debug(
            f"itr #{self.itr}: learning up from lower "
            f" wt {len(vec)}: {truncrepr(vec)}"
        )

        inds = self.vec_full - vec
        shuffle(inds)
        for i in inds:
            supvec = vec + i
            assert supvec not in self.system.lower
            if supvec in self.system.upper:
                continue

            is_lower, new_meta = self.query(supvec)
            if not is_lower:
                continue

            vec = supvec
            meta = new_meta

        assert vec not in self.system.lower
        assert vec not in self.system.upper

        self.system.add_lower(vec, meta=meta, is_prime=True)
        self.model_exclude_sub(vec)
        self.log.debug(
            f"learnt maximal lower vec wt {len(vec)}: {truncrepr(vec)}"
        )


class GainanovSAT(LearnModule):
    log = logging.getLogger(f"{__name__}:GainanovSAT")

    def __init__(
            self,
            sense: str = None,  # min/max/None
            solver: str = "cadical",
            save_rate=100,
            limit: int = None,
        ):
        assert self.sense in ("min", "max", None)
        self.do_min = self.sense == "min"
        self.do_max = self.sense == "max"
        self.do_opt = self.sense in ("min", "max")
        self.solver = solver
        self.save_rate = int(save_rate)
        self.limit = None if limit is None else int(limit)

    def learn(self):
        self.log.info(f"options: {self._options}")

        self.sat_init(init_sum=self.do_opt)

        self.level = None
        if not self.do_opt:
            # check if not exhausted
            unk = self.find_new_unknown(skip_optimization=True)
            if not unk:
                self.log.info("already exhausted, exiting")
                raise EOFError("all groups exhausted!")

            if self.do_min:
                self.level = 0
            elif self.do_max:
                self.level = self.N

            self.log.info(f"starting at level {self.level}")

        self.itr = 0
        while self.limit is None or self.itr < self.limit:
            if self.itr and self.itr % self.save_rate == 0:
                self.system.save()
            self.itr += 1

            unk = self.find_new_unknown()
            if not unk:
                self.refresh()
                raise EOFError("all groups exhausted!")
            self.learn_unknown(unk)

        self.refresh()

    def find_new_unknown(self, skip_optimization=False):
        while True:
            # <= level
            self.log.debug(
                f"itr #{self.itr}: optimizing (level={self.level})..."
            )

            assum = ()
            if not skip_optimization:
                if self.is_min:
                    # <= self.level
                    assum = [-self.xsum[self.level]]
                elif self.is_max:
                    # >= self.level
                    assum = [self.xsum[self.level]]

            sol = self.sat.solve(assumptions=assum)
            self.log.debug(f"SAT solve: {bool(sol)}")
            if sol:
                vec = SparseSet(
                    i for i, x in enumerate(self.xs) if sol[x] == 1
                )
                self.log.debug(
                    f"unknown #{self.itr}, wt {len(vec)}: {truncrepr(vec)} "
                    f"(upper: {self.n_upper}, lower: {self.n_lower})"
                )
                assert vec
                if self.level is not None:
                    assert len(vec) == self.level, "start level set incorrectly?"
                return vec

            # no sol at current level
            if not skip_optimization:
                if self.is_min:
                    self.level += 1
                    if self.level > self.N:
                        self.log.info("no new unknowns")
                        return False
                    self.log.info(f"increasing level to {self.level}")

                elif self.is_max:
                    self.level -= 1
                    if self.level < 0:
                        self.log.info("no new unknowns")
                        return False
                    self.log.info(f"decreasing level to {self.level}")
        assert 0

    def learn_unknown(self, vec):
        is_lower, meta = self.query(vec)

        self.log.debug(f"meta: {meta}")

        if is_lower:
            self.n_lower += 1
            if self.is_max:
                self.system.add_lower(vec, meta)
            else:
                self.learn_up(vec, meta)
        else:
            self.n_upper += 1
            if self.is_min:
                self.system.add_upper(vec, meta)
            else:
                self.learn_down(vec, meta)


# ========================================================
# Not updated
# ========================================================

class SupportLearner(LearnModule):
    def __init__(self, level=2):
        self.level = int(level)

    def learn(self):
        self.log = logging.getLogger(f"{__name__}:{type(self).__name__}")

        system = self.system
        oracle = self.oracle
        N = self.N

        start = getattr(system, "_support_learned", 1) + 1

        if start > self.level:
            self.log.info(
                f"support-{start-1} already learned "
                f"(requested {self.level})"
            )
            return

        sol_encoder = lambda ineq: IneqInfo(ineq, f"Support-{self.level}")

        self.log.info(
            f"generating support-{self.level} graph "
            f"(exhausting pairs, triples, ..., {self.level}-hyperedges), "
            f"starting from {start}"
        )

        for l in range(start, self.level+1):
            self.log.info(f"generating support, height={l}/{self.level}")

            n_good = 0
            n_total = 0

            if l == 2:
                # exhaust all pairs
                for inds in combinations(range(N), l):
                    fset = system.encode_bad_subset(inds)
                    n_total += 1

                    if fset in system.feasible.cache:
                        n_good += 1
                        continue
                    elif fset in system.infeasible.cache:
                        continue

                    ineq = oracle.query(Bin(fset, N))
                    if ineq:
                        system.add_feasible(fset, sol=sol_encoder(ineq))
                        n_good += 1
                    else:
                        system.add_infeasible(fset)
            else:
                # only extend feasible pairs/triples/etc.
                for prev_fset in system.feasible.cache[l-1]:
                    for k in range(max(prev_fset)+1, N):
                        fset = prev_fset | {k}
                        n_total += 1

                        good = 1
                        for j in prev_fset:
                            if (fset - {j}) in system.infeasible.cache:
                                good = 0
                                break
                        if not good:
                            continue

                        if fset in system.feasible.cache:
                            n_good += 1
                            continue
                        elif fset in system.infeasible.cache:
                            continue

                        ineq = oracle.query(Bin(fset, N))
                        if ineq:
                            system.add_feasible(fset, sol=sol_encoder(ineq))
                            n_good += 1
                        else:
                            system.add_infeasible(fset)

            self.log.info(
                f"generated support, height={l}/{self.level}: "
                f"feasible {n_good}/{n_total} "
                f"(frac. {(n_good+1)/(n_total+1):.3f})"
            )

            setattr(system, "_support_learned", l)


class RandomMaxFeasible(LearnModule):
    log = logging.getLogger(f"{__name__}:RandomMaxFeasible")

    def __init__(self, base_level=2, extremize_rate=50):
        assert base_level >= 2
        self.base_level = int(base_level)
        self.extremize_rate = int(extremize_rate)

        self._options = self.__dict__.copy()

    def learn(self, num=1_000_000):
        self.n_sample_feas = 0
        self.n_sample_feas_new = 0
        self.n_sample_infeas = 0
        self.n_sample_infeas_new = 0
        self.n_itr = num
        self.time_stats = defaultdict(lambda: 0)
        self.run_stats = defaultdict(lambda: 0)

        SL = SupportLearner(level=self.base_level)
        SL.init(system=self.system, oracle=self.oracle)
        SL.learn()

        self.itr = 0
        while self.itr < num:
            self.itr += 1
            if self.itr % self.extremize_rate == 1:
                self.refresh()

            method = self.sample_random_max_feasible
            t0 = time.time()
            method()
            elapsed = time.time() - t0

            self.time_stats[method] += elapsed
            self.run_stats[method] += 1

    def refresh(self):
        self.log.info("")
        self.log.info("---------------------------")
        self.log.info(f"refreshing, iteration {self.itr}/{self.n_itr}, stats:")
        for method in self.time_stats:
            avgtime = self.time_stats[method]/(0.1+self.run_stats[method])
            self.log.info(
                f"method {method.__name__}: "
                f"runs {self.run_stats[method]} "
                f"avg.time {avgtime:.3f}s"
            )
        self.log.info("before extremizing")
        self.system.log_info()
        self.log.info("system refresh: extremizing")
        self.system.refresh()
        self.log.info("---------------------------")
        self.log.info("")

    def sample_random_max_feasible(self, lp_only=False):
        self.n_sample_feas += 1
        order = list(range(self.N))
        shuffle(order)

        fset = frozenset([order.pop()])
        sol = True  # could retrieve self.system.solution[fset] but not useful
        for i in order:
            fset2 = fset | {i}
            if fset2 in self.system.infeasible.cache:
                continue

            is_bad = False
            for l in range(1, self.base_level):
                for sub in combinations(fset, l):
                    if frozenset(sub + (i,)) in self.system.infeasible.cache:
                        is_bad = True
                        break
                if is_bad:
                    break
            if is_bad:
                continue

            if lp_only:
                ineq = (fset2 in self.system.feasible.cache) \
                    or self.query(fset2)
            else:
                ineq = (fset2 in self.system.feasible) \
                    or self.query(fset2)

            if ineq:
                fset = fset2
                sol = ineq
            else:
                self.learn_down(fset2)

        # self.log.info("repeated max-feasible")
        if fset in self.system.feasible.cache:
            return
        assert fset not in self.system.infeasible.cache

        self.n_sample_feas_new += 1
        self.log.info(
            f"random max-feasible #{self.n_sample_feas_new} / "
            f"tries {self.n_sample_feas+1}, "
            f"size {len(fset)}: {tuple(fset)}"
        )

        assert sol is not True
        self.system.add_feasible(
            fset, sol=IneqInfo(sol, "sample_random_extreme_feasible")
        )


class UnknownFillMILP(LearnModule):
    log = logging.getLogger(f"{__name__}:UnknownFillMILP")

    def __init__(self, solver="gurobi", batch_size=10, extremize_rate=5):
        self.extremize_rate = int(extremize_rate)
        self.solver = solver
        self.batch_size = int(batch_size)

        self._options = self.__dict__.copy()

    def refresh(self):
        self.log.info("")
        self.log.info("---------------------------")
        self.log.info(f"refreshing, iteration {self.itr}")
        self.system.refresh()
        self.log.info("---------------------------")
        self.log.info("")

    def learn(self, maximization=False, num=10, level=None):
        self.log.info(
            f"searching for {num} iterations, "
            f"maximization? {maximization}, "
            f"fixed level {level}"
        )
        self.maximization = maximization
        self.milp_init(maximization=maximization)

        if level is not None:
            self.milp.add_constraint(self.xsum == level)

        self.n_good = 0
        self.n_bad = 0

        self.itr = 0
        while self.itr < num:
            self.itr += 1
            if self.itr % self.extremize_rate == 1:
                self.refresh()
            if not self.find_new_unknown():
                if level is None:
                    self.refresh()
                    self.log.info(f"exhausted on #{self.itr}")
                    raise EOFError("all groups exhausted!")
                break

        self.refresh()

    def find_new_unknown(self):
        self.log.info(f"itr #{self.itr}: optimizing...")
        size = self.milp.optimize(solution_limit=self.batch_size)
        if size is None:
            self.log.info(f"no new cliques, milp.err: {self.milp.err}")
            return False

        assert isinstance(size, int), size
        assert self.milp.solutions

        if self.maximization is True:
            self.milp.add_constraint(self.xsum <= size)
        elif self.maximization is False:
            self.milp.add_constraint(self.xsum >= size)

        for sol in self.milp.solutions:
            fset = self.system.encode_bad_subset(
                i for i, x in enumerate(self.xs) if sol[x] > 0.5
            )
            self.log.info(
                f"clique #{self.itr}, size {size}: "
                f"{tuple(fset)} (good: {self.n_good}, bads: {self.n_bad})"
            )
            assert fset

            ineq = self.query(fset)

            self.log.info(f"ineq: {ineq}")

            if self.maximization:
                if ineq:
                    self.n_good += 1
                    self.system.add_feasible(
                        fset, sol=IneqInfo(ineq, "UnknownFillMILP")
                    )
                    self.model_exclude_sub(fset)
                else:
                    self.n_bad += 1
                    self.learn_down(fset)
            else:
                if ineq:
                    self.n_good += 1
                    self.learn_up(fset, ineq)
                else:
                    self.n_bad += 1
                    self.system.add_infeasible(fset)
                    self.model_exclude_super(fset)
        return True





class Verifier(LearnModule):
    log = logging.getLogger(f"{__name__}:Verifier")

    def __init__(self, solver=None):
        self.solver = solver

    def learn(self, clean=False):
        self.system.refresh()

        self.log.info("verifying system")

        for fset in self.system.feasible:
            assert self.query(fset)

        self.log.info("feasible good!")

        for fset in self.system.infeasible:
            assert not self.query(fset)

        self.log.info("infeasible good!")

        self.milp_init()

        res = self.milp.optimize()
        self.log.info(f"milp optimize: {res}")
        assert res is None, "not all cliques explored!"

        self.log.info("all good!")

        if clean:
            self.system.feasible.clean_cache()
            self.system.infeasible.clean_cache()
            self.system.save()
            self.log.info("clean done!")


class SATVerifier(LearnModule):
    log = logging.getLogger(f"{__name__}:SATVerifier")

    def __init__(self, solver=None):
        self.solver = solver

    def learn(self, clean=False, correctness=True, completeness=True):
        self.system.refresh()

        if correctness:
            self.log.info("verifying correctness")

            for fset in self.system.feasible:
                assert self.query(fset)

            self.log.info("feasible good!")

            for fset in self.system.infeasible:
                assert not self.query(fset)

            self.log.info("infeasible good!")

        if completeness:
            self.log.info("verifying completeness")

            self.sat_init(init_sum=False)
            res = self.sat.solve()
            self.log.info(f"sat solve: {bool(res)}")
            assert not res, "not all cliques explored!"

        self.log.info("all good!")

        if clean:
            self.log.info("cleaning")
            self.system.feasible.clean_cache()
            self.system.infeasible.clean_cache()
            self.system.save()
            self.log.info("clean done!")


def truncrepr(s, n=100):
    s = repr(s)
    if len(s) > n:
        s = s[:n] + "..."
    return s
