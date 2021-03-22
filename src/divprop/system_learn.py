import time
import logging

from random import shuffle
from itertools import combinations

from collections import Counter, defaultdict

from binteger import Bin

from divprop.milp import MILP
from divprop.inequalities.monopool import IneqInfo


log = logging.getLogger(__name__)


class LearnModule:
    def init(self, system, oracle):
        self.N = system.N
        self.system = system
        self.oracle = oracle
        self.fset_full = frozenset(range(self.N))
        self.milp = None
        self.itr = 0

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
                f"initializing feasible constraints: {len(self.system.feasible)}"
            )
            for fset in self.system.infeasible:
                self.milp_exclude_supercliques(fset)
            self.log.info(
                f"initializing infeasible constraints: {len(self.system.infeasible)}"
            )
            for fset in self.system.feasible:
                self.milp_exclude_subcliques(fset)
            self.log.info("init done")

    def milp_exclude_subcliques(self, fset):
        if not self.milp:
            return
        self.milp.add_constraint(
            sum(self.xs[i] for i in range(self.N) if i not in fset) >= 1
        )

    def milp_exclude_supercliques(self, fset):
        if not self.milp:
            return
        self.milp.add_constraint(
            sum(self.xs[i] for i in fset) <= len(fset) - 1
        )

    def middle_infeasible_learn(self, fset):
        if fset in self.system.infeasible:
            return

        log.info(f"itr #{self.itr}: learning from inf. wt={len(fset)}: {tuple(fset)}")
        orig = fset

        self.bad_learn_hard_limit = 100
        self.max_repeated_streak = 3

        stat = Counter()
        repeated_streak = 0

        for itr in range(self.bad_learn_hard_limit):  # hard limit
            fset = orig
            inds = list(orig)
            shuffle(inds)
            fail = False
            for i in inds:
                and_fset = fset - {i}

                if and_fset in self.system.infeasible.cache:
                    fail = True
                    break

                ineq = (and_fset in self.system.feasible.cache) or \
                    self.oracle.query(Bin(and_fset, self.N))

                if not ineq:
                    fset = and_fset
            if fail:
                repeated_streak += 1
                if repeated_streak >= self.max_repeated_streak:
                    # log.info("stop because repeated streak")
                    break
                continue
            repeated_streak = 0

            assert fset not in self.system.infeasible.cache

            stat[len(fset)] += 1

            self.system.add_infeasible(fset)
            self.milp_exclude_supercliques(fset)

            if fset == orig:
                # log.info("stop because can not reduce")
                break

        statstr = " ".join(f"{l}:{cnt}" for l, cnt in sorted(stat.items()))
        self.log.info(f"learnt new infeasibles, stat: {statstr}")

    def middle_feasible_learn(self, fset, sol):
        if fset in self.system.feasible:
            return

        log.info(f"itr #{self.itr}: learning from feas. wt={len(fset)}: {tuple(fset)}")
        orig = fset

        self.good_learn_hard_limit = 250
        self.max_repeated_streak = 5

        stat = Counter()
        repeated_streak = 0

        for itr in range(self.good_learn_hard_limit):  # hard limit
            fset = orig
            inds = list(self.fset_full - orig)
            shuffle(inds)
            fail = False
            cursol = sol
            for i in inds:
                and_fset = fset | {i}

                if and_fset in self.system.feasible.cache:
                    fail = True
                    break

                ineq = (and_fset in self.system.feasible.cache) or \
                    self.oracle.query(Bin(and_fset, self.N))

                if ineq:
                    cursol = ineq
                    fset = and_fset
            if fail:
                repeated_streak += 1
                if repeated_streak >= self.max_repeated_streak:
                    break
                continue
            repeated_streak = 0

            assert fset not in self.system.feasible.cache
            assert cursol is not True

            stat[len(fset)] += 1

            self.system.add_feasible(
                fset, sol=IneqInfo(cursol, "LearnFeas")
            )
            self.milp_exclude_subcliques(fset)

            if fset == orig:
                break

        statstr = " ".join(f"{l}:{cnt}" for l, cnt in sorted(stat.items()))
        self.log.info(f"learnt new feasibles, stat: {statstr}")


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

    def __init__(self, base_level=2, refresh_rate=50):
        assert base_level >= 2
        self.base_level = int(base_level)
        self.refresh_rate = int(refresh_rate)

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
            if self.itr % self.refresh_rate == 1:
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
                    or self.oracle.query(Bin(fset2, self.N))
            else:
                ineq = (fset2 in self.system.feasible) \
                    or self.oracle.query(Bin(fset2, self.N))

            if ineq:
                fset = fset2
                sol = ineq
            else:
                self.middle_infeasible_learn(fset2)

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

    def __init__(self, solver="gurobi", batch_size=10, refresh_rate=5):
        self.refresh_rate = int(refresh_rate)
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
            f"searching for unknowns, "
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
            if self.itr % self.refresh_rate == 1:
                self.refresh()
            if not self.find_new_unknown():
                if level is None:
                    self.refresh()
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

        if self.maximization:
            self.milp.add_constraint(self.xsum <= size)
        else:
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

            ineq = self.oracle.query(Bin(fset, self.N))

            self.log.info(f"ineq: {ineq}")

            if self.maximization:
                if ineq:
                    self.n_good += 1
                    self.system.add_feasible(
                        fset, sol=IneqInfo(ineq, "UnknownFillMILP")
                    )
                    self.milp_exclude_subcliques(fset)
                else:
                    self.n_bad += 1
                    self.middle_infeasible_learn(fset)
            else:
                if ineq:
                    self.n_good += 1
                    self.middle_feasible_learn(fset, sol)
                else:
                    self.n_bad += 1
                    self.system.add_infeasible(fset)
                    self.milp_exclude_supercliques(fset)
        return True


class Verifier(LearnModule):
    log = logging.getLogger(f"{__name__}:Verifier")

    def __init__(self, solver=None):
        self.solver = solver

    def learn(self, clean=False):
        self.log.info("refreshing system")
        self.system.refresh()

        self.log.info("verifying system")

        for fset in self.system.feasible:
            assert self.oracle.query(Bin(fset, self.N))

        self.log.info("feasible good!")

        for fset in self.system.infeasible:
            assert not self.oracle.query(Bin(fset, self.N))

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
