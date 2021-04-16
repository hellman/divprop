import time

from random import shuffle, random
from itertools import combinations

from collections import Counter, defaultdict

from binteger import Bin
from subsets import SparseSet

from divprop.logs import logging

from divprop.milp import MILP
from divprop.sat import CNF

log = logging.getLogger(__name__)


# ========================================================
# Not updated
# ========================================================


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


def truncstr(s, n=100):
    s = str(s)
    if len(s) > n:
        s = s[:n] + "..."
    return s
