from random import shuffle, random
from itertools import combinations

from collections import Counter, defaultdict

from binteger import Bin
from subsets import SparseSet

from divprop.logs import logging

from .LearnModule import LearnModule


class RandomLearn(LearnModule):
    log = logging.getLogger(f"{__name__}")

    prob_lower = 0.5

    def __init__(
            self,
            max_repeat_rate: float = 0.9,
            save_rate: int = 100,
            combinatorial_level: int = 2,
            limit: int = None
        ):
        self.max_repeat_rate = float(max_repeat_rate)
        self.save_rate = float(save_rate)
        self.combinatorial_level = int(combinatorial_level)
        self.limit = None if limit is None else int(limit)

        self._options = self.__dict__.copy()

    def learn(self):
        self.itr = 0
        while self.limit is None or self.itr < self.limit:
            if self.itr and self.itr % self.save_rate == 0:
                self.system.save()
            self.itr += 1

            if random() < self.prob_lower:
                self.sample_lower()
            else:
                self.sample_upper()

    def sample_lower(self):
        order = list(range(self.N))
        shuffle(order)

        fset = SparseSet([order.pop()])

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


class RandomLower(RandomLearn):
    prob_lower = 1.0


class RandomUpper(RandomLearn):
    prob_lower = 0.0
