import time

from random import shuffle, random
from itertools import combinations

from collections import Counter, defaultdict

from binteger import Bin

from divprop.subsets import SparseSet

from divprop.logs import logging

from .LearnModule import LearnModule


class LevelLearn(LearnModule):
    log = logging.getLogger(f"{__name__}:LevelLearn")

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

