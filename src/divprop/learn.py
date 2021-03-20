import logging

from tqdm import tqdm
from random import choice, shuffle, randrange
from itertools import combinations

from collections import Counter
from queue import PriorityQueue

from binteger import Bin

from divprop.subsets import (
    DynamicLowerSet,
    DynamicUpperSet,
    OptDynamicUpperSet,
    support_int_le,
    antisupport_int_le,
)

from divprop.milp import MILP
from divprop.inequalities.monopool import IneqInfo


log = logging.getLogger(__name__)


class LowerSetLearn:
    def __init__(self, n, oracle):
        self.n = int(n)
        self.oracle = oracle

        self.n_checks = 0

    def learn(self, check_trivial=True):
        """
        returns max-set of the learnt lower set
        the min-set of the complementary upperset can be retrieved from
        .bad attribute
        """
        if check_trivial:
            if not self.oracle(Bin(0, self.n)):
                self.good = ()
                self.bad = tuple(2**i for i in range(self.n))
                return tuple(Bin(v, self.n) for v in self.good)
            if self.oracle(Bin(2**self.n-1, self.n)):
                self.good = 2**self.n-1,
                self.bad = tuple(2**self.n-1-2**i for i in range(self.n))
                return tuple(Bin(v, self.n) for v in self.good)

        self.good = {0}
        self.bad = {2**self.n-1}

        assert self.n_checks == 0, "already ran?"
        self.n_checks = 0

        log.info(f"starting with n = {self.n}")

        # order is crucial, with internal dfs order too
        for i in range(self.n):
            self.cur_i = i
            self.dfs(1 << (self.n - 1 - i))

        log.info(
            "final stat:"
            f" checks {self.n_checks}"
            f" good max-set {len(self.good)}"
            f" bad min-set {len(self.bad)}"
        )

        return [Bin(v, self.n) for v in self.good]

    def dfs(self, v):
        for u in self.good:
            # v \preceq u
            if u & v == v:
                return
        # if inside bad space - then is bad
        for u in self.bad:
            # v \succeq u
            if u & v == u:
                return

        is_lower = self.oracle(Bin(v, self.n))

        self.n_checks += 1
        if self.n_checks == 1000 or self.n_checks % 1_000 == 0:
            wts = Counter(Bin(a).hw() for a in self.good)
            wts = " ".join(f"{wt}:{cnt}" for wt, cnt in sorted(wts.items()))
            wts2 = Counter(Bin(a).hw() for a in self.bad)
            wts2 = " ".join(f"{wt}:{cnt}" for wt, cnt in sorted(wts2.items()))
            log.info(
                f"stat: bit {self.cur_i+1}/{self.n}"
                f" checks {self.n_checks}"
                f" good max-set {len(self.good)}"
                f" bad min-set {len(self.bad)}"
                f" | good max-set weights {wts}"
                f" | bad max-set weights {wts2}"
            )

        if is_lower:
            self.add_good(v)
            # order is crucial!
            for j in reversed(range(self.n)):
                if (1 << j) > v:
                    vv = v | (1 << j)
                    self.dfs(vv)
        else:
            self.add_bad(v)

    def add_good(self, v):
        # note: we know that v is surely not redundant itself
        #                                unless (u <= v)
        self.good = {u for u in self.good if u & v != u}
        self.good.add(v)

    def add_bad(self, v):
        # note: we know that v is surely not redundant itself
        #                              unless (u >= v)
        self.bad = {u for u in self.bad if u & v != v}
        self.bad.add(v)


# TBD: base class interface, dense class for smallish N (<100? < 1000?)
class DenseLowerSetLearn:
    """
    fset = set of indexes of bits
    TBD: switch to SparseSet from C++?
    """
    def __init__(self, N):
        self.N = int(N)

        self.feasible = DynamicLowerSet((), n=self.N)
        self.infeasible = DynamicUpperSet((), n=self.N)

        self.solution = {}  # data for feasible

        # self.unknown_upper = DynamicUpperSet([0], n=self.N)
        self.unknown_upper = OptDynamicUpperSet([0], n=self.N)
        # self.unknown_lower = DynamicLowerSet([2**self.N-1], n=self.N)

        # self.unknown_queue = PriorityQueue()
        # self.unknown_set = set()

    def finished(self):
        # return len(self.unknown_upper.set) == len(self.unknown_lower.set) == 0
        return len(self.unknown_upper.set) == 0

    def _clean_solution(self):
        todel = [k for k in self.solution if k not in self.feasible.set]
        for k in todel:
            del self.solution[k]

    def log_info(self):
        log.info("stat:")
        for (name, s) in [
            ("unk upper", self.unknown_upper.set),
            # ("unk lower", self.unknown_lower.set),
            ("feasible", self.feasible.set),
            ("infeasible", self.infeasible.set),
        ]:
            freq = Counter(Bin(v).hw() for v in s)
            freqstr = " ".join(
                f"{sz}:{cnt}" for sz, cnt in sorted(freq.items())
            )
            log.info(f"   {name}: {len(s)}: {freqstr}")

    def encode_fset(self, fset):
        return sum(1 << (self.N - 1 - i) for i in fset)

    def is_already_feasible(self, v):
        # quick check
        if v in self.feasible.set:
            return True
        # is in feasible lowerset?
        for u in self.feasible.set:
            # v <= u
            if v & u == v:
                return True
        return False

    def is_already_infeasible(self, v):
        # quick check
        if v in self.infeasible.set:
            return True
        # is in infeasible upperset?
        for u in self.infeasible.set:
            # v >= u
            if v | u == v:
                return True
        return False

    def add_feasible(self, v, sol=None, check=True):
        if check and v in self.feasible:
            return
        # assert v not in self.infeasible
        self.feasible.add_lower_singleton(v, check=False)
        # self.unknown_lower.remove_lower_singleton_extremes(v)
        self.unknown_upper.remove_lower_singleton(v)
        if 1:
            # can be replaced by check in pulling loop
            for u in self.unknown_upper._added_last:
                if u in self.infeasible:
                    self.unknown_upper.set.remove(u)
                # if u not in self.unknown_lower:
                #     self.unknown_upper.set.remove(u)

        self.solution[v] = sol
        self._clean_solution()

    def add_infeasible(self, v, check=True):
        if check and v in self.infeasible:
            return
        # assert v not in self.feasible
        self.infeasible.add_upper_singleton(v, check=False)
        # self.unknown_upper.remove_upper_singleton_extremes(v)
        # self.unknown_lower.remove_upper_singleton(v)
        # for u in self.unknown_lower._added_last:
        #     if u not in self.unknown_upper:
        #         self.unknown_lower.set.remove(u)

    def learn_simple(self, oracle, sol_encoder=lambda v: v):
        # for
        while not self.finished():
            if oracle.n_calls % 1000 == 0:
                print("n_calls", oracle.n_calls)
                self.log_info()

            if 0:
                for lo in self.unknown_upper.set:
                    break
                for hi in self.unknown_lower.set:
                    if lo & hi == lo:  # lo <= hi
                        break
                inds = support_int_le(lo ^ hi, self.N)
                shuffle(inds)
                inds = inds[:len(inds)//4]
                fset = lo + sum(1 << i for i in inds)

            todel = []
            for fset in self.unknown_upper.set:
                if fset in self.infeasible:
                    todel.append(fset)
                else:
                    break

            if len(todel) > 1:
                print("todel", len(todel))
            for v in todel:
                self.unknown_upper.set.remove(v)

            r = randrange(10)
            indexes = list(antisupport_int_le(fset, self.N))
            shuffle(indexes)
            for i in indexes[:r]:
                fset2 = fset | i
                if fset2 in self.infeasible:
                    break
                fset = fset2

            #fset = sorted(self.unknown_upper.set, key=lambda v: Bin(v).hw())[choice((0, -1))]
            # fset = sorted(self.unknown_upper.set, key=lambda v: Bin(v).hw())[-1]
            # for fset in self.unknown_lower.set:
            #     break
            # todel = set()
            # for fset in self.unknown_upper.set:
            #     if fset not in self.unknown_lower:
            #         todel.add(fset)
            #     else:
            #         break
            # assert not todel


            # print("chose", Bin(fset, self.N).str, end="; ")
            # print("feas", len(self.feasible.set), end="; ")  # fset in self.feasible, end="; ")
            # print("infe", len(self.infeasible.set), end="; ")  # fset in self.infeasible, end="; ")
            # print("unku", len(self.unknown_upper.set), end="; ")  # fset in self.unknown_upper, end="; ")
            # print("unkl", len(self.unknown_lower.set), end="; ")  # fset in self.unknown_lower, end="; ")
            # print()
            # assert fset not in self.feasible
            # assert fset not in self.infeasible
            # assert fset in self.unknown_upper
            # assert fset in self.unknown_lower
            ineq = oracle.query(Bin(fset, self.N))
            # print("visit", Bin(fset, self.N).str, ":", ineq)
            if not ineq:
                self.add_infeasible(fset)
                # if Bin(fset).hw() == 2:
                #     for i in Bin(fset, self.N).support():
                #         print("".join(map(str, oracle.pool.lo[i])))
                #     print()
            else:
                self.add_feasible(
                    fset, sol=sol_encoder(ineq)
                )
                # if Bin(fset).hw() >= 9:
                #     print("ineq", ineq)
                #     for i in Bin(fset, self.N).support():
                #         print("".join(map(str, oracle.pool.lo[i])))
                #     print()
            # print("result")
            # print("feas", self.feasible.to_DenseSet().to_Bins())
            # print("infe", self.infeasible.to_DenseSet().to_Bins())
            # print("unku", self.unknown_upper.to_DenseSet().to_Bins())
            # print("unkl", self.unknown_lower.to_DenseSet().to_Bins())
            # print()
        return {Bin(v, self.N) for v in self.feasible.set}


# # TBD: base class interface, dense class for smallish N (<100? < 1000?)
# class SparseLowerSetLearn:
#     def __init__(self, N):
#         self.N = int(N)

#         # TBD: optimization by hw?
#         # { set of indexes of covered bad points (hi)
#         #   :
#         #   ineq, source, state }
#         # state = int index of last unchecked bit up?
#         self.feasible = {}

#         self.unknown_queue = PriorityQueue()
#         self.unknown_set = set()

#         # { set of indexes of infeasible to cover bad points }
#         self.infeasible = set()

#         self._order_sbox = sample(range(self.N), self.N)

#     def queue_fset(self, fset):
#         if fset in self.feasible:
#             return
#         if fset in self.infeasible:
#             return
#         if fset in self.unknown_set:
#             return
#         self.unknown_set.add(fset)
#         self.unknown_queue.put((-len(fset), fset))

#     def log_info(self):
#         log.info(
#             "stat:"
#             f" good max-set {len(self.feasible)}"
#             f" ({len(self.feasible) - len(self.feasible_open)} final)"
#             f" bad min-set {len(self.infeasible)}"
#         )
#         freq = Counter(map(len, self.feasible))
#         log.info(
#             "freq: "
#             + " ".join(f"{sz}:{cnt}" for sz, cnt in sorted(freq.items()))
#         )

#     def encode_fset(self, fset):
#         return frozenset(map(int, fset))

#     def is_already_feasible(self, fset):
#         # quick check
#         if fset in self.feasible:
#             return True
#         # is in feasible lowerset?
#         for fset2 in self.feasible:
#             if fset <= fset2:
#                 return True
#         return False

#     def is_already_infeasible(self, fset):
#         # quick check
#         if fset in self.infeasible:
#             return True
#         # is in infeasible upperset?
#         for fset2 in self.infeasible:
#             if fset2 <= fset:
#                 return True
#         return False

#     def add_feasible(self, fset, ineq, source, check=True):
#         if check and self.is_already_feasible(fset):
#             return
#         # remove existing redundant
#         self.feasible = {
#             fset2: info2
#             for fset2, info2 in self.feasible.items()
#             if not (fset2 <= fset)
#         }
#         self.feasible[fset] = IneqInfo(
#             ineq,
#             source="basic",
#         )

#         for fset2 in self.neibs_up(fset):
#             self.queue_fset(fset)

#     def add_infeasible(self, fset, check=True):
#         if check and self.is_already_infeasible(fset):
#             return
#         # remove existing redundant
#         self.infeasible = {
#             fset2
#             for fset2 in self.infeasible
#             if not (fset2 >= fset)
#         }
#         self.infeasible.add(fset)

#     # def get_next_unknown_neighbour(self, fset):
#     #     assert fset in self.feasible_open
#     #     h, i = self.feasible[fset].state
#     #     good = 0
#     #     while i < self.N:
#     #         ii = self._get_real_index(h, i)
#     #         i += 1
#     #         if ii not in fset:
#     #             fset2 = fset | {ii}
#     #             if fset2 in self.infeasible:
#     #                 continue
#     #             if self.is_already_feasible(fset2):
#     #                 continue
#     #             if self.is_already_infeasible(fset2):
#     #                 continue
#     #             good = 1
#     #             break

#     #     if i >= self.N:
#     #         i = None
#     #         self.feasible_open.remove(fset)
#     #         self.feasible[fset] = self.feasible[fset]._replace(state=None)
#     #     else:
#     #         self.feasible[fset] = self.feasible[fset]._replace(state=(h, i))

#     #     if good:
#     #         return fset2

#     def neibs_up(self, fset):
#         for i in range(self.N):
#             if i not in fset:
#                 yield fset | {i}


class SupportLearner:
    def __init__(self, level=2):
        self.level = int(level)

    def learn_system(self, system, oracle):
        self.log = logging.getLogger(f"{__name__}:{type(self).__name__}")
        N = system.N

        start = getattr(system, "_support_learned", 1) + 1

        if start > self.level:
            self.log.info(
                f"support-{start} already learned "
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


class CliqueMountainHills:
    def __init__(
        self,
        base_level=2,
        max_mountains=0,
        min_height=10,
        max_repeated_streak=5,
        max_exclusion_size=7,
        n_random=10000,
        reuse_known=True,
        bad_learn_hard_limit_random=25,
        bad_learn_hard_limit_milp=200,
        solver="scip",
    ):
        assert base_level >= 2
        self.base_level = int(base_level)
        self.max_mountains = int(max_mountains)
        self.min_height = int(min_height)
        self.max_repeated_streak = int(max_repeated_streak)
        self.max_exclusion_size = int(max_exclusion_size)
        self.reuse_known = reuse_known
        self.n_random = int(n_random)

        self.bad_learn_hard_limit_random = int(bad_learn_hard_limit_random)
        self.bad_learn_hard_limit_milp = int(bad_learn_hard_limit_milp)

        self._options = self.__dict__.copy()

        self.solver = solver
        self.log = logging.getLogger(f"{__name__}:{type(self).__name__}")

    def learn_system(self, system, oracle):
        self.N = system.N
        self.sys = system
        self.oracle = oracle

        n_calls0 = self.oracle.n_calls

        self.milp = None

        self.log.info(f"starting, options: {self._options}")

        # =================================

        self.sol_encoder = lambda ineq: \
            IneqInfo(ineq, f"Support-{self.base_level}")

        SupportLearner(level=self.base_level).learn_system(
            system=system,
            oracle=oracle,
        )

        # =================================

        self.sol_encoder = lambda ineq: \
            IneqInfo(ineq, "Clique-Random")

        self.bad_learn_hard_limit = self.bad_learn_hard_limit_random
        self.generate_cliques_random()

        self.log.info("random clique statistics:")
        self.log.info(
            f"    {self.n_cliques} cliques enumerated: "
            f"{self.n_good} good, {self.n_bad} bad (sub)cliques"
        )
        self.log.info(f"    {self.oracle.n_calls - n_calls0} oracle calls")
        self.sys.log_info()

        # =================================

        self.sys.feasible.do_MaxSet()
        self.sys.infeasible.do_MinSet()
        self.sys.clean_solution()

        self.log.info("after MaxSet/MinSet")
        self.sys.log_info()

        # =================================

        self.sol_encoder = lambda ineq: \
            IneqInfo(ineq, "Clique-MILP")

        self.bad_learn_hard_limit = self.bad_learn_hard_limit_milp
        self.generate_cliques_milp()

        self.log.info("final statistics:")
        self.log.info(
            f"    {self.n_cliques} cliques enumerated: "
            f"{self.n_good} good, {self.n_bad} bad (sub)cliques"
        )
        self.log.info(f"    {self.oracle.n_calls - n_calls0} oracle calls")
        self.sys.log_info()

        # =================================

        self.sys.feasible.do_MaxSet()
        self.sys.infeasible.do_MinSet()
        self.sys.clean_solution()

        self.log.info("after MaxSet/MinSet")
        self.sys.log_info()

        self.log.info(f"recall options: {self._options}")

    def exclude_subcliques(self, fset):
        self.milp.add_constraint(
            sum(self.xs[i] for i in range(self.N) if i not in fset) >= 1
        )

    def exclude_supercliques(self, fset):
        self.milp.add_constraint(
            sum(self.xs[i] for i in fset) <= len(fset) - 1
        )

    def generate_cliques_random(self):
        self.n_cliques = 0
        self.n_good = 0
        self.n_bad = 0

        self.log.info(f"generating {self.n_random} cliques")

        for itr in range(self.n_random):
            order = list(range(self.N))
            shuffle(order)

            fset = frozenset([order.pop()])
            for i in order:
                if 1:
                    # check all edges from support
                    good = 1
                    for l in range(1, self.base_level):
                        for js in combinations(fset, l):
                            edge = frozenset(js + (i,))
                            if edge in self.sys.infeasible.cache:
                                good = 0
                                break
                        if not good:
                            break
                    if not good:
                        continue
                else:
                    # check only 2-edges
                    good = 1
                    for l in range(2, self.base_level+1):
                        for j in fset:
                            if frozenset((i, j)) in self.sys.infeasible.cache:
                                good = 0
                                break
                        if not good:
                            break
                    if not good:
                        continue

                fset2 = fset | {i}
                if fset2 in self.sys.infeasible.cache:
                    continue
                fset = fset2

            assert fset not in self.sys.infeasible.cache

            if fset in self.sys.feasible.cache:
                continue

            self.n_cliques += 1

            self.log.info(
                f"random clique #{self.n_cliques} (tries {itr+1}/{self.n_random}), size {len(fset)}: "
                f"{tuple(fset)} (good: {self.n_good}, bads: {self.n_bad})"
            )
            assert fset
            ineq = self.oracle.query(Bin(fset, self.N))

            self.log.info(f"ineq: {ineq}")
            if ineq:
                self.process_good_clique(fset, ineq)
            else:
                self.process_bad_clique(fset)

    def generate_cliques_milp(self):
        self.milp = MILP.maximization(solver=self.solver)
        try:
            # is buggy...
            # self.milp.set_reopt()
            pass
        except AttributeError:
            pass

        self.xs = [self.milp.var_binary("x%d" % i) for i in range(self.N)]
        self.xsum = self.milp.var_int("xsum", lb=self.base_level+1, ub=self.N)
        self.milp.add_constraint(sum(self.xs) == self.xsum)
        self.milp.set_objective(self.xsum)

        # exclude super-cliques of known infeasible ones
        # all known or support of base-level?
        if self.reuse_known:
            log.info(
                f"reusing {len(self.sys.infeasible)} infeasible and "
                f"{len(self.sys.feasible)} feasible cliques"
            )
            for fset in self.sys.infeasible:
                if len(fset) <= self.max_exclusion_size:
                    self.exclude_supercliques(fset)
            for fset in self.sys.feasible:
                self.exclude_subcliques(fset)
        else:
            log.info(f"using only infeasible support-{self.base_level}")
            for l in range(2, self.base_level+1):
                for fset in self.sys.infeasible.iter_wt(l):
                    self.exclude_supercliques(fset)

        self.log.info(
            "starting max-clique search, "
            f"max_mountains: {self.max_mountains} "
            f"(min height {self.min_height})"
        )

        self.n_cliques = 0
        self.n_bad = 0
        self.n_good = 0
        while True:
            size = self.milp.optimize(solution_limit=100, only_best=True)
            if size is None:
                self.log.info(f"no new cliques, milp.err: {self.milp.err}")
                break

            assert isinstance(size, int), size
            assert size > self.base_level + 0.5

            assert self.milp.solutions
            for sol in self.milp.solutions:
                fset = self.sys.encode_bad_subset(
                    i for i, x in enumerate(self.xs) if sol[x] > 0.5
                )
                self.log.info(
                    f"clique #{self.n_cliques}, size {size}: "
                    f"{tuple(fset)} (good: {self.n_good}, bads: {self.n_bad})"
                )
                assert fset

                self.n_cliques += 1

                ineq = self.oracle.query(Bin(fset, self.N))

                self.log.info(f"ineq: {ineq}")
                if ineq:
                    self.process_good_clique(fset, ineq)
                else:
                    self.process_bad_clique(fset)

                self.milp.add_constraint(self.xsum <= size)

    def process_good_clique(self, fset, ineq):
        self.sys.add_feasible(fset, sol=self.sol_encoder(ineq))

        for i in fset:
            self.log.debug(f"{i:4d} {Bin(self.oracle.pool.i2bad[i]).str}")
        self.log.debug("")

        self.n_good += 1

        if self.milp:
            self.exclude_subcliques(fset)

            # take mountains ?
            # then hunt for the hills
            if self.n_good <= self.max_mountains and len(fset) >= self.min_height:
                self.log.info(
                    f"mountain #{self.n_good}/{self.max_mountains}, "
                    f"height {len(fset)}>={self.min_height}"
                )
                # exclude this variables (assume this points are already covered)
                for i in fset:
                    self.milp.set_ub(self.xs[i], 0)

    def process_bad_clique(self, fset):
        # exclude this clique (& super-cliques)
        orig = fset
        repeated_streak = 0
        best_exclude = float("+inf"), None
        excluded_something = 0
        for itr in range(self.bad_learn_hard_limit):  # hard limit
            fset = orig
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
                best_exclude = min(best_exclude, (len(fset), fset))

                self.n_bad += 1
                if len(fset) <= self.max_exclusion_size:
                    self.log.info(f"exclude wt={len(fset)}: {fset}")

                self.sys.add_infeasible(fset)

                if self.milp and len(fset) <= self.max_exclusion_size:
                    # exclude this clique (&super-cliques since it's reduced)
                    self.exclude_supercliques(fset)
                    excluded_something = 1

                repeated_streak = 0
                if fset == orig:
                    break
            else:
                repeated_streak += 1
                if repeated_streak >= self.max_repeated_streak:
                    break

        assert best_exclude[1]
        if self.milp and not excluded_something:
            self.log.info(f"force exclude wt={len(fset)}: {fset}")
            fset = best_exclude[1]
            self.exclude_supercliques(fset)
