import logging

from random import choice, shuffle, randrange

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
            if oracle.n_calls % 100 == 0:
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

            r = randrange(15)
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
            else:
                self.add_feasible(
                    fset, sol=sol_encoder(ineq)
                )
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


class NewDenseLowerSetLearn:
    """
    fset = set of indexes of bits
    TBD: switch to SparseSet from C++?
    """
    def __init__(self, N):
        self.N = int(N)

        self.feasible = DynamicLowerSet((), n=self.N)
        self.infeasible = DynamicUpperSet((), n=self.N)
        self.infeasible_cache = set()
        self.feasible_cache = set()

        self.solution = {}  # data for feasible

    def finished(self):
        xxx

    def _clean_solution(self):
        todel = [k for k in self.solution if k not in self.feasible.set]
        for k in todel:
            del self.solution[k]

    def log_info(self):
        log.info("stat:")
        for (name, s) in [
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
        while not self.finished():
            if oracle.n_calls % 100 == 0:
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

            r = randrange(15)
            indexes = list(antisupport_int_le(fset, self.N))
            shuffle(indexes)
            for i in indexes[:r]:
                fset2 = fset | i
                if fset2 in self.infeasible:
                    break
                fset = fset2

            ineq = oracle.query(Bin(fset, self.N))
            # print("visit", Bin(fset, self.N).str, ":", ineq)
            if not ineq:
                self.add_infeasible(fset)
            else:
                self.add_feasible(
                    fset, sol=sol_encoder(ineq)
                )

        return {Bin(v, self.N) for v in self.feasible.set}
