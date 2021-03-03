import logging

from collections import Counter

from binteger import Bin


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
        if self.n_checks % 250_000 == 0:
            wts = Counter(Bin(a).hw() for a in self.good)
            wts = " ".join(f"{wt}:{cnt}" for wt, cnt in sorted(wts.items()))
            log.verbose(
                f"stat: bit {self.cur_i+1}/{self.n}"
                f" checks {self.n_checks}"
                f" good max-set {len(self.good)}"
                f" bad min-set {len(self.bad)}"
                f" | good max-set weights {wts}"
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
