import logging
from collections import Counter

from binteger import Bin

from .base import satisfy, MIPSolverException
from .random_group_cut import RandomGroupCut


log = logging.getLogger(__name__)


class GemCut(RandomGroupCut):
    def generate(self):
        self.N = len(self.lo)

        self.good = {0: None}
        self.bad = {(1 << self.N) - 1}

        # from the end we can get longer chains (large lower sets)
        log.info(f"starting with N = {self.N}")

        self.n_checks = 0
        # order is crucial, with internal dfs order too
        for i in reversed(range(self.N)):
            self.dfs(1 << i)

        log.info(
            "final stat:"
            f" checks {self.n_checks}"
            f" good max-set {len(self.good)}"
            f" bad min-set {len(self.bad)}"
        )
        tops = {}
        sorter = lambda it: Bin(it[0], self.N).hw()
        for v, sol in sorted(self.good.items(), key=sorter):
            v = Bin(v, self.N)
            covered = [self.lo[i] for i in v.support()]
            # print("top", v.str, "%3d" % v.hw(), v.support(), "|", sol)

            assert all(satisfy(q, sol) for q in self.hi)
            assert all(not satisfy(q, sol) for q in covered)

            if self.inverted:
                func, value_good = sol[:-1], -sol[-1]
                # x1a1 + x2a2 + x3a3 >= t
                # =>
                # x1(1-a1) + x2(1-a2) + x3(1-a3) >= t
                # -x1a1 -x2a2 -x3a3 >= t-sum(x)
                value = value_good - sum(func)
                sol = tuple(-x for x in func) + (-value,)
                covered = [tuple(1 - a for a in q) for q in covered]

                assert all(satisfy(q, sol) for q in self.orig_lo)
                for q in covered:
                    assert q in self.orig_hi
            else:
                assert all(satisfy(q, sol) for q in self.orig_hi)
                for q in covered:
                    assert q in self.orig_lo

            assert all(not satisfy(q, sol) for q in covered)
            tops[sol] = covered
        return tops

    def dfs(self, v):
        # dbg = 0
        # if dbg: print("visit", Bin(v, self.N).str, v)
        # if inside good space - then is good
        for u in self.good:
            # v \preceq u
            if u & v == v:
                # if dbg: print("is in good", Bin(u, self.N).str)
                return
        # if inside bad space - then is bad
        for u in self.bad:
            # v \succeq u
            if u & v == u:
                # if dbg: print("is in bad", Bin(u, self.N).str)
                return

        grp = Bin(v, self.N).support()
        sol = self.check_group(grp)
        self.n_checks += 1
        # if dbg: print("check is", sol)
        # if dbg: print()
        if self.n_checks % 10_000 == 0:
            wts = Counter(Bin(a).hw() for a in self.good)
            wts = " ".join(f"{wt}:{cnt}" for wt, cnt in sorted(wts.items()))
            log.info(
                "stat:"
                f" checks {self.n_checks}"
                f" good max-set {len(self.good)}"
                f" bad min-set {len(self.bad)}"
                f" | good max-set weights {wts}"
            )
        if sol:
            self.add_good(v, sol)
            # order is crucial!
            for j in reversed(range(self.N)):
                if (1 << j) > v:
                    vv = v | (1 << j)
                    self.dfs(vv)
        else:
            self.add_bad(v)

    def add_good(self, v, sol):
        # note: we know that v is surely not redundant itself
        for u in list(self.good):
            # u \preceq v
            if u & v == u:
                del self.good[u]
        self.good[v] = sol

    def add_bad(self, v):
        # note: we know that v is surely not redundant itself
        #                                  u \succeq v
        self.bad = {u for u in self.bad if u & v != v}
        self.bad.add(v)

    def check_group(self, bads):
        LP = self.model.__copy__()

        for i in bads:
            q = self.lo[i]
            LP.add_constraint(self.cs_per_lo[q])

        try:
            LP.solve()
        except MIPSolverException:
            return False

        val_xs = tuple(LP.get_values(x) for x in self.xs)
        if all(abs(v - round(v)) < 0.00001 for v in val_xs):
            # is integral
            val_xs = tuple(int(v + 0.5) for v in val_xs)
            val_c = int(LP.get_values(self.c) + 0.5)
        else:
            # keep real
            val_c = LP.get_values(self.c) - 0.5
        ineq = val_xs + (-val_c,)

        assert all(satisfy(p, ineq) for p in self.hi)
        assert all(not satisfy(self.lo[i], ineq) for i in bads)
        return ineq
