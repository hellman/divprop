from itertools import combinations
from collections import Counter
from queue import PriorityQueue

from binteger import Bin

from divprop.subsets import (
    DenseSet,
    DivCore_StrongComposition,
    DivCore_StrongComposition8,
    DivCore_StrongComposition16,
    DivCore_StrongComposition32,
    DivCore_StrongComposition64,
    Sbox,
    GrowingUpperFrozen,
)

import divprop.logs as logging


def mask(m):
    return (1 << m) - 1


class DivCore:
    """
    Division Core of the S-Box = reduced DPPT of its graph (as a dense set).

    Notation: vectors (u, v), with bit-length (n, m).
    """
    log = logging.getLogger(f"{__name__}:DivCore")

    def __init__(self, data, n, m):
        self.data = set(map(int, data))
        self.n = int(n)
        self.m = int(m)
        self.mask_u = mask(n) << m
        self.mask_v = mask(m)

    def get_dense(self):
        # list because swig does not map set straightforwardly.. need a typemap
        return DenseSet(list(self.data), self.n + self.m)

    def to_Bins(self):
        return {Bin(v, self.n+self.m) for v in self.data}

    @classmethod
    def from_sbox(cls, sbox: Sbox, n: int = None, m: int = None,
                  method="dense", debug=False):
        if n is None or m is None:
            assert isinstance(sbox, Sbox)
            n = sbox.n
            m = sbox.m
        if not isinstance(sbox, Sbox):
            sbox = Sbox(sbox, n, m)
        method = getattr(cls, "from_sbox_" + method)
        if not method:
            raise ValueError(f"Unknown method DenseDivCore.from_sbox:{method}")
        return method(sbox, n, m, debug)

    @classmethod
    def from_sbox_dense(cls, sbox: Sbox, n: int, m: int, debug=False):
        n = int(n)
        m = int(m)

        graph = sbox.graph_indicator()

        if debug:
            cls.log.info(f"  graph {graph}")

        graph.do_Mobius()

        if debug:
            cls.log.info(f"    anf {graph}")

        graph.do_MaxSet()

        if debug:
            cls.log.info(f"anf-max {graph}")

        graph.do_Not()

        if debug:
            cls.log.info(f"divcore {graph}")

        return cls(graph, n, m)

    @classmethod
    def from_sbox_peekanfs(cls, sbox: Sbox, n: int, m: int, debug=False):
        n = int(n)
        m = int(m)
        assert n == m, "only bijections supported yet"
        if not isinstance(sbox, Sbox):
            sbox = Sbox(sbox, n, m)
        divcore = SboxPeekANFs(sbox).compute(debug=debug)
        return cls(divcore, n=n, m=m)

    def get_Invalid(self) -> DenseSet:
        """Set I_S from the paper"""
        ret = self.get_dense()
        ret.do_ComplementU2L()
        return ret

    def get_Minimal(self) -> DenseSet:
        """Set M_S from the paper. = MinDPPT up to negating (u)."""
        ret = self.get_dense()
        ret.do_UpperSet(self.mask_u)
        ret.do_MinSet(self.mask_v)
        return ret

    def get_Minimal_Bounds(self) -> (DenseSet, DenseSet):
        """Set M_S from the paper, in the form of its (MinSet,Maxset)"""
        lo = self.get_dense()
        hi = self.get_Minimal()
        hi.do_MaxSet()
        return lo, hi

    def get_Redundant(self) -> DenseSet:
        ret = self.get_dense()
        ret.do_UpperSet_Up1(True, self.mask_v)  # is_minset=true
        ret.do_MinSet()
        return ret

    def get_RedundantAlternative(self) -> DenseSet:
        ret = self.get_Minimal()
        ret.do_MaxSet()
        ret.do_ComplementL2U()
        return ret

    def LB(self) -> DenseSet:
        """
        Outer lower bound for MinDPPT,
        in the form of MaxSet of invalid vectors
        (ones that are a bit lower than vectors from divcore)
        """
        ret = self.get_dense()
        ret.do_ComplementU2L()
        return ret

    def UB(self, method="redundant") -> DenseSet:
        """
        (note: MinDPPT here means extra "not u")

        Outer upper bound for MinDPPT,

        method="redundant"
        in the form of MinSet of redundant vectors
        (ones that are a bit upper in (v) than reduced vectors from divcore)

        method="complement"
        in the form of the complementary MinSet of the MaxSet of MinDPPT

        """
        if method == "redundant":
            ret = self.get_dense()
            ret.do_UpperSet_Up1(True, self.mask_v)  # is_minset=true
            ret.do_MinSet()
        else:
            # ret = self.MinDPPT()
            # ret.do_Not(self.mask_u)
            ret = self.get_dense()
            ret.do_UpperSet(self.mask_u)
            ret.do_MinSet(self.mask_v)

            ret.do_LowerSet()
            ret.do_Complement()
            ret.do_MinSet()
        return ret

    def FullDPPT(self) -> DenseSet:
        """
        DenseSet of all valid transitions, including redundant ones.
        """
        ret = self.get_dense()
        ret.do_UpperSet()
        ret.do_Not(self.mask_u)
        return ret

    def MinDPPT(self) -> DenseSet:
        """
        DenseSet of all valid reduced transitions.
        """
        ret = self.get_dense()
        ret.do_UpperSet(self.mask_u)
        ret.do_MinSet(self.mask_v)
        ret.do_Not(self.mask_u)
        return ret

    def __eq__(self, other):
        assert isinstance(other, DivCore)
        assert self.n == other.n
        assert self.m == other.m
        return self.data == other.data


class SboxPeekANFs:
    log = logging.getLogger(f"{__name__}:SboxPeekANFs")

    def __init__(self, sbox: Sbox):
        assert isinstance(sbox, Sbox)
        self.n = int(sbox.n)
        self.sbox = sbox
        self.isbox = ~sbox

    def compute(self, debug=False):
        n = self.n

        divcore = GrowingUpperFrozen(n=2*n, disable_cache=True)
        divcore.add(frozenset(range(n)))
        divcore.add(frozenset(range(n, 2*n)))

        q = PriorityQueue()
        for i in range(n):
            tocheck = {frozenset((j, n+i)) for j in range(n)}
            q.put((1, False, frozenset({n+i}), tocheck))
            tocheck = {frozenset((n+j, i)) for j in range(n)}
            q.put((1, True, frozenset({i}), tocheck))

        stat = Counter()
        itr = 0
        while q.qsize():
            _, inverse, fset, tocheck = q.get()

            if all(fset2 in divcore for fset2 in tocheck):
                continue

            itr += 1
            self.log.debug(f"run #{itr} fset {fset} inv? {inverse}")
            stat[(inverse, len(fset))] += 1

            mask = Bin(fset, 2*n).int
            if inverse:
                mask >>= n
            res = self.run_mask(mask, inverse=inverse)

            added = {frozenset(Bin(uv, 2*n).support()) for uv in res}
            divcore.update(added)

            if itr % 10 == 0:
                divcore.do_MinSet()

            tocheck -= added
            if not tocheck:
                continue

            rng = range(max(fset)+1, n if inverse else 2*n)
            for i in rng:
                fset2 = fset | {i}
                tocheck2 = {v | {i} for v in tocheck}
                q.put((len(fset)+1, inverse, fset2, tocheck2))

        divcore.do_MinSet()
        statstr = " ".join(
            f"{l}:{cnt}" for (inverse, l), cnt in sorted(stat.items())
        )
        self.log.info(
            f"computed divcore n={n} in {itr} bit-ANF calls, "
            f"stat {statstr}, size {len(divcore)}"
        )
        return set(divcore.to_Bins())

    def run_mask(self, mask, inverse=False):
        assert 0 <= mask < 1 << self.n
        sbox = self.isbox if inverse else self.sbox
        func = sbox.coordinate_product(mask)
        func.do_Mobius()
        func.do_MaxSet()
        func.do_Not()
        if inverse:
            return {(mask << self.n) | u for u in func}
        else:
            return {(u << self.n) | mask for u in func}


if __name__ == '__main__':
    logging.setup(level="DEBUG")

    import sys
    n = m = int(sys.argv[1])
    sbox = list(range(2**n))
    from random import shuffle
    shuffle(sbox)
    # ans = sorted(DenseDivCore.from_sbox(sbox, n, m).data.to_Bins())

    sbox = Sbox(sbox, n, m)
    pa = SboxPeekANFs(sbox)
    res = sorted(pa.compute())
    # print(*res)
    # assert res == ans
    print("OK")
