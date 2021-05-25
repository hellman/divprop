from collections import Counter
from queue import PriorityQueue

from binteger import Bin

from subsets import DenseSet
from subsets.WeightedSet import GrowingUpperFrozen

from divprop.divprop import (
    DivCore_StrongComposition,
    DivCore_StrongComposition8,
    DivCore_StrongComposition16,
    DivCore_StrongComposition32,
    DivCore_StrongComposition64,
    Sbox,
    Sbox32,
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
        self._set = set(map(int, data))
        self.n = int(n)
        self.m = int(m)
        self.mask_u = mask(n) << m
        self.mask_v = mask(m)

    def to_dense(self):
        # list because swig does not map set straightforwardly.. need a typemap
        d = DenseSet(self.n + self.m)
        for x in self._set:
            d.set(x)
        return d
    to_DenseSet = to_dense

    def to_Bins(self):
        return {Bin(v, self.n+self.m) for v in self._set}

    def __iter__(self):
        return iter(self._set)

    @classmethod
    def from_Sbox(cls, sbox: Sbox, method="dense", debug=False):
        assert isinstance(sbox, Sbox.classes)
        method = getattr(cls, "from_Sbox_" + method)
        if not method:
            raise ValueError(f"Unknown method DenseDivCore.from_Sbox_{method}")
        return method(sbox, debug)
    from_sbox = from_Sbox

    @classmethod
    def from_Sbox_dense(cls, sbox: Sbox, debug=False):
        ret = sbox.graph_dense()

        if debug:
            cls.log.info(f"  graph {ret}")

        ret.do_Mobius()

        if debug:
            cls.log.info(f"    anf {ret}")

        ret.do_MaxSet()

        if debug:
            cls.log.info(f"anf-max {ret}")

        ret.do_Not()

        if debug:
            cls.log.info(f"divcore {ret}")

        return cls(ret, sbox.n, sbox.m)

    @classmethod
    def from_Sbox_peekanfs(cls, sbox: Sbox, debug=False):
        assert sbox.n == sbox.m, "only bijections supported yet"
        divcore = SboxPeekANFs(sbox).compute(debug=debug)
        return cls(divcore, n=sbox.n, m=sbox.m)

    def to_propagation_map(self):
        d = self.get_Minimal()
        ret = [list() for _ in range(2**self.n)]
        for uv in d.to_Bins():
            u, v = uv.split(ns=(self.n, self.m))
            ret[(~u).int].append(v.int)
        return ret

    def get_Invalid(self) -> DenseSet:
        """Set I_S from the paper"""
        ret = self.to_dense()
        ret.do_ComplementU2L()
        return ret

    def get_Minimal(self) -> DenseSet:
        """Set M_S from the paper. = MinDPPT up to negating (u)."""
        ret = self.to_dense()
        ret.do_UpperSet(self.mask_u)
        ret.do_MinSet(self.mask_v)
        return ret

    def get_Minimal_Bounds(self) -> (DenseSet, DenseSet):
        """Set M_S from the paper, in the form of its (MinSet,Maxset)"""
        lo = self.to_dense()
        hi = self.get_Minimal()
        hi.do_MaxSet()
        return lo, hi

    def get_Redundant(self) -> DenseSet:
        ret = self.to_dense()
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
        ret = self.to_dense()
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
            ret = self.to_dense()
            ret.do_UpperSet_Up1(True, self.mask_v)  # is_minset=true
            ret.do_MinSet()
        else:
            # ret = self.MinDPPT()
            # ret.do_Not(self.mask_u)
            ret = self.to_dense()
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
        ret = self.to_dense()
        ret.do_UpperSet()
        ret.do_Not(self.mask_u)
        return ret

    def MinDPPT(self) -> DenseSet:
        """
        DenseSet of all valid reduced transitions.
        """
        ret = self.to_dense()
        ret.do_UpperSet(self.mask_u)
        ret.do_MinSet(self.mask_v)
        ret.do_Not(self.mask_u)
        return ret

    def __eq__(self, other):
        assert isinstance(other, DivCore)
        assert self.n == other.n
        assert self.m == other.m
        return self._set == other._set


class SboxPeekANFs:
    log = logging.getLogger(f"{__name__}:SboxPeekANFs")

    def __init__(self, sbox: Sbox, isbox: Sbox = None):
        assert isinstance(sbox, Sbox.classes)
        self.n = int(sbox.n)
        self.sbox = sbox
        self.isbox = ~sbox if isbox is None else isbox

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
            self.log.debug(
                f"run #{itr} fset {fset} inv? {inverse}, "
                f"queue size {q.qsize()}"
            )
            stat[(inverse, len(fset))] += 1

            mask = Bin(fset, 2*n).int
            if inverse:
                mask >>= n
            res = self.run_mask(mask, inverse=inverse)

            added = {frozenset(Bin(uv, 2*n).support) for uv in res}
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
            f"computed divcore n={n:02d} in {itr} bit-ANF calls, "
            f"stat {statstr}, size {len(divcore)}"
        )
        return set(divcore.to_Bins())

    def get_product(self, mask, inverse):
        sbox = self.isbox if inverse else self.sbox
        return sbox.coordinate_product(mask)

    def run_mask(self, mask, inverse=False):
        assert 0 <= mask < 1 << self.n
        func = self.get_product(mask, inverse)
        func.do_Mobius()
        func.do_MaxSet()
        func.do_Not()
        if inverse:
            return {(mask << self.n) | u for u in func}
        else:
            return {(u << self.n) | mask for u in func}
