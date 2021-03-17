from array import array

from divprop.subsets import (
    DenseSet,
    DivCore_StrongComposition,
    DivCore_StrongComposition8,
    DivCore_StrongComposition16,
    DivCore_StrongComposition32,
    DivCore_StrongComposition64,
    Sbox2GraphIndicator,
    Sbox2Coordinates,
    Sbox,
)

import divprop.logs as logging


def mask(m):
    return (1 << m) - 1


class DenseDivCore:
    """
    Division Core of the S-Box = reduced DPPT of its graph (as a dense set).

    Notation: vectors (u, v), with bit-length (n, m).

    Wrapper for DenseSet, stored in :attr:`data`.
    """
    log = logging.getLogger(f"{__name__:DenseDivCore}")

    def __init__(self, data, n, m):
        assert isinstance(data, DenseSet)
        assert data.n == n + m
        self.data = data
        self.n = int(n)
        self.m = int(m)
        self.mask_u = mask(n) << m
        self.mask_v = mask(m)

    @classmethod
    def from_sbox(cls, sbox, n, m, debug=False, method="simple"):
        method = getattr(cls, "from_sbox_" + method)
        if not method:
            raise ValueError(f"Unknown method DenseDivCore.from_sbox:{method}")
        return method(sbox, n, m, debug)

    @classmethod
    def from_sbox_simple(cls, sbox, n, m, debug=False):
        graph = Sbox2GraphIndicator(sbox, n, m)

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

    def get_Invalid(self) -> DenseSet:
        """Set I_S from the paper"""
        return self.data.ComplementU2L()

    def get_Minimal(self) -> DenseSet:
        """Set M_S from the paper. = MinDPPT up to negating (u)."""
        ret = self.data.copy()
        ret.do_UpperSet(self.mask_u)
        ret.do_MinSet(self.mask_v)
        return ret

    def get_Minimal_Bounds(self) -> (DenseSet, DenseSet):
        """Set M_S from the paper, in the form of its (MinSet,Maxset)"""
        ret = self.get_Minimal()
        ret.do_MaxSet()
        return self.data.copy(), ret

    def get_Redundant(self) -> DenseSet:
        ret = self.data.copy()
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
        ret = self.data.copy()
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
            ret = self.data.copy()
            ret.do_UpperSet_Up1(True, self.mask_v)  # is_minset=true
            ret.do_MinSet()
        else:
            # ret = self.MinDPPT()
            # ret.do_Not(self.mask_u)
            ret = self.data.copy()
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
        ret = self.data.copy()
        ret.do_UpperSet()
        ret.do_Not(self.mask_u)
        return ret

    def MinDPPT(self) -> DenseSet:
        """
        DenseSet of all valid reduced transitions.
        """
        ret = self.data.copy()
        ret.do_UpperSet(self.mask_u)
        ret.do_MinSet(self.mask_v)
        ret.do_Not(self.mask_u)
        return ret


class SparseDivCore:
    log = logging.getLogger(f"{__name__:SparseDivCore}")

    def __init__(self, data, n, m):
        self.data = set(data)
        self.n = int(n)
        self.m = int(m)
        self.mask_u = mask(n) << m
        self.mask_v = mask(m)

    @classmethod
    def from_sbox(cls, sbox, n, m, debug=False, method="peekanfs"):
        method = getattr(cls, "from_sbox_" + method)
        if not method:
            raise ValueError(f"Unknown method DenseDivCore.from_sbox:{method}")
        return method(sbox, n, m, debug)

    @classmethod
    def from_sbox_peekanfs(cls, sbox: Sbox, n, m, debug=False):
        assert n == m, "only bijections supported yet"
        return SboxPeekANFs(sbox, n).compute()


class SboxPeekANFs:
    def __init__(self, sbox: Sbox, n: int):
        self.n = int(n)
        self.sbox = sbox
        self.isbox = ~sbox
        for x, y in enumerate(sbox):
            self.isbox[y] = x

    def compute(self):
        pass

    def run_bit(self, mask, inverse=False):
        sbox = self.isbox if inverse else self.sbox
        func = sbox.coordinate_product(mask)
        func.do_Mobius()
        func.do_MaxSet()
        return func.to_Bins()

