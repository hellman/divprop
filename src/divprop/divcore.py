from divprop.subsets import Sbox2GI, DenseSet


def mask(m):
    return (1 << m) - 1


class DenseDivCore:
    """
    Division Core of the S-Box = reduced DPPT of its graph (as a dense set).

    Notation: vectors (u, v), with bit-length (n, m).

    Wrapper for DenseSet, stored in :attr:`data`.
    """
    def __init__(self, data, n, m):
        assert isinstance(data, DenseSet)
        assert data.n == n + m
        self.data = data
        self.n = int(n)
        self.m = int(m)
        self.mask_u = mask(n) << m
        self.mask_v = mask(m)

    @classmethod
    def from_sbox(cls, sbox, n, m, log=False):
        graph = Sbox2GI(sbox, n, m)

        if log:
            graph.log_info("graph")

        graph.do_Mobius()

        if log:
            graph.log_info("anf")

        graph.do_MaxSet()

        if log:
            graph.log_info("anf-max")

        graph.do_Not()

        if log:
            graph.log_info("divcore")

        return cls(graph, n, m)

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
            #ret = self.MinDPPT()
            #ret.do_Not(self.mask_u)
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
