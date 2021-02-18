from divprop.subsets import Sbox2GI, DenseSet


def mask(m):
    return (1 << m) - 1


class DivCore:
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
            graph.log_info("not-anf-max")

        return cls(graph, n, m)

    def LB(self):
        ret = self.data.copy()
        ret.do_ComplementU2L()
        return ret

    def UB(self):
        ret = self.data.copy()
        ret.do_UpperSet_Up1(True, self.mask_v)  # is_minset=true
        ret.do_MinSet()
        return ret

    def FullDPPT(self):
        ret = self.data.copy()
        ret.do_UpperSet()
        ret.do_Not(self.mask_u)
        return ret

    def MinDPPT(self):
        ret = self.data.copy()
        ret.do_UpperSet(self.mask_u)
        ret.do_MinSet(self.mask_v)
        ret.do_Not(self.mask_u)
        return ret
