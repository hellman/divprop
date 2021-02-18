from divprop.subsets import Sbox2GI, DenseSet


class DivCore:
    def __init__(self, data, n, m):
        assert isinstance(data, DenseSet)
        assert data.n == n + m
        self.data = data
        self.n = int(n)
        self.m = int(m)

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
