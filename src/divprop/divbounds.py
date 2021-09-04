import logging

from optisolveapi.sat import ConvexFormula
from optisolveapi.vector import Vector

from divprop import DivCore, Sbox
from divprop.utils import cached_method


class DivBounds:
    log = logging.getLogger()

    def __init__(self, sbox, divcore=None):
        self.sbox = sbox
        self.cache_key = "%016X" % sbox.get_hash()

        self._divcore = divcore

    @property
    @cached_method
    def divcore(self):
        return DivCore.from_sbox(self.sbox)

    @property
    @cached_method
    def full(self):
        return self.divcore.UpperSet()

    @property
    @cached_method
    def lb(self):
        return self.divcore.get_Invalid()

    @property
    @cached_method
    def ub(self):
        return self.divcore.get_Redundant()

    @property
    @cached_method
    def ub2(self):
        return self.divcore.get_RedundantAlternative()

    @property
    @cached_method
    def ubest(self):
        """best of ub and ub2"""
        if self.ub.get_weight() < self.ub2.get_weight():
            return self.ub
        return self.ub2

    @property
    @cached_method
    def hull(self):
        return self.divcore.get_Minimal()

    @property
    @cached_method
    def dppt_full(self):
        return self.divcore.FullDPPT()

    @property
    @cached_method
    def dppt_min(self):
        return self.divcore.MinDPPT()

    # CONSTRAINTS

    @property
    @cached_method
    def cnf_lb(self):
        return ConvexFormula(lb=self.lb)

    @property
    @cached_method
    def cnf_ub(self):
        return ConvexFormula(ub=self.ub)

    @property
    @cached_method
    def cnf_ub2(self):
        return ConvexFormula(ub=self.ub2)

    @property
    @cached_method
    def cnf_ubest(self):
        return ConvexFormula(ub=self.ubest)

    @cached_method
    def box(self, dimensions):
        assert sum(dimensions) == self.sbox.n + self.sbox.m
        print(self.hull)
        return self.hull.to_DenseBox(dimensions)

    @cached_method
    def box_lb(self, dimensions):
        return self.box(dimensions).MinSet().ComplementU2L()

    @cached_method
    def box_ub(self, dimensions):
        return self.box(dimensions).MaxSet().ComplementL2U()

    def sat_constraint_lb(self, solver, xs, ys):
        assert len(xs) == self.sbox.n
        assert len(ys) == self.sbox.m
        solver.apply(self.cnf_lb, Vector(-xs).concat(ys))

    def sat_constraint_ub(self, solver, xs, ys):
        assert len(xs) == self.sbox.n
        assert len(ys) == self.sbox.m
        solver.apply(self.cnf_ub, Vector(-xs).concat(ys))

    def sat_constraint_ub2(self, solver, xs, ys):
        assert len(xs) == self.sbox.n
        assert len(ys) == self.sbox.m
        solver.apply(self.cnf_ub2, Vector(-xs).concat(ys))

    def sat_constraint_ubest(self, solver, xs, ys):
        assert len(xs) == self.sbox.n
        assert len(ys) == self.sbox.m
        solver.apply(self.cnf_ubest, Vector(-xs).concat(ys))


if __name__ == '__main__':
    import justlogs
    justlogs.setup(level="DEBUG")

    db = DivBounds(Sbox([1, 2, 3, 0], 2, 2))
    print(db.divcore._dense)
    print(db.box_ub([2, 2]))
