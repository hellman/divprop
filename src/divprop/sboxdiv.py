import logging

from optisolveapi.sat import ConvexFormula
from optisolveapi.vector import Vector

from subsets import DenseSet

from divprop import DivCore, Sbox
from divprop.utils import cached_method


class SboxDivision:
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
    Invalid = lb

    @property
    @cached_method
    def ub(self):
        return self.divcore.get_Redundant()
    Redundant = ub

    @property
    @cached_method
    def ub2(self):
        return self.divcore.get_RedundantAlternative()
    RedundantAlternative = ub2

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
    Minimal = hull

    @property
    @cached_method
    def hull_max(self):
        return self.hull.MaxSet()
    hull_min = divcore

    @property
    @cached_method
    def dppt_full(self):
        return self.divcore.FullDPPT()

    @property
    @cached_method
    def dppt_min(self):
        return self.divcore.MinDPPT()

    @cached_method
    def components_anf_closures(self, remove_dups_by_maxset=True, only_minimal=True):
        """
        (unique/non-redundant) closures of ANFs of components
        """
        # linear-time build all components
        cs = list(self.sbox.coordinates())
        xors = [DenseSet(16)] + [None] * (2**16-1)
        for i in range(16):
            for j in range(2**i):
                xors[j + 2**i] = xors[j] ^ cs[15 - i]

        # ANF closures
        anfs_full = {
            mask: xor.Mobius().UpperSet()
            for mask, xor in enumerate(xors)
        }
        anfs_max = {
            mask: anf.MaxSet()
            for mask, anf in anfs_full.items()
        }
        del xors
        assert len(anfs_full) == 2**16

        if remove_dups_by_maxset:
            unique = []
            seen = {}
            for mask in range(1, 2**16):
                mx = anfs_max[mask]
                h = mx.get_hash()
                if h in seen:
                    # check that it's not a hash collision
                    assert seen[h] == mx
                    continue
                seen[h] = mx
                unique.append(mask)
            del seen
            self.log.debug(f"unique masks {len(unique)}")
            anfs_full = {mask: anfs_full[mask] for mask in unique}

        if only_minimal:
            minimal = []
            for mask1 in anfs_full:
                for mask2 in anfs_full:
                    if mask1 == mask2:
                        continue
                    if is_max_preceq_full(anfs_max[mask2], anfs_full[mask1]):
                        break
                else:
                    minimal.append(mask1)
            anfs_full = {mask: anfs_full[mask] for mask in minimal}
        return anfs_full

    def inverse(self):
        return SboxDivision(sbox=self.sbox.inverse())

    # CONSTRAINTS
    # ======================================

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


def is_max_preceq_full(s1: DenseSet, s2: DenseSet):
    """Faster variant. s2 must be lower-closed"""
    for u in s1:
        if u not in s2:
            return False
    return True


if __name__ == '__main__':
    import justlogs
    justlogs.setup(level="DEBUG")

    db = SboxDivision(Sbox([1, 2, 3, 0], 2, 2))
    print(db.divcore._dense)
    print(db.box_ub([2, 2]))
