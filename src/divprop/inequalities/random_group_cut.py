from random import shuffle

from .base import (
    inner, satisfy,
    MixedIntegerLinearProgram, MIPSolverException,
)


class RandomGroupCut:
    """
    Algorithm to separate a lower set from an upper set (must be disjoint)
    by linear inequalities ("cuts").
    Sets may be given by their extremes (maxset and minset).

    Algorithm:
        1. choose incrementally a set of bad points
        2. try to find a linear functional separating them from good points
           (solving linear inequalities over reals)

    Algorithm is implemented for removing 'lo' while keeping 'hi'.
    For the other way around, vectors are flipped and swapped and
    output inequalities are adapted.
    """
    def __init__(self, lo, hi, inverted=False, solver="GLPK"):
        """
        pairwise is an old deprecated method
        where instead of having explicit separator constant c,
        the inequalities were <p,x> >= <q,x> for each p in good, q in bad
        (quadratic number).
        Seems not be useful, as introducing 1 variable for c reduces the number
        of inequalities significantly.
        """
        self.inverted = inverted
        self.orig_lo = lo
        self.orig_hi = hi
        if self.inverted:
            self.lo = [tuple(a ^ 1 for a in p) for p in hi]
            self.hi = [tuple(a ^ 1 for a in p) for p in lo]
        else:
            self.lo = [tuple(p) for p in lo]
            self.hi = [tuple(p) for p in hi]
        assert self.lo and self.hi
        self.n = len(self.lo[0])
        self.solver = solver
        self._prepare_constraints()
        self.seen_sorts = set()

    def integralize(self, ineq):
        pass

    def _prepare_constraints(self):
        self.model = MixedIntegerLinearProgram(solver=self.solver)
        self.var = self.model.new_variable(real=True, nonnegative=True)
        self.xs = [self.var["x%d" % i] for i in range(self.n)]
        self.c = self.var["c"]

        for p in self.hi:
            self.model.add_constraint(inner(p, self.xs) >= self.c)

        cs = {}
        for q in self.lo:
            cs[q] = inner(q, self.xs) <= self.c - 1

        self.cs_per_lo = cs

    def generate_inequality(self):
        LP = self.model.__copy__()
        covered_lo = []

        lstq = list(self.lo)
        shuffle(lstq)

        itr = 5
        while tuple(lstq) in self.seen_sorts:
            shuffle(lstq)
            itr -= 1
            if itr == 0:
                raise EOFError("exhausted")
        self.seen_sorts.add(tuple(lstq))

        for i, q in enumerate(lstq):
            constr_id = LP.number_of_constraints()
            LP.add_constraint(self.cs_per_lo[q])

            try:
                LP.solve()
            except MIPSolverException:
                assert i != 0
                LP.remove_constraint(constr_id)
            else:
                # print(f"covering #{i}/{len(lstq)}: {q}")
                covered_lo.append(q)
        LP.solve()

        val_xs = tuple(LP.get_values(x) for x in self.xs)
        if all(abs(v - round(v)) < 0.00001 for v in val_xs):
            # is integral
            val_xs = tuple(int(v + 0.5) for v in val_xs)
            val_c = int(LP.get_values(self.c) + 0.5)
        else:
            # keep real
            val_c = LP.get_values(self.c) - 0.5

        if self.inverted:
            # x1a1 + x2a2 + x3a3 >= t
            # =>
            # x1(1-a1) + x2(1-a2) + x3(1-a3) >= t
            # -x1a1 -x2a2 -x3a3 >= t-sum(x)
            value = val_c - sum(val_xs)
            sol = tuple(-x for x in val_xs) + (-value,)
            ret_covered = [tuple(1 - a for a in q) for q in covered_lo]
        else:
            # x1a1 + x2a2 + x3a3 >= t
            value = val_c
            sol = val_xs + (-value,)
            ret_covered = covered_lo

        # print(
        #     f"inequality coveres {len(covered_lo)}/{len(self.hi)}:",
        #     f"{func} >= {value_good}"
        # )
        return sol, ret_covered
