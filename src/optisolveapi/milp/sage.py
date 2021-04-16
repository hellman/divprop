try:
    import sage.all
    from sage.numerical.mip import MixedIntegerLinearProgram
    from sage.numerical.mip import MIPSolverException
    from sage.all import Polyhedron
    has_sage = True
except ImportError:
    MixedIntegerLinearProgram = None
    MIPSolverException = None
    Polyhedron = None

    has_sage = False


from .base import MILP


@MILP.register("sage/glpk")
@MILP.register("sage/coin")
@MILP.register("sage/glpk/exact")  # LP only
@MILP.register("sage/ppl")  # LP only
@MILP.register("sage/cvxopt")  # LP only
@MILP.register("sage/cplex")  # need to be installed, commercial
@MILP.register("sage/gurobi")  # need to be installed, commercial
class SageMath_MixedIntegerLinearProgram(MILP):
    def __init__(self, maximization, solver):
        assert has_sage
        assert solver.startswith("sage/")
        solver = solver[len("sage/"):]
        self.model = MixedIntegerLinearProgram(
            maximization=maximization, solver=solver,
        )
        self._var_int = self.model.new_variable(
            integer=True, nonnegative=False, name="I",
        )
        self._var_real = self.model.new_variable(
            real=True, nonnegative=False, name="C",
        )
        self.vars = []
        self.constraints = []

    def set_lb(self, var, lb=None):
        self.model.set_min(var, lb)

    def set_ub(self, var, ub=None):
        self.model.set_max(var, ub)

    def var_int(self, name, lb=None, ub=None):
        res = self._var_int[name]
        self.set_lb(res, lb)
        self.set_ub(res, ub)
        self.vars.append(res)
        return res

    def var_real(self, name, lb=None, ub=None):
        res = self._var_real[name]
        self.set_lb(res, lb)
        self.set_ub(res, ub)
        self.vars.append(res)
        return res

    def add_constraint(self, c):
        cid = str(c)
        n1 = self.model.number_of_constraints()
        self.model.add_constraint(c)
        n2 = self.model.number_of_constraints()
        assert n2 == n1 + 1
        assert len(self.constraints) == n1
        self.constraints.append(cid)
        assert len(self.constraints) == n2
        return cid

    def remove_constraint(self, cid):
        assert isinstance(cid, str)
        for i, ccid in enumerate(self.constraints):
            if ccid == cid:
                del self.constraints[i]
                self.model.remove_constraint(i)
                break
        else:
            raise KeyError(f"unknown constraint str={cid}")
        assert len(self.constraints) == self.model.number_of_constraints()

    def remove_constraints(self, cs):
        csids = set(map(str, cs))
        inds = set()
        for i, cid in enumerate(self.constraints):
            if cid in csids:
                inds.add(i)
                csids.remove(cid)
        assert not csids, "not all constraints found"
        self.constraints = [
            c for i, c in enumerate(self.constraints)
            if i not in inds
        ]
        self.model.remove_constraints(sorted(inds))
        assert len(self.constraints) == self.model.number_of_constraints()

    def set_objective(self, obj):
        return self.model.set_objective(obj)

    # def copy(self):
    #     obj = object.__new__(type(self))
    #     obj.model = self.model.__copy__()
    #     obj.constraints = self.model.constraints[::]
    #     obj.vars = self.model.vars[::]
    #     return obj

    def optimize(self, solution_limit=1, log=None, only_best=True):
        self.err = None
        try:
            obj = self.trunc(self.model.solve(log=log))
        except MIPSolverException as err:
            self.err = err
            return
        if solution_limit == 0:
            return obj

        # sagemath returns only 1 solution
        vec = {v: self.trunc(self.model.get_values(v)) for v in self.vars}
        self.solutions = vec,
        return obj
