try:
    from pyscipopt import Model as SCIPModel
    has_scip = True
except ImportError:
    has_scip = False

from .base import MILP


@MILP.register("scip")
class SCIP(MILP):
    def __init__(self, maximization, solver):
        assert has_scip
        assert solver == "scip"
        self.model = SCIPModel()
        self.maximization = maximization
        self.vars = []
        self.reopt = False

    # def set_reopt(self):
    #     """
    #     has to be called befor setting the problem
    #     """
    #     raise NotImplementedError("scip's reopt seems buggy")
    #     self.model.enableReoptimization(True)
    #     self.reopt = True

    def set_lb(self, var, lb=None):
        self.model.chgVarLbGlobal(var, lb)

    def set_ub(self, var, ub=None):
        self.model.chgVarUbGlobal(var, ub)

    def var_int(self, name, lb=None, ub=None):
        res = self.model.addVar(name, vtype="I")
        self.set_lb(res, lb)
        self.set_ub(res, ub)
        self.vars.append(res)
        return res

    def var_real(self, name, lb=None, ub=None):
        res = self.model.addVar(name, vtype="C")  # continuous
        self.set_lb(res, lb)
        self.set_ub(res, ub)
        self.vars.append(res)
        return res

    def add_constraint(self, c):
        return self.model.addCons(c)

    def remove_constraint(self, c):
        assert not self.reopt, "can not remove constraints in reopt..."
        return self.model.delCons(c)

    def remove_constraints(self, cs):
        assert not self.reopt, "can not remove constraints in reopt..."
        for c in cs:
            return self.model.delCons(c)

    def set_objective(self, obj):
        self._obj = obj
        if self.maximization:
            return self.model.setObjective(obj, sense="maximize")
        else:
            return self.model.setObjective(obj, sense="minimize")

    # def copy(self):
    #     ret = object.__new__(type(self))
    #     ret.model = SCIPModel(sourceModel=self.model)
    #     ret.maximization = self.maximization
    #     ret.vars = list(self.getVars())
    #     return ret

    def optimize(self, solution_limit=1, log=None, only_best=True):
        if not log:
            self.model.hideOutput(True)
        else:
            self.model.hideOutput(False)

        self.solutions = []
        self.model.optimize()

        status = self.model.getStatus()
        assert status in ("optimal", "infeasible"), status
        if status == "infeasible":
            self.model.freeTransform()
            return

        obj = self.trunc(self.model.getObjVal())
        if solution_limit != 0:
            for sol in self.model.getSols():
                solobj = self.model.getSolObjVal(sol)
                if solobj + self.EPS < obj and only_best:
                    continue

                vec = IdResolver({
                    str(v): self.trunc(self.model.getSolVal(sol, v))
                    for v in self.vars
                })
                self.solutions.append(vec)
                if solution_limit and len(self.solutions) >= solution_limit:
                    break

        if self.reopt:
            self.model.freeReoptSolve()
        else:
            self.model.freeTransform()
        return obj


class IdResolver:
    """Stub to allow dict-like solutions when var is not hashable..."""
    __slots__ = "sol",

    def __init__(self, sol):
        self.sol = sol

    def __getitem__(self, v):
        return self.sol[str(v)]

    def items(self):
        return [(v, y) for v, y in self.sol.items()]

    def __repr__(self):
        return "(" + ", ".join(f"{y}" for v, y in self.items()) + ")"
