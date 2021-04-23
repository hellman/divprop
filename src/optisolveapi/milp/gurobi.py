try:
    import gurobipy as gp
    from gurobipy import GRB
    has_gurobi = True
except ImportError:
    has_gurobi = False

from .base import MILP


@MILP.register("gurobi")
class Gurobi(MILP):
    def __init__(self, maximization, solver):
        assert has_gurobi
        assert solver == "gurobi"
        self.model = gp.Model()
        self.model.setParam("OutputFlag", 0)
        # self.model.setParam("OutputFile", "")
        self.model.setParam("LogToConsole", 0)
        self.maximization = maximization
        self.vars = []

    def set_lb(self, var, lb=None):
        if lb is None:
            var.setAttr("lb", float("-inf"))
        else:
            var.setAttr("lb", lb)

    def set_ub(self, var, ub=None):
        if ub is None:
            var.setAttr("ub", float("-inf"))
        else:
            var.setAttr("ub", ub)

    def var_int(self, name, lb=None, ub=None):
        res = self.model.addVar(name=name, vtype="I")
        self.set_lb(res, lb)
        self.set_ub(res, ub)
        self.vars.append(res)
        return res

    def var_real(self, name, lb=None, ub=None):
        res = self.model.addVar(name=name, vtype="C")
        self.set_lb(res, lb)
        self.set_ub(res, ub)
        self.vars.append(res)
        return res

    def add_constraint(self, c):
        return self.model.addConstr(c)

    def remove_constraint(self, c):
        return self.model.remove(c)

    def remove_constraints(self, cs):
        for c in cs:
            return self.model.remove(c)

    def set_objective(self, obj):
        self._obj = obj
        if self.maximization:
            return self.model.setObjective(obj, GRB.MAXIMIZE)
        else:
            return self.model.setObjective(obj, GRB.MINIMIZE)

    def optimize(self, solution_limit=1, log=None, only_best=True):
        if not log:
            self.model.setParam("LogToConsole", 0)
        else:
            self.model.setParam("LogToConsole", 1)
            self.model.setParam("OutputFlag", 1)

        if solution_limit <= 1:
            self.model.setParam("PoolSearchMode", 0)
        else:
            self.model.setParam("PoolSearchMode", 2)
            self.model.setParam("PoolSolutions", solution_limit)

        self.solutions = []
        self.model.optimize()
        status = self.model.Status
        if status == GRB.INTERRUPTED:
            raise KeyboardInterrupt("gurobi was interrupted")
        assert status in (GRB.OPTIMAL, GRB.INFEASIBLE), status
        if status == GRB.INFEASIBLE:
            return

        if self.maximization is None:
            obj = True
        else:
            obj = self.trunc(self.model.objVal)

        if solution_limit != 0:
            for i in range(min(solution_limit, self.model.SolCount)):
                self.model.setParam("SolutionNumber", i)

                solobj = self.model.PoolObjVal
                if solobj + self.EPS < obj and only_best:
                    continue

                vec = {v: self.trunc(v.Xn) for v in self.vars}
                self.solutions.append(vec)
        return obj

    def write_lp(self, filename):
        assert filename.endswith(".lp")
        self.model.write(filename)
