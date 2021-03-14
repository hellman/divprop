# sage/pure python compatibility
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

try:
    from pyscipopt import Model as SCIPModel
    has_scip = True
except ImportError:
    has_scip = False

from divprop import logging

log = logging.getLogger(__name__)

# from pyscipopt.scip import PY_SCIP_STAGE
# print(PY_SCIP_STAGE.__dict__)


class MILP:
    BY_SOLVER = {}
    EPS = 1e-9
    debug = 0
    err = None

    @classmethod
    def maximization(cls, *args, solver="glpk", **opts):
        log.info(f"MILP maximization with solver '{solver}'")
        assert cls is MILP
        return cls.BY_SOLVER[solver.lower()](maximization=True, solver=solver)

    @classmethod
    def minimization(cls, solver="glpk"):
        log.info(f"MILP minimization with solver '{solver}'")
        assert cls is MILP
        return cls.BY_SOLVER[solver.lower()](maximization=False, solver=solver)

    @classmethod
    def register(cls, name):
        def deco(subcls):
            assert name not in cls.BY_SOLVER
            cls.BY_SOLVER[name.lower()] = subcls
            return subcls
        return deco

    def var_binary(self, name):
        return self.var_int(name, lb=0, ub=1)

    def trunc(self, v):
        if abs(round(v) - v) < self.EPS:
            return int(v + 0.5)
        return v


@MILP.register("glpk")
@MILP.register("coin")
@MILP.register("glpk/exact")  # LP only
@MILP.register("ppl")  # LP only
@MILP.register("cvxopt")  # LP only
@MILP.register("gurobi")  # need to be installed, commercial
@MILP.register("cplex")  # need to be installed, commercial
class SageMath_MixedIntegerLinearProgram(MILP):
    def __init__(self, maximization, solver):
        assert has_sage
        self.model = MixedIntegerLinearProgram(
            maximization=maximization, solver=solver,
        )
        self._var_int = self.model.new_variable(
            integer=True, nonnegative=False, name="I",
        )
        self._var_real = self.model.new_variable(
            real=True, nonnegative=False, name="R",
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


@MILP.register("scip")
class SCIP(MILP):
    def __init__(self, maximization, solver):
        assert has_scip
        assert solver == "scip"
        self.model = SCIPModel()
        self.maximization = maximization
        self.vars = []
        self.reopt = False

    def set_reopt(self):
        """
        has to be called befor setting the problem
        """
        self.model.enableReoptimization(True)
        self.reopt = True

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
