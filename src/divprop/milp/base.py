import logging

log = logging.getLogger(__name__)


class MILP:
    BY_SOLVER = {}
    EPS = 1e-9
    debug = 0
    err = None

    @classmethod
    def maximization(cls, *args, solver="sage/glpk", **opts):
        if not solver:
            solver = "glpk"
        log.info(f"MILP maximization with solver '{solver}'")
        assert cls is MILP
        return cls.BY_SOLVER[solver.lower()](
            *args,
            maximization=True, solver=solver,
            **opts
        )

    @classmethod
    def minimization(cls, *args, solver="sage/glpk", **opts):
        if not solver:
            solver = "glpk"
        log.info(f"MILP minimization with solver '{solver}'")
        assert cls is MILP
        return cls.BY_SOLVER[solver.lower()](
            *args,
            maximization=False, solver=solver,
            **opts
        )

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
        r = round(v)
        if abs(r - v) < self.EPS:
            return int(r)
        else:
            return v
