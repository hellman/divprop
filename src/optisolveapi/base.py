class SolverBase:
    BY_SOLVER = {}

    @classmethod
    def register(cls, name):
        def deco(subcls):
            assert name not in cls.BY_SOLVER
            cls.BY_SOLVER[name.lower()] = subcls
            return subcls
        return deco

    @classmethod
    def new(cls, *args, solver, **opts):
        return cls.BY_SOLVER[solver.lower()](
            *args,
            solver=solver,
            **opts
        )
