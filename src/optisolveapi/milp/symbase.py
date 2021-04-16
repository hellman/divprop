from .base import MILP


class SymBase(MILP):
    def __init__(self, maximization, solver):
        self.maximization = maximization
        self.solver = solver
        self.obj = None

        self.vars = {}
        self.vartype = {}
        self.n_ints = 0
        self.n_reals = 0

        self.constraints = set()
        self.lb = {}
        self.ub = {}

    def set_lb(self, var: "Var", lb=None):
        assert var in self.vars
        self.lb[var] = lb

    def set_ub(self, var: "Var", ub=None):
        assert var in self.vars
        self.ub[var] = ub

    def var_int(self, name, lb=None, ub=None):
        self.n_ints += 1
        v = Var(name)
        self.vartype[v] = "I"
        assert name not in self.vars
        self.vars[name] = v
        self.set_lb(v, lb)
        self.set_ub(v, ub)
        return v

    def var_real(self, name, lb=None, ub=None):
        self.n_reals += 1
        v = Var(name)
        self.vartype[v] = "C"
        assert name not in self.vars
        self.vars[name] = v
        self.set_lb(v, lb)
        self.set_ub(v, ub)
        return v

    def add_constraint(self, c):
        assert isinstance(c, Ineq)
        self.constraints.add(c)
        return c

    def remove_constraint(self, c):
        assert isinstance(c, Ineq)
        self.constraints.remove(c)

    def remove_constraints(self, cs):
        for c in cs:
            self.remove_constraint(c)

    def set_objective(self, obj):
        assert isinstance(obj, LinExpr)
        self.obj = obj

    def write_lp(self, filename):
        tab = ""
        with open(filename, "w") as f:
            if self.maximization:
                print("Maximize", file=f)
            else:
                print("Minimize", file=f)
            obj = self.obj
            if not obj:
                obj = sum(0*v for v in self.vars.items())
            print(tab, obj.strvars(), file=f)
            print("Subject To", file=f)
            for cons in self.constraints:
                print(tab, cons, file=f)
            if not self.constraints:
                for v in self.vars:
                    break
                print(tab, f"0 {v} >= 0", file=f)  # dummy
            print("Bounds", file=f)
            for varname, var in self.vars.items():
                assert var == varname
                lb = self.lb.get(var)
                ub = self.ub.get(var)
                if lb is None and ub is None:
                    print(tab, f"{varname} Free", file=f)
                elif lb is None:
                    print(tab, f"{varname} <= {ub}", file=f)
                elif ub is None:
                    print(tab, f"{lb} <= {varname}", file=f)
                else:
                    print(tab, f"{lb} <= {varname} <= {ub}", file=f)
            if self.n_reals:
                print("Generals", file=f)
                for varname, var in self.vars.items():
                    if self.vartype[var] == "C":
                        print(tab, f"{varname}", file=f)

            bins = set()
            for varname, var in self.vars.items():
                if self.vartype[var] == "I" and \
                   self.lb.get(var) == 0 and self.ub.get(var) == 1:
                    bins.add(var)

            if self.n_ints - len(bins):
                print("Integers", file=f)
                for varname, var in self.vars.items():
                    if self.vartype[var] == "I" and var not in bins:
                        print(tab, f"{varname}", file=f)

            if bins:
                print("Binary", file=f)
                for var in bins:
                    print(tab, f"{var}", file=f)
            print("End", file=f)
        return

    def _sumvars(self, args):
        res = {}
        for v in args:
            assert isinstance(v, Var)
            res[v] = 1
        return LinExpr(pairs=res, const=0)


class Var(str):
    # def __init__(self, name, vtype):
    #     assert vtype in "IR"
    #     self.vtype = vtype
    #     self.name = name

    def lift(self):
        return LinExpr(
            pairs={self: 1},
            const=0,
        )

    def cmul(self, c):
        return LinExpr(
            pairs={self: c},
            const=0,
        )

    def __add__(self, other):
        return self.lift() + other

    def __radd__(self, other):
        return self.lift() + other

    def __sub__(self, other):
        return self.lift() - other

    def __rsub__(self, other):
        return other - self.lift()

    def __mul__(self, other):
        assert isinstance(other, (int, float))
        return self.cmul(other)

    def __rmul__(self, other):
        assert isinstance(other, (int, float))
        return self.cmul(other)

    # def __hash__(self):
    #     return hash(self.name)

    # def __eq__(self, other):
    #     return self.name == other.name

    def __neg__(self):
        return self.cmul(-1)

    # def __str__(self):
    #     return self.name


class LinExpr:
    def __init__(self, pairs: dict, const: float):
        self.pairs = pairs
        self.const = const

    def _combine(self, other, cself=1, cother=1):
        # print("combine", self, other, cself, cother)
        assert isinstance(other, (LinExpr, Var, int, float))
        if isinstance(other, (int, float)):
            return LinExpr(
                pairs=self.pairs,
                const=cself*self.const + cother*other,
            )
        if isinstance(other, Var):
            other = other.lift()
        assert isinstance(other, LinExpr)
        pairs = {}
        for var, val in self.pairs.items():
            pairs.setdefault(var, 0)
            pairs[var] += cself * val
        for var, val in other.pairs.items():
            pairs.setdefault(var, 0)
            pairs[var] += cother * val
        const = cself * self.const + cother * other.const
        return LinExpr(pairs=pairs, const=const)

    def __add__(self, other):
        return self._combine(other, 1, 1)

    def __radd__(self, other):
        return self._combine(other, 1, 1)

    def __sub__(self, other):
        return self._combine(other, 1, -1)

    def __rsub__(self, other):
        return self._combine(other, -1, 1)

    def __ge__(self, other):
        # a >= b
        # a - b >= 0
        return Ineq(self - other)

    def __le__(self, other):
        # a <= b
        # b - a >= 0
        return Ineq(other - self)

    def __eq__(self, other):
        if isinstance(other, LinExpr):
            return self.pairs == other.pairs and self.const == other.const
        if isinstance(other, (int, float)):
            return Ineq(self - other, eq=True)
        raise TypeError(f"compare how? {other}")

    def __hash__(self):
        return hash(tuple(sorted(self.pairs.items()))) ^ hash(self.const)

    def strvars(self):
        return " + ".join(
            f"{coef} {var}" if coef != 1 else f"{var}"
            for var, coef in self.pairs.items()
        )

    def __str__(self):
        return self.strvars() + " + " + str(self.const)


class Ineq:
    def __init__(self, expr, eq=False):
        assert isinstance(expr, LinExpr)
        self.expr = expr
        self.eq = eq

    def __str__(self):
        if self.eq:
            return f"{self.expr.strvars()} = {-self.expr.const}"
        else:
            return f"{self.expr.strvars()} >= {-self.expr.const}"


class LPwriter:
    def __init__(self, filename):
        self.f = open(filename, "w")

    def print(self, *args):
        print(*args, file=self.f)

    def sum(self, args):
        return " + ".join(args)

    def objective(self, objective, sense="Maximize"):
        assert sense.lower() in ("maximize", "minimize")
        self.print(sense.capitalize())

        self.print("", objective)

        self.print("Subject To")

    def constraint(self, cons):
        self.print("", cons)

    def binaries(self, vs):
        self.print("Binary")
        for v in vs:
            self.print("", v)

    def generals(self, vs):
        """non-negative integers..."""
        self.print("General")
        for v in vs:
            self.print("", v)

    def bounds(self, vbs):
        for lb, varname, ub in vbs:
            if lb is None and ub is None:
                self.print("", f"{varname} Free")
            elif lb is None:
                self.print("", f"{varname} <= {ub}")
            elif ub is None:
                self.print("", f"{lb} <= {varname}")
            else:
                self.print("", f"{lb} <= {varname} <= {ub}")

    def close(self):
        self.print("End")
        self.f.close()
