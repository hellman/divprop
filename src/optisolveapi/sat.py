# https://github.com/pysathq/pysat
# pip install python-sat[pblib,aiger]
'''
Available ( https://pysathq.github.io/docs/html/api/solvers.html#list-of-classes ):
    Cadical : CaDiCaL SAT solver
    Glucose3 : Glucose 3 SAT solver
    Glucose4 : Glucose 4.1 SAT solver
    Lingeling : Lingeling SAT solver
    MapleChrono : MapleLCMDistChronoBT SAT solver
    MapleCM : MapleCM SAT solver
    Maplesat : MapleCOMSPS_LRB SAT solver
    Minicard : Minicard SAT solver
    Minisat22 : MiniSat 2.2 SAT solver
    MinisatGH : MiniSat SAT solver (version from github)
'''
from random import shuffle

try:
    from pysat.solvers import Solver
    has_pysat = True
except ImportError:
    has_pysat = False

from .base import SolverBase


class CNF(SolverBase):
    def __init__(self, solver="pysat/cadical"):
        self.init_solver(solver)
        self._var_cnt = 0

        self.ZERO = self.var()
        self.ONE = self.var()
        self.add_clause([-self.ZERO])
        self.add_clause([self.ONE])

    def init_solver(self, solver):
        raise NotImplementedError()

    def solve(self, assumptions=()):
        raise NotImplementedError()

    def var(self):
        self._var_cnt += 1
        return self._var_cnt

    def add_clause(self, c):
        self._solver.add_clause(c)

    def constraint_unary(self, vec):
        for a, b in zip(vec, vec[1:]):
            self.add_clause([a, -b])

    def constraint_and(self, a, b, ab):
        # a=1 b=1 => ab=1
        self.add_clause([-a, -b, ab])
        # a=0 => ab=0
        self.add_clause([a, -ab])
        # b=0 => ab=0
        self.add_clause([b, -ab])

    def SeqInc(self, vec):
        return [self.ONE] + list(vec)

    def SeqAddConst(self, vec, c):
        return [self.ONE] * c + list(vec)

    def SeqAdd(self, vec1, vec2):
        n1 = len(vec1)
        n2 = len(vec2)
        vec3 = [self.var() for i in range(n1 + n2)]
        ands = {}

        # self.constraint_unary(vec1)  # optional
        # self.constraint_unary(vec2)  # optional
        self.constraint_unary(vec3)

        for i in range(n1):
            ands[i, -1] = vec1[i]
            for j in range(n2):
                ands[i, j] = self.var()
                self.constraint_and(vec1[i], vec2[j], ands[i, j])
                ands[-1, j] = vec2[j]

        for isum in range(1, n1+n2+1):
            clause0 = [-vec3[isum-1]]
            for i in range(min(isum + 1, n1 + 1)):
                vi = vec1[i-1] if i else 0
                j = isum - i
                if j > n2:
                    continue
                vj = vec2[j-1] if j else 0

                # vec1[i] = 1, vec2[j] = 1 => vec3[i][isum] = 1
                clause = [vec3[isum-1], -vi, -vj]

                clause = [c for c in clause if c]
                self.add_clause(clause)

                clause0.append(ands[i-1, j-1])

            # FORALL i, j vec1[i] & vec2[j] = 0 => vec3[i][isum] = 0
            clause0 = [c for c in clause0 if c]
            self.add_clause(clause0)
        return vec3

    def SeqAddMany(self, *vecs):
        lst = list(vecs)
        while len(lst) >= 2:
            lst2 = []
            shuffle(lst)
            while len(lst) >= 2:
                lst2.append(self.SeqAdd(lst.pop(), lst.pop()))
            if lst:
                lst2.append(lst.pop())
            lst = lst2
        return lst[0]

    def SeqEq(self, vec1, vec2):
        if len(vec1) < len(vec2):
            self.add_clause([-vec2[len(vec1)]])
        elif len(vec2) < len(vec1):
            self.add_clause([-vec1[len(vec2)]])
        for a, b in zip(vec1, vec2):
            self.add_clause([a, -b])
            self.add_clause([-a, b])

    def SeqEqConst(self, vec, c):
        assert 0 <= c <= len(vec)
        if c == 0:
            self.add_clause([-vec[0]])
        elif c == len(vec):
            self.add_clause([vec[-1]])
        else:
            self.add_clause(vec[c-1])
            self.add_clause(-vec[c])

    # def SeqFloor(src, c):
    #     n = len(src)
    #     m = n // c
    #     dst = VarVec(m)
    #     for i in range(0, len(src), n):
    #         sub = src[i:i+c]
    #         if len(sub) != c:
    #             continue
    #         # dst = a & b & c

    #         # dst = 1 => a = 1
    #         # dst = 1 => b = 1
    #         vdst = dst[i//c]
    #         for vsrc in sub:
    #             S.add_clause([-vdst, vsrc])
    #         # dst = 0 => a = 0 v b = 0 v ...
    #         S.add_clause([vdst] + [-vsrc for vsrc in sub])
    #     return dst

    # def SeqCeil(src, c):
    #     n = len(src)
    #     m = (n + c - 1) // c
    #     dst = VarVec(m)
    #     for i in range(0, len(src), n):
    #         sub = src[i:i+c]
    #         # dst = a v b v c

    #         # dst = 0 => a = 0
    #         # dst = 0 => b = 0
    #         vdst = dst[i//c]
    #         for vsrc in sub:
    #             S.add_clause([vdst, -vsrc])
    #         # dst = 1 => a = 1 v b = 1 v ...
    #         S.add_clause([-vdst] + [vsrc for vsrc in sub])
    #     return dst

    def SeqMultConst(self, src, c):
        res = []
        for v in src:
            res += [v] * c
        return res

    def AlignPad(self, a, b):
        n = min(len(a), len(b)) + 1
        a = list(a) + [self.ZERO] * (n - len(a))
        b = list(b) + [self.ZERO] * (n - len(b))
        return a, b

    def SeqLess(self, a, b):
        # 1 0
        # 0 0
        a, b = self.AlignPad(a, b)
        n = len(a)

        # Bad (equal):
        # 1 0
        # 1 0
        for i in range(n-1):
            self.add_clause([-a[i], -b[i], a[i+1], b[i+1]])

        # Bad (greater):
        # 1
        # 0
        for i in range(n):
            self.add_clause([-a[i], b[i]])

    def SeqLessEqual(self, a, b):
        # 1 0
        # 0 0
        a, b = self.AlignPad(a, b)
        n = len(a)

        # Bad (greater):
        # 1
        # 0
        for i in range(n):
            self.add_clause([-a[i], b[i]])


@CNF.register("pysat/cadical")  # CaDiCaL SAT solver
@CNF.register("pysat/glucose3")  # Glucose 3 SAT solver
@CNF.register("pysat/glucose4")  # Glucose 4.1 SAT solver
@CNF.register("pysat/lingeling")  # Lingeling SAT solver
@CNF.register("pysat/maplechrono")  # MapleLCMDistChronoBT SAT solver
@CNF.register("pysat/maplecm")  # MapleCM SAT solver
@CNF.register("pysat/maplesat")  # MapleCOMSPS_LRB SAT solver
@CNF.register("pysat/minicard")  # Minicard SAT solver
@CNF.register("pysat/minisat22")  # MiniSat 2.2 SAT solver
@CNF.register("pysat/minisatgh")  # MiniSat SAT solver (version from github)
class PySAT(CNF):
    def init_solver(self, solver):
        assert has_pysat
        assert solver.startswith("pysat/")
        solver = solver[len("pysat/"):]
        self._solver = Solver(name=solver)

    def solve(self, assumptions=()):
        sol = self._solver.solve(assumptions=assumptions)
        if sol is None or sol is False:
            return False
        model = self._solver.get_model()
        return {(i+1): int(v > 0) for i, v in enumerate(model)}
