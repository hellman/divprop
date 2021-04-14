import os
import sys
import tempfile
import subprocess

from .base import MILP
from .symbase import SymBase


class External(SymBase):
    exec_path = NotImplemented

    def __init__(self, maximization, solver, exec_path=None):
        if exec_path is not None:
            self.exec_path = exec_path
        super().__init__(maximization, solver)

    def optimize_lp(self, *a, **k):
        raise NotImplementedError()

    def optimize(self, solution_limit=1, log=None, only_best=True):
        self.err = None
        with tempfile.NamedTemporaryFile(mode="w+") as f:
            self.write_lp(f.name)
            return self.optimize_lp(
                f.name,
                solution_limit=solution_limit,
                log=log,
                only_best=only_best,
            )


@MILP.register("external/glpk")
class ExtGLPK(External):
    exec_path = "glpsol"

    def optimize_lp(self, filename, solution_limit, log, only_best):
        # log = 1  # debug
        fmodel = tempfile.NamedTemporaryFile(mode="w+")
        fsol = tempfile.NamedTemporaryFile(mode="w+")

        if log:
            with open(filename, "r") as f:
                print("========================", file=sys.stderr)
                print("INPUT LP FILE", file=sys.stderr)
                print("========================", file=sys.stderr)
                print(f.read(), file=sys.stderr)

        cmd = [
            self.exec_path,
            "--lp", filename,
            "--write", fsol.name,
            "--wglp", fmodel.name,
        ]
        if self.n_ints == 0:
            cmd.append("--no-mip")
        if log:
            p = subprocess.Popen(
                cmd,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            out, err = p.communicate()  # ???
        else:
            p = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            out, err = p.communicate()

        if p.returncode:
            print("========================", file=sys.stderr)
            print("ERROR, GLPK OUTPUT", file=sys.stderr)
            print("========================", file=sys.stderr)
            print(out.decode(), file=sys.stderr)
            print(err.decode(), file=sys.stderr)
            print("========================", file=sys.stderr)
            print("MODEL FILE", file=sys.stderr)
            print("========================", file=sys.stderr)
            with open(filename, "r") as f:
                print(f.read(), file=sys.stderr)
            raise RuntimeError(f"GLPK returned {p.returncode}")

        id2var = {}
        for line in open(fmodel.name, "r"):
            if line.startswith("n j "):
                parts = line.split()
                assert len(parts) == 4
                varname = parts[-1]
                varid = int(parts[-2])
                assert varname in self.vars
                id2var[varid] = self.vars[varname]

        # c Status:     INTEGER OPTIMAL
        # c Objective:  obj = 6.3 (MINimum)
        # j 1 3
        # j 2 0
        # j 3 0
        sol = {}
        status = None
        obj = None

        if log:
            print("========================", file=sys.stderr)
            print("SOL FILE", file=sys.stderr)
            print("========================", file=sys.stderr)

        for line in open(fsol.name, "r"):
            line = line.strip()
            if log:
                print(line, file=sys.stderr)

            if line[0] == "c":
                parts = line.split()
                if len(parts) < 2:
                    continue
                if parts[1] == "Status:":
                    status = line.split(None, 2)[-1]

                    if status == "UNDEFINED":  # lp infeasible
                        return None
                    elif status == "OPTIMAL":  # lp feasible
                        pass
                    elif status == "INTEGER EMPTY":  # milp infeasible
                        return None
                    elif status == "INTEGER OPTIMAL":  # milp feasible
                        pass
                    else:
                        raise RuntimeError(f"unknown status '{status}'")
                elif parts[1] == "Objective:":
                    obj = float(parts[4])
                    if self.maximization:
                        assert parts[5] == "(MAXimum)"
                    else:
                        assert parts[5] == "(MINimum)"
            elif line[0] == "j":
                parts = line.split()
                assert len(parts) == 3
                varid = int(parts[1])
                var = id2var[varid]
                if self.vartype[var] == "C":
                    value = float(parts[2])
                else:
                    value = int(parts[2])
                sol[var] = value
        assert status is not None
        self.solutions = sol,
        return obj
