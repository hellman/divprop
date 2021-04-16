import time

from divprop.milp import MILP, has_scip, has_sage, has_gurobi


def test_milp():
    solvers = []
    if has_scip:
        solvers.append("scip")
    if has_sage:
        solvers.append("sage/glpk")
        solvers.append("sage/coin")
        if has_gurobi:
            solvers.append("sage/gurobi")
    if has_gurobi:
        solvers.append("gurobi")

    solvers.append("external/glpk")

    for i in range(2):
        for solver in solvers:
            t0 = time.time()
            check_solver(solver)
            print("solver", solver, "elapsed", f"{time.time() - t0:.3f}")
        print()


def check_solver(solver):
    print("SOLVER", solver)
    if 0:
        milp = MILP.maximization(solver=solver)
        x = milp.var_int("x", 3, 7)
        y = milp.var_int("y", 2, 4)

        milp.set_objective(2*x)

        assert 14 == milp.optimize()
        assert milp.solutions
        for sol in milp.solutions:
            assert sol[x] == 7

    # ==================================

    milp = MILP.maximization(solver=solver)
    x = milp.var_int("x", 3, 7)
    y = milp.var_int("y", 2, 4)
    milp.set_objective(-3*y)

    assert -6 == milp.optimize()
    assert milp.solutions
    for sol in milp.solutions:
        assert sol[y] == 2

    milp.add_constraint(x + y >= 9)

    obj = milp.optimize()
    assert -6 == obj, obj
    assert milp.solutions
    for sol in milp.solutions:
        assert sol[y] == 2

    for i in range(100):
        c1 = milp.add_constraint(x + y >= 10)

        assert -9 == milp.optimize()
        assert milp.solutions
        for sol in milp.solutions:
            assert sol[y] == 3

        milp.remove_constraint(c1)
        obj = milp.optimize()
        assert -6 == obj, obj
        assert milp.solutions
        for sol in milp.solutions:
            assert sol[y] == 2

        c2 = milp.add_constraint(x + y == 10)

        assert -9 == milp.optimize()
        assert milp.solutions
        for sol in milp.solutions:
            assert sol[y] == 3

        milp.remove_constraint(c2)
        obj = milp.optimize()
        assert -6 == obj, obj
        assert milp.solutions
        for sol in milp.solutions:
            assert sol[y] == 2


# reopt is buggy..
def disabled_test_scip_reopt():
    if not has_scip:
        return

    milp = MILP.maximization(solver="scip")
    milp.set_reopt()

    x = milp.var_int("x", 3, 7)
    y = milp.var_int("y", 2, 4)
    milp.set_objective(-3*y)

    assert -6 == milp.optimize()
    assert milp.solutions
    for sol in milp.solutions:
        assert sol[y] == 2

    milp.add_constraint(x + y >= 9)

    obj = milp.optimize()
    assert -6 == obj, obj
    assert milp.solutions
    for sol in milp.solutions:
        assert sol[y] == 2

    milp.add_constraint(x + y >= 10)

    assert -9 == milp.optimize()
    assert milp.solutions
    for sol in milp.solutions:
        assert sol[y] == 3

    milp.add_constraint(x + y >= 11)

    assert -12 == milp.optimize()
    assert milp.solutions
    for sol in milp.solutions:
        assert sol[y] == 4

    milp.add_constraint(x + y >= 12)

    assert None is milp.optimize()
    assert not milp.solutions


if __name__ == '__main__':
    test_milp()
