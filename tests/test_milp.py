from divprop.milp import MILP, has_scip, has_sage


def test_milp():
    solvers = []
    if has_sage:
        solvers.append("glpk")
        solvers.append("coin")
    if has_scip:
        solvers.append("scip")
    for solver in solvers:
        check_solver(solver)


def check_solver(solver):
    print("SOLVER", solver)
    milp = MILP.maximization(solver=solver)
    x = milp.var_int("x", 3, 7)
    y = milp.var_int("y", 2, 4)

    milp.set_objective(2*x)

    assert 14 == milp.optimize()
    assert milp.solutions
    for sol in milp.solutions:
        assert sol[x] == 7

    milp = MILP.maximization(solver=solver)
    x = milp.var_int("x", 3, 7)
    y = milp.var_int("y", 2, 4)
    milp.set_objective(-3*y)

    assert -6 == milp.optimize()
    assert milp.solutions
    for sol in milp.solutions:
        assert sol[y] == 2

    milp.add_constraint(x + y >= 9)

    assert -6 == milp.optimize()
    assert milp.solutions
    for sol in milp.solutions:
        assert sol[y] == 2

    c1 = milp.add_constraint(x + y >= 10)

    assert -9 == milp.optimize()
    assert milp.solutions
    for sol in milp.solutions:
        assert sol[y] == 3

    milp.remove_constraint(c1)

    assert -6 == milp.optimize()
    assert milp.solutions
    for sol in milp.solutions:
        assert sol[y] == 2


if __name__ == '__main__':
    test_milp()
