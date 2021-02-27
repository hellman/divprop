# sage/pure python compatibility
try:
    import sage.all
    from sage.numerical.mip import MixedIntegerLinearProgram
    from sage.numerical.mip import MIPSolverException
    from sage.all import Polyhedron
    is_sage = True
except ImportError:
    is_sage = False


def inner(a, b):
    return sum(aa * bb for aa, bb in zip(a, b))


def satisfy(pt, eq):
    """
    Inequality format:
    (a0, a1, a2, ..., a_{n-1}, c)
    a0*x0 + a1*x1 + ... + a_{n-1}*x_{n-1} + c >= 0
    """
    assert len(pt) + 1 == len(eq)
    return inner(pt, eq) + eq[-1] >= 0
