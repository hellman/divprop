def inner(a, b):
    return sum(aa * bb for aa, bb in zip(a, b))


def satisfy(pt, ineq):
    """
    Inequality format:
    (a0, a1, a2, ..., a_{n-1}, c)
    a0*x0 + a1*x1 + ... + a_{n-1}*x_{n-1} + c >= 0
    """
    assert len(pt) + 1 == len(ineq)
    return inner(pt, ineq) + ineq[-1] >= 0
