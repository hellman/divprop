def truncrepr(s, n=100):
    s = repr(s)
    if len(s) > n:
        s = s[:n] + "..."
    return s


def truncstr(s, n=100):
    s = str(s)
    if len(s) > n:
        s = s[:n] + "..."
    return s
