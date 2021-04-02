import os
import sys
import ast

metafile = sys.argv[1]
assert metafile.endswith(".meta")
prefix = metafile[:-5]


for i in range(2**30):
    if not os.path.exists(prefix + ".lp_%d.sol" % i):
        break

i -= 1

solfile = prefix + ".lp_%d.sol" % i
print("using solfile", solfile)

taken = set()
with open(solfile) as f:
    for line in f:
        if line.startswith("x"):
            v, take = line.split()
            if take == "1":
                v = int(v[1:])
                taken.add(v)

print("lp taken", len(taken), ":", taken)

res = []
n_greedy = 0
for line in open(metafile):
    i, fset, ineq, greedy_taken = line.split()
    i = int(i)
    if greedy_taken == "1" or i in taken:
        res.append(tuple(map(int, ineq.split(":"))))
        print(res[-1])
        if greedy_taken == "1":
            n_greedy += 1
        else:
            taken.remove(i)

assert not taken
print(len(res), "total")
