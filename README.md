# Installation:

Can be installed for pure python 3 (or pypy3),
however SageMath components (such as the GLPK solver) won't be enabled.

Here is recommended installation process for SageMath environment.

a) development mode:

```
$ make venv
$ . activate
$ pip install -U pip
$ pip install -U .
```

b) user mode:

```
$ sage -pip install -U pip
$ sage -pip install -U .

# For using solver=sage/gurobi frontend, Gurobi must be installed:
$ sage -pip install -U sage-numerical-backends-gurobi

# Note that solver=gurobi (gurobipy frontend) can also be used directly if installed.

# For using SCIPopt, SCIP suite must be installed:
$ sage -pip install -U pyscipopt
```

# Usage:

Simple call using default parameters:

```
$ divprop.sbox2ddt present
00:00:00.099 INFO root: starting at 2021-04-28.14:45:02
00:00:00.101 INFO divprop.tools: computing ddt support for 'present', output to data/sbox_present/ddt
00:00:00.102 INFO divprop.tools:  ddt: <DenseSet hash=3f030c7ecb0d54c4 n=8 wt=97 | 0:1 3:27 4:36 5:23 6:7 7:2 8:1>
00:00:00.102 INFO divprop.tools: ~ddt: <DenseSet hash=69011c8a56dedd3a n=8 wt=159 | 1:8 2:28 3:29 4:34 5:33 6:21 7:6>
00:00:00.103 INFO divprop.tools: by pairs: 0,0:1 1,2:14 1,3:9 1,4:1 2,1:13 2,2:16 2,3:10 2,4:1 3,1:11 3,2:10 3,3:6 3,4:1 4,1:2 4,3:1 4,4:1
Saving DenseSet(n=8) to data/sbox_present/ddt.set
Saving DenseSet(n=8) to data/sbox_present/ddt.good.set
Saving DenseSet(n=8) to data/sbox_present/ddt.bad.set

$ optimodel.milp data/sbox_present/ddt
...
00:00:05.271 INFO optimodel.tool_milp: inequalities (16):
00:00:05.272 INFO optimodel.tool_milp: (3, -3, 1, 2, 3, 4, -2, -3, 4)
00:00:05.272 INFO optimodel.tool_milp: (2, -5, -5, -1, 3, -6, 3, -4, 15)
00:00:05.272 INFO optimodel.tool_milp: (-6, 2, -5, -4, 4, -2, -5, -1, 17)
00:00:05.272 INFO optimodel.tool_milp: (-6, -5, 2, -4, -5, -2, 4, -1, 17)
00:00:05.273 INFO optimodel.tool_milp: (-1, 2, -3, 7, 5, 7, 7, -3, 0)
00:00:05.273 INFO optimodel.tool_milp: (2, -6, 2, 1, 4, 5, 4, 5, 0)
00:00:05.273 INFO optimodel.tool_milp: (3, 4, -3, -4, -4, -2, 2, 1, 9)
00:00:05.273 INFO optimodel.tool_milp: (3, -3, -2, 1, -1, -1, 3, -3, 7)
00:00:05.273 INFO optimodel.tool_milp: (3, -3, 4, -4, 2, -2, -4, 1, 9)
00:00:05.274 INFO optimodel.tool_milp: (2, 2, -6, 1, 4, 5, 4, 5, 0)
00:00:05.274 INFO optimodel.tool_milp: (-7, 5, 5, 11, 3, -4, 3, 10, 0)
00:00:05.274 INFO optimodel.tool_milp: (5, 5, 5, -6, 2, 1, 2, 4, 0)
00:00:05.274 INFO optimodel.tool_milp: (-4, 6, 6, -6, -2, 1, -2, -5, 13)
00:00:05.275 INFO optimodel.tool_milp: (-4, -2, -2, 3, -4, 1, -4, -2, 14)
00:00:05.282 INFO optimodel.tool_milp: (9, 7, 6, 7, -2, -1, -3, -3, 0)
00:00:05.283 INFO optimodel.tool_milp: (2, -4, -4, -1, -4, 2, -4, 3, 13)
00:00:05.283 INFO optimodel.tool_milp: end
```

## Customization

Default call for `optimodel.milp` uses the following commands:

```
AutoShifts
=
AutoChain ShiftLearn:threads=7 AutoSelect
=
Chain:LevelLearn,levels_lower=3 Chain:GainanovSAT,sense=min,save_rate=100,solver=pysat/cadical
ShiftLearn:threads=7
SubsetGreedy: SubsetWriteMILP:solver=sage/glpk SubsetMILP:solver=sage/glpk
```

Modification can be done at any level, for example to change the solver and threads:

```
$ optimodel.milp data/sbox_present/ddt AutoChain ShiftLearn:threads=3 SubsetMILP:solver=gurobi
```

# Remarks

- Intermediate computations are often saved and reused.