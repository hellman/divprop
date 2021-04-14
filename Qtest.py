from binteger import Bin
from divprop.subsets import DenseSet

n = 20
d = DenseSet(n)
for i in range(0, n, 2):
    v = (2**n-1) - (2**i + 2**(i+1))
    print(i, Bin(v, n).str)
    d.set(v)

print("MX           ", d)
d.do_LowerSet()
print("LOWER        ", d)
d.do_Complement()
print("UPPPER       ", d)
print("UPPER MIN SET", d.MinSet())
d.do_Not()
print("UPPER NOT    ", d)
d.do_LowerSet()
print("UPPER NOTLOW ", d)
d.do_MaxSet()
print("UPPER NOTMAX ", d)

print("bnd", n//2 * n)
print("-----------------")


n = 16
d = DenseSet(n)
# lst = [(2,), (1, 3), (1, 4)]
lst = [(1+i, 1+i+1) for i in range(0, n, 2)]
for inds in lst:
    v = sum(2**(i-1) for i in inds)
    # print("d", d, "v", v)
    d.set(v)

print("min              ", d)
d.do_UpperSet()
print("???              ", d.Complement().MaxSet())#ComplementU2L())
print("upper            ", d)
d.do_Complement()
print("lower            ", d)
d.do_Not()
print("lower NOT        ", d)
d.do_MinSet()
print("lower NOT MINSET ", d)#, d.to_Bins())
