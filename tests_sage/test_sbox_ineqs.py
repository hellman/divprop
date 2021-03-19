from divprop.sbox_ineqs import get_sbox, get_sbox_sizes
from divprop.divcore import DivCore
from divprop.inequalities import satisfy
from binteger import Bin
from glob import glob


def test_ineqs():
    todo = (
        "present",
        "rectangle",
        "gift",
        "ascon",
        "midori_sb0",
        "aes",
    )
    for name in todo:
        sbox = get_sbox(name)
        n, m = get_sbox_sizes(sbox)

        print("checking sbox", name)
        check_sbox_full(name.lower(), sbox, n, m)


def check_sbox_full(name, sbox, n, m):
    dc = DivCore.from_sbox(sbox, n, m)
    mid = dc.MinDPPT().Not(dc.mask_u)
    lb = dc.LB()
    # dclo = mid.MinSet()
    dcup = mid.MaxSet()

    for filename in glob(f"results/{name}_sbox.*_ineq"):
        print("\nfile:", filename)
        _, typ, num = filename.split(".")
        num = int(num.split("_")[0])

        with open(filename) as f:
            n_ineqs = int(f.readline())
            ineqs = [tuple(map(int, line.split())) for line in f]
            assert len(ineqs) == n_ineqs
            for ineq in ineqs:
                print("ineq", ineq)

        if typ == "lb":
            points_good = dc.to_dense()
            points_bad = lb
            type_good = "upper"
        elif typ == "ubo":
            points_good = dcup.LowerSet()
            points_bad = points_good.Complement() - lb.LowerSet()

            points_good = points_good.MaxSet()
            points_bad = points_bad.MinSet()
            type_good = "lower"
        elif typ == "ubc":
            points_good = dcup.LowerSet()
            points_bad = points_good.Complement()

            points_good = points_good.MaxSet()
            points_bad = points_bad.MinSet()
            type_good = "lower"
        elif typ == "full":
            continue
        else:
            assert 0

        print(f"points_good {points_good}")
        print(f"points_bad {points_bad}")
        if 1:
            if type_good == "upper":
                points_good.do_UpperSet()
                points_bad.do_LowerSet()
            else:
                points_good.do_LowerSet()
                points_bad.do_UpperSet()

        points_good = {Bin(p, n+m).tuple for p in points_good}
        points_bad = {Bin(p, n+m).tuple for p in points_bad}

        for p in points_good:
            for ineq in ineqs:
                if not satisfy(p, ineq):
                    print("good point removed", p)
                    # assert False, (p, ineq)
                    break

        for p in points_bad:
            for ineq in ineqs:
                if not satisfy(p, ineq):
                    break
            else:
                print("bad point kept", p)
                # assert False, p
                break


if __name__ == '__main__':
    test_ineqs()
