from divprop.subsets import Sbox2GI


def test_Sbox2GraphIndicator():
    a = Sbox2GI([1, 2, 3, 4, 0, 7, 6, 5], 3, 3)
    print(a.info("SetA"))


if __name__ == '__main__':
    test_Sbox2GraphIndicator()
