from divprop.subsets import Sbox2GI


def test_Sbox2GraphIndicator():
    a = Sbox2GI([1, 2, 3, 4, 0, 7, 6, 5], 3, 3)
    assert str(a) == "b441a66a51d4a4f8 n=6 wt=8 | 1:2 2:1 3:2 4:1 5:2"


if __name__ == '__main__':
    test_Sbox2GraphIndicator()
