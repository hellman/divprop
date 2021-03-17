from divprop.subsets import Sbox2GraphIndicator, Sbox2Coordinates


def test_Sbox2GraphIndicator():
    a = Sbox2GraphIndicator([1, 2, 3, 4, 0, 7, 6, 5], 3, 3)
    print(a)
    assert str(a) == "b441a66a51d4a4f8 n=6 wt=8 | 1:2 2:1 3:2 4:1 5:2"


def test_Sbox2Coordinates():
    cs = Sbox2Coordinates([1, 2, 3, 4, 0, 7, 6, 5], 3, 3)

    assert str(cs[0]) == "3d139ee2d974e0e8 n=3 wt=4 | 2:3 3:1"
    assert str(cs[1]) == "53022f5ea743192d n=3 wt=4 | 1:2 2:2"
    assert str(cs[2]) == "5e0e6f8cfbc012e4 n=3 wt=4 | 0:1 1:1 2:1 3:1"

    assert list(cs[0]) == [3, 5, 6, 7]
    assert list(cs[1]) == [1, 2, 5, 6]
    assert list(cs[2]) == [0, 2, 5, 7]


if __name__ == '__main__':
    test_Sbox2GraphIndicator()
    test_Sbox2Coordinates()
