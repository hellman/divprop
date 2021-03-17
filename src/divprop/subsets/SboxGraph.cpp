#include "SboxGraph.hpp"

template<> DenseSet T_Sbox2GraphIndicator<u8>(const vector<u8> &sbox, int n, int m);
template<> DenseSet T_Sbox2GraphIndicator<u16>(const vector<u16> &sbox, int n, int m);
template<> DenseSet T_Sbox2GraphIndicator<u32>(const vector<u32> &sbox, int n, int m);
template<> DenseSet T_Sbox2GraphIndicator<u64>(const vector<u64> &sbox, int n, int m);
template<> DenseSet T_Sbox2GraphIndicator<int>(const vector<int> &sbox, int n, int m);

template<> vector<DenseSet> T_Sbox2Coordinates<u8>(const vector<u8> &sbox, int n, int m);
template<> vector<DenseSet> T_Sbox2Coordinates<u16>(const vector<u16> &sbox, int n, int m);
template<> vector<DenseSet> T_Sbox2Coordinates<u32>(const vector<u32> &sbox, int n, int m);
template<> vector<DenseSet> T_Sbox2Coordinates<u64>(const vector<u64> &sbox, int n, int m);
template<> vector<DenseSet> T_Sbox2Coordinates<int>(const vector<int> &sbox, int n, int m);