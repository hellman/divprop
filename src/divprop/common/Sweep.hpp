#pragma once

#include "common.hpp"

// #include "bitvec.cc"

template<typename T>
vector<T> neibs_up(T u, int n) {
    vector<T> res;
    fori (i, n) {
        uint64_t bit = (1ull << i);
        if ((u & bit) == 0) {
            res.push_back(u ^ bit);
        }
    }
    return res;
}
template<typename T>
vector<T> neibs_down(T u, int n) {
    vector<T> res;
    fori (i, n) {
        uint64_t bit = (1ull << i);
        if ((u & bit) == 1) {
            res.push_back(u ^ bit);
        }
    }
    return res;
}

TTi void XOR_down(T &a, T &b) { a ^= b; }
TTi void XOR_up(T &a, T &b) { b ^= a; }
TTi void OR_down(T &a, T &b) { a |= b; }
TTi void OR_up(T &a, T &b) { b |= a; }
TTi void AND_down(T &a, T &b) { a &= b; }
TTi void AND_up(T &a, T &b) { b &= a; }

TTi void SWAP(T &a, T &b) { swap(a, b); }

// b &= (a < b)
TTi void LESS_up(T &a, T &b) { b &= ~a; }
// template<>
// void LESS_up(BitVec &a, BitVec &b) {
//     assert(a.n == b.n);
//     fori (i, a.nwords()) {
//         b.data[i] &= ~a.data[i];
//     }
// }
// a &= (a > b)
TTi void MORE_down(T &a, T &b) { a &= ~b; }
// template<>
// void MORE_down(BitVec &a, BitVec &b) {
//     assert(a.n == b.n);
//     fori (i, a.nwords()) {
//         a.data[i] &= ~b.data[i];
//     }
// }


template<auto func, typename T>
void GenericSweep(vector<T> &arr, uint64_t mask) {
    auto size = arr.size();
    int n = log2(size);
    fori (k, n) {
        if ((mask & (1ull << k)) == 0)
            continue;

        uint64_t bit = (1ull << k);
        fori (i, 1ull << n) {
            if (i & bit) {
                func(arr[i ^ bit], arr[i]);
            }
        }
    }
}

template<auto func>
static inline void GenericSweepWordBit(uint64_t &word, int shift, uint64_t mask) {
    // /!\ here mask corresponds to word mask,
    // not index mask as in other places !!!
    uint64_t lo, hi;
    lo = word & mask;
    hi = (word >> shift) & mask;
    func(lo, hi);
    word = (hi << shift) | lo;
}
template<auto func>
void GenericSweepWord(uint64_t &word, uint64_t mask) {
    if (mask & (32)) GenericSweepWordBit<func>(word, 32, 0x00000000ffffffffull);
    if (mask & (16)) GenericSweepWordBit<func>(word, 16, 0x0000ffff0000ffffull);
    if (mask & ( 8)) GenericSweepWordBit<func>(word,  8, 0x00ff00ff00ff00ffull);
    if (mask & ( 4)) GenericSweepWordBit<func>(word,  4, 0x0f0f0f0f0f0f0f0full);
    if (mask & ( 2)) GenericSweepWordBit<func>(word,  2, 0x3333333333333333ull);
    if (mask & ( 1)) GenericSweepWordBit<func>(word,  1, 0x5555555555555555ull);
}