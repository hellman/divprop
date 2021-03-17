#pragma once

#include "common.hpp"
#include "DenseSet.hpp"

template<typename T>
struct T_Sbox {
    int n, m;
    std::vector<T> data;
    uint64_t xmask;
    uint64_t ymask;

    T_Sbox(int n, int m) {
        this->data.resize(1ull << n);
        this->n = n;
        this->m = m;
        init();
    }
    T_Sbox(const std::vector<T> &data, int n, int m) {
        this->data = data;
        this->n = n;
        this->m = m;
        init();
    }
    void init() {
        ensure(0 <= n && n <= 63);
        ensure(0 <= m && uint64_t(m) <= sizeof(T) * 8);
        ensure(data.size() == (1ull << n));
        xmask = (1ull << n) - 1;
        ymask = (m == 64) ? -1ull : ((1ull << m) - 1);
    }

    T_Sbox<T> inverse() const {
        ensure(n == m);
        T_Sbox<T> ret(n, m);
        fori (x, xmask + 1) {
            ret.set(data[x], x);
        }
        return ret;
    }
    T_Sbox<T> __invert__() const {
        return inverse();
    }

    T get(uint64_t x) const {
        ensure(x <= xmask);
        return data[x];
    }
    T set(uint64_t x, const T y) {
        ensure(x <= xmask);
        ensure(y <= ymask);
        data[x] = y;
        return y;
    }
    T __getitem__(uint64_t x) const {
        return get(x);
    }
    T __setitem__(uint64_t x, const T y) {
        return set(x, y);
    }

    DenseSet coordinate_product(T mask) const {
        DenseSet f(n);
        fori (x, 1ull << n) {
            uint64_t y = (uint64_t)data[x];
            int bit = (y & mask) == mask;
            f.set(x, bit);
        }
        return f;
    }
};