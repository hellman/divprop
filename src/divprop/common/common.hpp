#pragma once

#include <bits/stdc++.h>
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>

#include <assert.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

using namespace std;

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;

typedef int64_t i64;
typedef int32_t i32;
typedef int16_t i16;
typedef int8_t i8;

#define TTi template<typename T> inline

#define __CONCAT3_NX(x, y, z) x ## y ## z
#define __CONCAT3(x, y, z) __CONCAT3_NX(x, y, z)
#define __VAR(name) __CONCAT3(__tmpvar__, name, __LINE__)

// https://stackoverflow.com/a/11763277
// overload by number of arguments
#define GET_MACRO3(_1, _2, _3, NAME, ...) NAME

#define fori(i, ...) GET_MACRO3(__VA_ARGS__, fori3, fori2, fori1)(i, __VA_ARGS__)
#define fori1(i, end) \
    for (int64_t i = 0, __VAR(vend) = (end); i < __VAR(vend); i++)
#define fori2(i, start, end) \
    for (int64_t i = (start), __VAR(vend) = (end); i < __VAR(vend); i++)
#define fori3(i, start, end, step) \
    for (int64_t i = (start), __VAR(vend) = (end), __VAR(vstep) = (step); i < __VAR(vend); i += __VAR(vstep))

#define rfori(i, ...) GET_MACRO3(__VA_ARGS__, rfori3, rfori2, rfori1)(i, __VA_ARGS__)
#define rfori1(i, end) \
    for (int64_t i = (end)-1; i >= 0; i--)
#define rfori2(i, start, end) \
    for (int64_t i = (end)-1, __VAR(vstart) = (start); i >= __VAR(vstart); i--)
#define rfori3(i, start, end, step) \
    for (int64_t __VAR(vstart) = (start), __VAR(vstep) = (step), i = __VAR(vstart) + (((end)-1-__VAR(vstart)) / __VAR(vstep)) * __VAR(vstep); i >= __VAR(vstart); i -= __VAR(vstep))

inline u64 HI(u64 x) {
    return x >> 6;
}

inline u64 LO(u64 x) {
    return x & 0x3f;
}

inline u64 HICEIL(u64 x) {
    return (x >> 6) + ((x & 0x3f) != 0);
}

inline int hw(u64 x) {
    return __builtin_popcountll(x);
}

inline int log2(u64 n) {
    assert((n & (n - 1)) == 0);
    assert(n != 0);
    int ret = __builtin_ctzll(n);
    assert((1ull << ret) == n);
    return ret;
}
inline u64 GF_MUL2(u64 x, int n, u64 poly) {
    x <<= 1;
    if (x >> n) {
        x ^= poly;
    }
    return x;
}
inline u64 GF_MUL(u64 x, u64 y, int n, u64 poly) {
    u64 res = 0;
    while (y) {
        if (y & 1) {
            res ^= x;
        }
        x = GF_MUL2(x, n, poly);
        y >>= 1;
    }
    return res;
}

#define GET_MACRO2(_1, _2, NAME, ...) NAME
#define ensure1(cond) _ensure(cond, __FILE__, __LINE__, __PRETTY_FUNCTION__, #cond, NULL)
#define ensure2(cond, err) _ensure(cond, __FILE__, __LINE__, __PRETTY_FUNCTION__, #cond, err)
#define ensure(...) GET_MACRO2(__VA_ARGS__, ensure2, ensure1)(__VA_ARGS__)

TTi void _ensure(T cond, const char *file, int lno, const char *func, const char *condstr, const char * err) {
    if (!cond) {
        if (err) {
            fprintf(stderr, "error: %s\n", err);
        }
        fprintf(stderr, "at %s:%d %s (in %s)\n", file, lno, condstr, func);
        perror("libc");
        exit(1);
    }
}