#pragma once

#include "common.hpp"

#ifdef SWIG
%pythoncode %{
from binteger import Bin
%}
#endif

/*
Working with single bit vector, represented as vector<uint64_t>.
*/
struct DenseSet {
    int n; // n input bits
    std::vector<uint64_t> data; // n / 64 words of 64 bits

    DenseSet();
    DenseSet(int _n);
    DenseSet(const std::vector<uint64_t> &ints, int n);
    DenseSet(const std::unordered_set<uint64_t> &ints, int n);

    void resize(int n);

    DenseSet copy() const;
    void free(); // set to empty set with n=0

    void clear(); // set to empty set, keep n
    void fill(); // set to full set, keep n

    bool is_empty() const;
    bool __bool__() const;
    bool is_full() const;

    static const uint64_t VERSION1 = 0xf1c674e0bf03fea6ull;
    static const uint64_t MARKER_END = 0xc6891a2b5f8bb0b7ull;
    static bool QUIET;  // set to true to disable stderr printing

    static void set_quiet(bool value=true);

    // ========================================
    // Read/Write & info
    // ========================================
    void save_to_file(const char *filename) const;
    static DenseSet load_from_file(const char *filename);
    uint64_t get_hash() const;
    std::string info() const;
    std::string __str__() const;

    // ========================================
    // Single bit get/set
    // ========================================
    int get(uint64_t x) const;
    void set(uint64_t x);
    void set(uint64_t x, uint64_t value);

    // python set style
    bool __contains__(uint64_t x) const;
    void add(uint64_t x);
    void remove(uint64_t x);
    void discard(uint64_t x);

    // ========================================
    // Tools
    // ========================================
    #ifndef SWIG
    template<auto func>
    void do_Sweep(uint64_t mask = -1ull);
    #endif
    // for python low-level API
    void do_Sweep_OR_up(uint64_t mask = -1ull);
    void do_Sweep_OR_down(uint64_t mask = -1ull);
    void do_Sweep_XOR_up(uint64_t mask = -1ull);
    void do_Sweep_XOR_down(uint64_t mask = -1ull);
    void do_Sweep_AND_up(uint64_t mask = -1ull);
    void do_Sweep_AND_down(uint64_t mask = -1ull);
    void do_Sweep_SWAP(uint64_t mask = -1ull);
    void do_Sweep_LESS_up(uint64_t mask = -1ull);
    void do_Sweep_MORE_down(uint64_t mask = -1ull);

    // ========================================
    // Bitwise ops
    // ========================================
    bool is_compatible_set(const DenseSet & b) const;
    bool operator==(const DenseSet & b) const;
    bool operator!=(const DenseSet & b) const;
    bool operator<(const DenseSet & b) const;
    bool operator<=(const DenseSet & b) const;
    bool operator>(const DenseSet & b) const;
    bool operator>=(const DenseSet & b) const;

    DenseSet & operator|=(const DenseSet & b);
    DenseSet & operator^=(const DenseSet & b);
    DenseSet & operator&=(const DenseSet & b);
    DenseSet & operator-=(const DenseSet & b);
    DenseSet operator|(const DenseSet & b) const;
    DenseSet operator^(const DenseSet & b) const;
    DenseSet operator&(const DenseSet & b) const;
    DenseSet operator-(const DenseSet & b) const;
    // DenseSet operator~() const;

    DenseSet get_head_fixed(int h, uint64_t value);

    // ========================================
    // Support
    // ========================================
    #ifndef SWIG
    void iter_support(function<void(uint64_t)> const & func) const;
    #endif

    std::vector<uint64_t> get_support() const;
    #ifdef SWIG
    %pythoncode %{
        def __iter__(self):
            return iter(self.get_support())

        def to_Bins(self):
            n = int(self.n)
            return [Bin(v, n) for v in self]
    %}
    #endif

    uint64_t get_weight() const;
    uint64_t __len__() const;

    std::vector<uint64_t> get_counts_by_weights() const;
    std::map<std::pair<int,int>,u64> get_counts_by_weight_pairs(int n1, int n2) const;
    std::string str_stat_by_weights() const;
    std::string str_stat_by_weight_pairs(int n1, int n2) const;

    // ========================================
    // Main methods
    // ========================================
    void do_UnsetUp(uint64_t mask = -1ull);
    void do_UnsetDown(uint64_t mask = -1ull);
    void do_SetUp(uint64_t mask = -1ull);
    void do_SetDown(uint64_t mask = -1ull);

    void do_Mobius(uint64_t mask = -1ull);
    void do_Complement();
    void do_Not(uint64_t mask = -1ull);
    void do_UpperSet(uint64_t mask = -1ull);
    void do_LowerSet(uint64_t mask = -1ull);
    void do_MinSet(uint64_t mask = -1ull);
    void do_MaxSet(uint64_t mask = -1ull);
    void do_DivCore(uint64_t mask = -1ull);
    void do_ComplementU2L(bool is_upper=false, uint64_t mask = -1ull);
    void do_ComplementL2U(bool is_lower=false, uint64_t mask = -1ull);
    void do_UpperSet_Up1(bool is_minset=false, uint64_t mask = -1ull);

    DenseSet Mobius(uint64_t mask = -1ull) const;
    DenseSet Complement() const;
    DenseSet Not(uint64_t mask = -1ull) const;
    DenseSet UpperSet(uint64_t mask = -1ull) const;
    DenseSet LowerSet(uint64_t mask = -1ull) const;
    DenseSet MinSet(uint64_t mask = -1ull) const;
    DenseSet MaxSet(uint64_t mask = -1ull) const;
    DenseSet DivCore(uint64_t mask = -1ull) const;
    DenseSet ComplementU2L(bool is_upper=false, uint64_t mask = -1ull) const;
    DenseSet ComplementL2U(bool is_lower=false, uint64_t mask = -1ull) const;
    DenseSet UpperSet_Up1(bool is_minset=false, uint64_t mask = -1ull) const;
};