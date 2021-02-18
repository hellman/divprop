#pragma once

#include "common.hpp"

/*
Working with single bit vector, represented as vector<uint64_t>.
*/

struct DenseSet {
    int n; // n input bits
    std::vector<uint64_t> data; // n / 64 words of 64 bits

    DenseSet();
    DenseSet(int _n);
    DenseSet copy() const;
    void free();
    void reset();

    // ========================================
    // Read/Write & info
    // ========================================
    void save_to_file(const char *filename) const;
    static DenseSet load_from_file(const char *filename);
    uint64_t get_hash() const;
    std::string info(const char *name = NULL) const;
    void log_info(const char *name = NULL) const;

    // ========================================
    // Single bit get/set
    // ========================================
    int get(uint64_t x) const;
    void set(uint64_t x);
    void set(uint64_t x, uint64_t value);

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
    DenseSet & operator|=(const DenseSet & b);
    DenseSet & operator^=(const DenseSet & b);
    DenseSet & operator&=(const DenseSet & b);
    DenseSet & operator-=(const DenseSet & b);

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
    %}
    #endif

    uint64_t get_weight() const;
    uint64_t __len__() const;

    std::vector<uint64_t> get_counts_by_weight() const;

    // ========================================
    // Main methods
    // ========================================
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
};