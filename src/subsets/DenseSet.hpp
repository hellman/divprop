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

    // ========================================
    // Basic
    // ========================================
    DenseSet & operator|=(const DenseSet & b);
    DenseSet & operator^=(const DenseSet & b);
    DenseSet & operator&=(const DenseSet & b);
    DenseSet & operator-=(const DenseSet & b);

    void iter_support(function<void(uint64_t)> const & func) const;

    std::vector<uint64_t> get_support() const;

    uint64_t get_weight() const;

    std::vector<uint64_t> get_counts_by_weight() const;

    DenseSet get_head_fixed(int h, uint64_t value);

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

    void save_to_file(const char *filename) const;
    static DenseSet load_from_file(const char *filename);
    uint64_t get_hash() const;
    void log_info(const char *name) const;
};