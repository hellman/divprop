%module(package="subsets") lib

%include "std_vector.i"
%include "stdint.i"

%typedef uint64_t u64;
%typedef uint32_t u32;
%typedef uint16_t u16;
%typedef uint8_t u8;

%typedef int64_t i64;
%typedef int32_t i32;
%typedef int16_t i16;
%typedef int8_t i8;

%template(MyVector_u64) std::vector<uint64_t>;
%template(MyVector_u8) std::vector<uint8_t>;
%template(MyVector_int) std::vector<int>;

%include "DenseSet.hpp"

%{
#include "DenseSet.hpp"
%}