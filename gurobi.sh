#!/bin/bash -ex

echo "$1"_* "$1".*
echo "Remove?"
read qwe

rm -f "$1"_* "$1".*

# gurobi_cl "MIPFocus=3" "ResultFile=$1.sol" "SolFiles=$1" "LogFile=$1.log" "$1"
#gurobi_cl "ObjBound=1899" "ResultFile=$1.sol" "SolFiles=$1" "LogFile=$1.log" "$1"
gurobi_cl "ResultFile=$1.sol" "SolFiles=$1" "LogFile=$1.log" "$1"