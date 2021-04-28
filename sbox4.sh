#!/bin/bash -xef

for s in `cat $1`; do
	divprop.sbox2ddt "$s"
done

for s in `cat $1`; do
	#python src/optimodel/tool_milp.py "data/sbox_${s}/ddt" AutoChain ShiftLearn:threads=7
	python src/optimodel/tool_milp.py "data/sbox_${s}/ddt" SubsetMILP:solver=scip
done