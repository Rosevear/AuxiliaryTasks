#!/bin/bash
echo "Running auxiliary task experiments..."
echo "Running auxiliary experiments for deterministic gridworld with rich rewards..."
python grid_exp.py --sweep --name "rich deterministic gridworld coord"
echo "Running auxiliary experiments for deterministic gridworld with sparse rewards..."
python grid_exp.py --sweep --sparse --name "sparse deterministic gridworld coord"
echo "Running auxiliary experiments for stochastic gridworld with rich rewards..."
python grid_exp.py --sweep --stochastic --name "stochastic rich gridworld coord"
echo "Running auxiliary experiments for stochastic gridworld with sparse rewards..."
python grid_exp.py --sweep --stochastic --sparse --name "stochastic sparse gridworld coord"
echo "All experiments completed!"
mkdir Results
mkdir Results/model_summaries
mv *coord.txt Results/model_summaries
mv *coord.png Results
