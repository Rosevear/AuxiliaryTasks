#!/bin/bash
echo "Running auxiliary task experiments..."
echo "Running auxiliary experiments for deterministic gridworld with rich rewards..."
python grid_exp.py --sweep --name "rich deterministic gridworld hot" --hot
echo "Running auxiliary experiments for deterministic gridworld with sparse rewards..."
python grid_exp.py --sweep --sparse --name "sparse deterministic gridworld hot" --hot
echo "Running auxiliary experiments for stochastic gridworld with rich rewards..."
python grid_exp.py --sweep --stochastic --name "stochastic rich gridworld hot" --hot
echo "Running auxiliary experiments for stochastic gridworld with sparse rewards..."
python grid_exp.py --sweep --stochastic --sparse --name "stochastic sparse gridworld hot" --hot
echo "All experiments completed!"
mkdir ResultsHot
mkdir ResultsHot/model_summaries
mv *hot.txt ResultsHot/model_summaries
mv *hot.png ResultsHot
