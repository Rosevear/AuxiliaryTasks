#!/bin/bash
echo "Running auxiliary task experiments..."
echo "Running auxiliary experiments for deterministic gridworld with rich rewards..."
python grid_exp.py --sweep --name "rich deterministic gridworld"
echo "Running auxiliary experiments for deterministic gridworld with sparse rewards..."
python grid_exp.py --sweep --sparse --name "sparse deterministic gridworld"
echo "Running auxiliary experiments for stochastic gridworld with rich rewards..."
python grid_exp.py --sweep --stochastic --name "stochastic rich gridworld"
echo "Running auxiliary experiments for stochastic gridworld with sparse rewards..."
python grid_exp.py --sweep --stochastic --sparse --name "stochastic sparse gridworld"
echo "All experiments completed!"
mkdir Results
mkdir Results/model_summaries
mv *.txt Results/model_summaries
mv *.png Results
