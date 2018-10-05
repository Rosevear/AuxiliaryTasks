#!/bin/bash
echo "Running auxiliary task experiments..."
echo "Running auxiliary experiments for deterministic gridworld with rich rewards..."
python grid_exp.py  -run 50 --name "rich deterministic gridworld hot singular" --hot
echo "Running auxiliary experiments for deterministic gridworld with sparse rewards..."
python grid_exp.py  -run 50 --name "sparse deterministic gridworld hot singular" --hot
echo "Running auxiliary experiments for stochastic gridworld with rich rewards..."
python grid_exp.py -run 50 --stochastic --name "stochastic rich gridworld hot singular" --hot
echo "Running auxiliary experiments for stochastic gridworld with sparse rewards..."
python grid_exp.py -run 50 --stochastic --sparse --name "stochastic sparse gridworld hot singular" --hot
echo "All experiments completed!"
mkdir ResultsHotSingular
mkdir ResultsHotSingular/model_summaries
mv *hot.txt ResultsHotSingular/model_summaries
mv *hot.png ResultsHotSingular
