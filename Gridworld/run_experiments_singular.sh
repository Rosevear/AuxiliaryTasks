#!/bin/bash
echo "Running auxiliary task experiments..."
echo "Running auxiliary experiments for deterministic gridworld with rich rewards..."
python grid_exp.py --name "rich deterministic gridworld coord singular"
echo "Running auxiliary experiments for deterministic gridworld with sparse rewards..."
python grid_exp.py --sparse --name "sparse deterministic gridworld coord singular"
echo "Running auxiliary experiments for stochastic gridworld with rich rewards..."
python grid_exp.py --stochastic --name "stochastic rich gridworld coord singular"
echo "Running auxiliary experiments for stochastic gridworld with sparse rewards..."
python grid_exp.py --stochastic --sparse --name "stochastic sparse gridworld coord singular"
echo "All experiments completed!"
mkdir ResultsSingular
mkdir ResultsSingular/model_summaries
mv *coord.txt ResultsSingular/model_summaries
mv *coord.png ResultsSingular
