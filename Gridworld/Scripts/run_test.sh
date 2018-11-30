#!/bin/bash
screen -S 'TEST' python ../grid_exp.py -name "TEST" -run 1 -epi 2 -max 1 --sweep_neural --q_plot -env 'continuous_grid'
