#!/bin/bash
screen -S "Neural network sweep" python ../grid_exp.py -name "sweep 15 runs 150 episodes -epsilon decay 0.05 epislon min 0.01 batch size 9 freq 1000 50 neurons clip gradient 1" -run 15 -epi 150 -max 10000 --sweep_neural --q_plot -env 'continuous_grid'
