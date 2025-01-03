#!/bin/bash

# Activate the virtual environment
source ~/ocr/parseq/.env/bin/activate

# Change directory to the project folder
cd ~/rl/trpo-pytorch

# Run the training script with nohup to keep it running after the terminal is closed
nohup python train.py --model-name hopper --ver diagonal_fisher --seed 58 & exit
