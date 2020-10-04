#!/bin/sh

# set mode
MODE="resume"

# run toy data experiments
python generative_experiments.py --dataset "fashion_mnist" --mode $MODE
python generative_experiments.py --dataset "mnist" --mode $MODE

# print all done
echo "vae experiments done!"

# run analysis scripts
python generative_analysis.py
