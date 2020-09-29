#!/bin/sh

# set mode
MODE="resume"

# run toy data experiments
python regression_experiments_v2.py --algorithm "Detlefsen" --dataset toy --mode $MODE
python regression_experiments_v2.py --algorithm "Detlefsen (fixed)" --dataset toy --mode $MODE
python regression_experiments_v2.py --algorithm "Normal" --dataset toy --mode $MODE
python regression_experiments_v2.py --algorithm "Student" --dataset toy --mode $MODE
declare -a Algorithms=("Gamma-Normal")
declare -a PriorTypes=("MLE" "Standard" "VAMP" "VAMP*" "xVAMP" "xVAMP*" "VBEM" "VBEM*")
for alg in "${Algorithms[@]}"; do
  for prior in "${PriorTypes[@]}"; do
    python regression_experiments_v2.py --dataset toy --algorithm $alg --prior_type $prior --mode $MODE --k 20
  done
done

# print all done
echo "toy experiments done!"

# run analysis scripts
python regression_analysis.py --experiment toy
