#!/bin/sh

# download data
python regression_data.py

# set mode
MODE="resume"

# number of parallel jobs
N=4

# run UCI regression experiments
declare -a BatchIterations=(10000 200000)
declare -a Datasets=("boston" "carbon" "concrete" "energy" "naval" "power plant" "superconductivity" "wine-red" "wine-white" "yacht")
declare -a Algorithms=("Gamma-Normal" "LogNormal-Normal")
declare -a PriorTypes=("MLE" "VAMP" "VAMP*" "xVAMP" "xVAMP*" "VBEM" "VBEM*")
for iters in "${BatchIterations[@]}"; do
  for data in "${Datasets[@]}"; do
    python regression_experiments_v2.py --dataset "$data" --algorithm "Detlefsen" --batch_iterations $iters --mode $MODE
    for alg in "${Algorithms[@]}"; do
      for prior in "${PriorTypes[@]}"; do

        if [ $N -gt 1 ]; then
          # run jobs in parallel
          python regression_experiments_v2.py --dataset "$data" --algorithm $alg --prior_type $prior --mode $MODE \
            --batch_iterations $iters --k 100 --parallel 1 &

          # allow N jobs in parallel
          if [[ $(jobs -r -p | wc -l) -ge $N ]]; then
            wait -n
          fi
        else
          # run single job
          python regression_experiments_v2.py --dataset "$data" --algorithm $alg --prior_type $prior --mode $MODE \
            --batch_iterations $iters --k 100 --parallel 0
        fi
      done
    done
  done
done
wait
echo "UCI done!"
