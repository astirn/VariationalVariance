#!/bin/sh

# download data
python regression_data.py

# set mode
MODE="resume"

# number of parallel jobs
N=4

# dataset iterators
declare -a Datasets=("boston" "carbon" "concrete" "energy" "naval" "power plant" "superconductivity" "wine-red" "wine-white" "yacht")

# MLE algorithms
declare -a MaximumLikelihoodAlgorithms=("Normal" "Student" "Detlefsen")

# Bayesian algorithms and priors
declare -a BayesianAlgorithms=("Gamma-Normal") # "LogNormal-Normal")
declare -a PriorTypes=("MLE" "VAMP" "VAMP*" "xVAMP" "xVAMP*" "VBEM" "VBEM*")

# loop over datasets
for data in "${Datasets[@]}"; do

  # loop over MLE algorithms
  for alg in "${MaximumLikelihoodAlgorithms[@]}"; do

    # run jobs in parallel if specified
    if [ $N -gt 1 ]; then
      python regression_experiments_v2.py --dataset "$data" --algorithm $alg --mode $MODE  --parallel 1 &

    # otherwise, run job in foreground
    else
      python regression_experiments_v2.py --dataset "$data" --algorithm $alg --mode $MODE  --parallel 0
    fi

    # check/wait for maximum jobs
    if [[ $(jobs -r -p | wc -l) -ge $N ]]; then
      wait -n
    fi
  done

  # loop over Bayesian algorithms
  for alg in "${BayesianAlgorithms[@]}"; do
    for prior in "${PriorTypes[@]}"; do

      # run jobs in parallel if specified
      if [ $N -gt 1 ]; then
        python regression_experiments_v2.py --dataset "$data" --algorithm $alg --prior_type $prior --mode $MODE \
          --k 100 --parallel 1 &

      # otherwise, run job in foreground
      else
        python regression_experiments_v2.py --dataset "$data" --algorithm $alg --prior_type $prior --mode $MODE \
          --k 100 --parallel 0
      fi

      # check/wait for maximum jobs
      if [[ $(jobs -r -p | wc -l) -ge $N ]]; then
        wait -n
      fi
    done
  done
done

# wait for all jobs to finish
wait

# print all done!
echo "UCI done!"

# run analysis scripts
python regression_analysis.py --experiment uci
