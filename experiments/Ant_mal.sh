#!/bin/bash

# Function to train a model with a given configuration
commit_changes () {
    learning_curve_path=$1
    policy_path=$2
    git add $learning_curve_path $policy_path
    git commit -m "Updated models on $(date)"
}


train_model () {
    config_path=$1
    script=$2
    data=$3
    model=$4
    agent=$5
    iters=$6
    for i in $(seq 1 $iters)
    do
      echo "Training run $i for configuration: $config_path"
      /home/pearl0/miniconda3/envs/MMJC-maddpg/bin/python $script --config $config_path --train True --mal_agent $agent
    done
    commit_changes $data $model
    git push origin UNITYxMaMuJuCo
}

# Training models with different configurations
#train_model ./configs/ant_config_4.yaml ./Training/train_mujuco.py ./learning_curves/Ant.2x4.0.001.350.0.99/ ./tmp/policy/Ant.2x4.0.001.350.0.99/ 0
# Train malfunction
train_model ./configs/ant_config_4.yaml ./Training/train_mujuco_malfunction.py ./learning_curves/Ant.2x4.0.001.350.0.99/malfunction/ ./tmp/policy/Ant.2x4.0.001.350.0.99malfunction/ 0 2
#train_model ./configs/ant_config_4.yaml ./Training/train_mujuco_malfunction.py ./learning_curves/Ant.2x4.0.001.350.0.99/malfunction/ ./tmp/policy/Ant.2x4.0.001.350.0.99malfunction/ 2 4

