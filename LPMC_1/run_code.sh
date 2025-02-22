#!/bin/bash
centralized_model_name=("MNL" "ASU_DNN" "E_MNL" "L_MNL")
L1_rate=(1e-4 1e-3 0.01 0.1 1.0)
L2_rate=(1e-4 1e-3 0.01 0.1 1.0)
droprate=(0.5 0.3 0.1 0.01 0.001)
learning_rate=(0.001 0.002 0.005 0.01 0.02)
epoch_num=(100 200 400 800 1600)
# default hyperparameters
for num in {1..30}
do
 for i in ${centralized_model_name[*]}
  do
    python main_centralized_LPMC.py --model=$i --model_num=$num
  done;
  python main_FL_LPMC.py --model_num=$num
  python main_FM_LPMC.py --model_num=$num
done;
# L1 regularization
for num in {1..30}
do
  for L1 in ${L1_rate[*]}
  do
    for i in ${centralized_model_name[*]}
    do
      python main_centralized_LPMC.py --model=$i --model_num=$num --L1=$L1 --save_file=L1_$L1
    done;
    python main_FL_LPMC.py --model_num=$num --L1=$L1 --save_file=L1_$L1
    python main_FM_LPMC.py --model_num=$num --L1=$L1 --save_file=L1_$L1
  done;
done;
# L2 regularization
for num in {1..30}
do
  for L2 in ${L2_rate[*]}
  do
    for i in ${centralized_model_name[*]}
    do
      python main_centralized_LPMC.py --model=$i --model_num=$num --L2=$L2 --save_file=L2_$L2
    done;
    python main_FL_LPMC.py --model_num=$num --L2=$L2 --save_file=L2_$L2
    python main_FM_LPMC.py --model_num=$num --L2=$L2 --save_file=L2_$L2
  done;
done;
# dropout
for num in {1..30}
do
  for drop_rate in ${droprate[*]}
  do
    for i in ${centralized_model_name[*]}
    do
      python main_centralized_LPMC.py --model=$i --model_num=$num --dropout=$drop_rate --save_file=dropout_$drop_rate
    done;
    python main_FL_LPMC.py --model_num=$num --dropout=$drop_rate --save_file=dropout_$drop_rate
    python main_FM_LPMC.py --model_num=$num --dropout=$drop_rate  --save_file=dropout_$drop_rate
  done;
done;
# learning rate
for num in {1..30}
do
  for lr in ${learning_rate[*]}
  do
    for i in ${centralized_model_name[*]}
    do
      python main_centralized_LPMC.py --model=$i --model_num=$num --inner_lr=$lr --save_file=lr_$lr
    done;
    python main_FL_LPMC.py --model_num=$num --inner_lr=$lr --save_file=lr_$lr
    python main_FM_LPMC.py --model_num=$num --inner_lr=$lr --outer_lr=$lr --save_file=lr_$lr
  done;
done;
# epoch number
for num in {1..30}
do
  for epoch in ${epoch_num[*]}
  do
    for i in ${centralized_model_name[*]}
    do
      python main_centralized_LPMC.py --model=$i --model_num=$num --epochs=$epoch --save_file=epoch_$epoch
    done;
    python main_FL_LPMC.py --model_num=$num --epochs=$epoch --save_file=epoch_$epoch
    python main_FM_LPMC.py --model_num=$num --epochs=$epoch --save_file=epoch_$epoch
  done;
done;
