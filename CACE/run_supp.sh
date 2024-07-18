#!/bin/bash

PIDS=()

kill_processes() {
  echo "Caught SIGINT signal. Killing all processes..."
  kill -9 "${PIDS[@]}"
  exit 1
}

trap 'kill_processes' INT

run_command() {
  python supv_main.py $@ &
  PID=$!
  PIDS+=($PID)
}


# 损失函数不同系数
CUDA_VISIBLE_DEVICES=1 run_command --gpu 1 --lr 0.0007 --clip_gradient 0.5 --snapshot_pref "./Exps/Supv_supp/expLoss" --n_epoch 200 --b 64 --test_batch_size 64 --print_freq 10 --seed 3917 --guide Co-Guide --psai 0.3 --contrastive --Lambda 0.6 --contras_coeff 1.0

CUDA_VISIBLE_DEVICES=5 run_command --gpu 1 --lr 0.0007 --clip_gradient 0.5 --snapshot_pref "./Exps/Supv_supp/expLoss" --n_epoch 200 --b 64 --test_batch_size 64 --print_freq 10 --seed 3917 --guide Co-Guide --psai 0.3 --contrastive --Lambda 0.6 --contras_coeff 2.0

CUDA_VISIBLE_DEVICES=6 run_command --gpu 1 --lr 0.0007 --clip_gradient 0.5 --snapshot_pref "./Exps/Supv_supp/expLoss" --n_epoch 200 --b 64 --test_batch_size 64 --print_freq 10 --seed 3917 --guide Co-Guide --psai 0.3 --contrastive --Lambda 0.6 --contras_coeff 3.0

CUDA_VISIBLE_DEVICES=7 run_command --gpu 1 --lr 0.0007 --clip_gradient 0.5 --snapshot_pref "./Exps/Supv_supp/expLoss" --n_epoch 200 --b 64 --test_batch_size 64 --print_freq 10 --seed 3917 --guide Co-Guide --psai 0.3 --contrastive --Lambda 0.6 --contras_coeff 4.0

CUDA_VISIBLE_DEVICES=0 run_command --gpu 1 --lr 0.0007 --clip_gradient 0.5 --snapshot_pref "./Exps/Supv_supp/expLoss" --n_epoch 200 --b 64 --test_batch_size 64 --print_freq 10 --seed 3917 --guide Co-Guide --psai 0.3 --contrastive --Lambda 0.6 --contras_coeff 5.0