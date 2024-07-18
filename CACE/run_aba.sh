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


# 验证不引导、单向引导、双向引导
CUDA_VISIBLE_DEVICES=0 run_command --gpu 1 --lr 0.0007 --clip_gradient 0.5 --snapshot_pref "./Exps/Supv/expGuide" --n_epoch 200 --b 64 --test_batch_size 64 --print_freq 10 --seed 3917 --guide None --psai 0.3

CUDA_VISIBLE_DEVICES=1 run_command --gpu 1 --lr 0.0007 --clip_gradient 0.5 --snapshot_pref "./Exps/Supv/expGuide" --n_epoch 200 --b 64 --test_batch_size 64 --print_freq 10 --seed 3917 --guide Audio-Guide --psai 0.3

CUDA_VISIBLE_DEVICES=2 run_command --gpu 1 --lr 0.0007 --clip_gradient 0.5 --snapshot_pref "./Exps/Supv/expGuide" --n_epoch 200 --b 64 --test_batch_size 64 --print_freq 10 --seed 3917 --guide Visual-Guide --psai 0.3

CUDA_VISIBLE_DEVICES=3 run_command --gpu 1 --lr 0.0007 --clip_gradient 0.5 --snapshot_pref "./Exps/Supv/expGuide" --n_epoch 200 --b 64 --test_batch_size 64 --print_freq 10 --seed 3917 --guide Co-Guide --psai 0.3



# 验证双向引导中, psai的系数
CUDA_VISIBLE_DEVICES=0 run_command --gpu 1 --lr 0.0007 --clip_gradient 0.5 --snapshot_pref "./Exps/Supv/expPsai" --n_epoch 200 --b 64 --test_batch_size 64 --print_freq 10 --seed 3917 --guide Co-Guide --psai 0.0

CUDA_VISIBLE_DEVICES=1 run_command --gpu 1 --lr 0.0007 --clip_gradient 0.5 --snapshot_pref "./Exps/Supv/expPsai" --n_epoch 200 --b 64 --test_batch_size 64 --print_freq 10 --seed 3917 --guide Co-Guide --psai 0.15

CUDA_VISIBLE_DEVICES=2 run_command --gpu 1 --lr 0.0007 --clip_gradient 0.5 --snapshot_pref "./Exps/Supv/expPsai" --n_epoch 200 --b 64 --test_batch_size 64 --print_freq 10 --seed 3917 --guide Co-Guide --psai 0.3

CUDA_VISIBLE_DEVICES=3 run_command --gpu 1 --lr 0.0007 --clip_gradient 0.5 --snapshot_pref "./Exps/Supv/expPsai" --n_epoch 200 --b 64 --test_batch_size 64 --print_freq 10 --seed 3917 --guide Co-Guide --psai 0.45


# 验证对比学习中, lambda(lmbda)的系数
CUDA_VISIBLE_DEVICES=0 run_command --gpu 1 --lr 0.0007 --clip_gradient 0.5 --snapshot_pref "./Exps/Supv/expEta" --n_epoch 200 --b 64 --test_batch_size 64 --print_freq 10 --seed 3917 --guide Co-Guide --psai 0.3 --contrastive --Lambda 0.2

CUDA_VISIBLE_DEVICES=1 run_command --gpu 1 --lr 0.0007 --clip_gradient 0.5 --snapshot_pref "./Exps/Supv/expEta" --n_epoch 200 --b 64 --test_batch_size 64 --print_freq 10 --seed 3917 --guide Co-Guide --psai 0.3 --contrastive --Lambda 0.4

CUDA_VISIBLE_DEVICES=2 run_command --gpu 1 --lr 0.0007 --clip_gradient 0.5 --snapshot_pref "./Exps/Supv/expEta" --n_epoch 200 --b 64 --test_batch_size 64 --print_freq 10 --seed 3917 --guide Co-Guide --psai 0.3 --contrastive --Lambda 0.6

CUDA_VISIBLE_DEVICES=3 run_command --gpu 1 --lr 0.0007 --clip_gradient 0.5 --snapshot_pref "./Exps/Supv/expEta" --n_epoch 200 --b 64 --test_batch_size 64 --print_freq 10 --seed 3917 --guide Co-Guide --psai 0.3 --contrastive --Lambda 0.8
