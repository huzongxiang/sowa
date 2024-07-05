#!/bin/bash

# 定义数据集、k-shot、seed和重复次数的值
datasets=("anomaly_clip_mvt" "anomaly_clip_visa")
k_shots=(1 2 4)
seeds=(42)
repetitions=1

# 条件判断ckpt_path的值
ckpt_path_mvt="/home/hzx/Projects/Weight/mvtech-kshot/learnable/0-shot/2024-05-27_13-08-55/checkpoints/epoch_000.ckpt"
ckpt_path_visa="/home/hzx/Projects/Weight/visa-kshot/learnable/0-shot/2024-05-27_11-07-16/checkpoints/epoch_000.ckpt"

# 循环遍历每个参数组合
for dataset in "${datasets[@]}"; do
    if [[ "$dataset" == "anomaly_clip_mvt" ]]; then
        ckpt_path=$ckpt_path_mvt
    elif [[ "$dataset" == "anomaly_clip_visa" ]]; then
        ckpt_path=$ckpt_path_visa
    fi

    for k_shot in "${k_shots[@]}"; do
        for seed in "${seeds[@]}"; do
            for (( i=0; i<repetitions; i++ )); do
                echo "Running repetition $i with data=$dataset, k-shot=$k_shot, seed=$seed, ckpt_path=$ckpt_path"
                python src/eval.py trainer=gpu data=$dataset model=anomaly_clip_swin data.dataset.kshot.k_shot=$k_shot seed=$seed ckpt_path=$ckpt_path
                echo "Completed repetition $i with data=$dataset, k-shot=$k_shot, seed=$seed, ckpt_path=$ckpt_path"
            done
        done
    done
done