#!/bin/bash

# 定义测试数据路径数组
test_data_paths=(
    "/home/hzx/Projects/Data/BTAD"
    "/home/hzx/Projects/Data/MPDD"
    "/home/hzx/Projects/Data/SDD"
    "/home/hzx/Projects/Data/SDD-unboxes"
    "/home/hzx/Projects/Data/DTD-Synthetic"
)

# 定义seed和k-shot的值
seeds=(42 2020 2023 2024 2025)
k_shots=(1 2 4)

# 定义重复次数
repetitions=2

# 循环遍历每个参数组合
for test_data_path in "${test_data_paths[@]}"; do
    for seed in "${seeds[@]}"; do
        for k_shot in "${k_shots[@]}"; do
            for (( repetition=0; repetition<$repetitions; repetition++ )); do
                echo "Running repetition $repetition with seed=$seed, k-shot=$k_shot, test_data_path=$test_data_path"
                # 替换这里的命令为你的完整命令
                python src/eval.py trainer=gpu data=anomaly_clip_infer model=anomaly_clip_swin data.data_dir.test=$test_data_path seed=$seed data.dataset.kshot.k_shot=$k_shot
                echo "Completed repetition $repetition with seed=$seed, k-shot=$k_shot, test_data_path=$test_data_path"
            done
        done
    done
done