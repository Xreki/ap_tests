#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

# https://github.com/PaddlePaddle/Paddle/pull/79083 添加,通过环境变量配置 AP 使用的 cutlass 路径
export AP_CUTLASS_DIR=/work/Paddle/third_party/cutlass

export FLAGS_prim_all=True
export FLAGS_prim_enable_dynamic=true
export FLAGS_use_cinn=1

# 5 个单测脚本
TESTS=(
    test_matmul_add_relu.py
    test_matmul_add_gelu.py
    test_matmul_add_multiply.py
    test_matmul_add_divide_multiply.py
    test_matmul_add_divide_multiply_add.py
)

# 需要遍历的 batch_size 列表
BATCH_SIZES=(1 8 32)

# 日志输出目录
LOG_DIR=logs
mkdir -p "${LOG_DIR}"

for test in "${TESTS[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
        log_file="${LOG_DIR}/${test%.py}_bs${bs}.txt"
        echo ">>> Running tests/${test} with batch_size=${bs}"
        echo ">>> Log: ${log_file}"
        echo
        python tests/${test} --batch_size ${bs} > ${log_file} 2>&1
    done
done
