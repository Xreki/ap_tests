#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export AP_CUTLASS_DIR=/work/Paddle/third_party/cutlass
export FLAGS_prim_all=True
export FLAGS_prim_enable_dynamic=true
export FLAGS_use_cinn=1

# matmul 单测脚本
TESTS=(
    test_matmul_add_relu.py
    test_matmul_add_gelu.py
    test_matmul_add_multiply.py
    test_matmul_add_divide_multiply.py
    test_matmul_add_divide_multiply_add.py
)
BATCH_SIZES=(1 8 32) # 需要遍历的 batch_size 列表
LOG_DIR=logs # 日志输出目录
mkdir -p "${LOG_DIR}"

echo "========================================================================"
echo " AP Fusion Test Suite"
echo " Start time : $(date '+%Y-%m-%d %H:%M:%S')"
echo " Tests      : ${#TESTS[@]}"
echo " Batch sizes: ${BATCH_SIZES[*]}"
echo " Log dir    : ${LOG_DIR}"
echo "========================================================================"
suite_start=$(date +%s)
for test in "${TESTS[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
        log_file="${LOG_DIR}/${test%.py}_bs${bs}.txt"
        case_start=$(date +%s)
        python tests/${test} --batch_size ${bs} > ${log_file} 2>&1
        status=$?
        case_end=$(date +%s)
        echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] Finished tests/${test} with batch_size=${bs} in $((case_end - case_start))s, exit code ${status}, log: ${log_file}"
    done
done
suite_end=$(date +%s)

echo "========================================================================"
echo " End time      : $(date '+%Y-%m-%d %H:%M:%S')"
echo " Total elapsed : $((suite_end - suite_start))s"
echo "========================================================================"
