#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# https://github.com/PaddlePaddle/Paddle/pull/79083 添加,通过环境变量配置 AP 使用的 cutlass 路径
export AP_CUTLASS_DIR=/work/Paddle/third_party/cutlass

export FLAGS_prim_all=True
export FLAGS_prim_enable_dynamic=true
export FLAGS_use_cinn=1

# 测试类型：matmul 或 conv2d，可通过第一个参数指定，默认为 matmul
TEST_TYPE="${1:-matmul}"

case "${TEST_TYPE}" in
    matmul)
        TESTS=(
            test_matmul_add_relu.py
            test_matmul_add_gelu.py
            test_matmul_add_multiply.py
            test_matmul_add_divide_multiply.py
            test_matmul_add_divide_multiply_add.py
        )
        # matmul 需要遍历的 batch_size 列表
        BATCH_SIZES=(1 8 32)
        ;;
    conv2d)
        TESTS=(
            test_conv2d_add_relu.py
            test_conv2d_add_bias_residual_relu.py
            test_conv2d_leaky_relu.py
        )
        # conv2d 需要遍历的 batch_size 列表
        BATCH_SIZES=(1 8 32)
        ;;
    *)
        echo "Unknown test type: ${TEST_TYPE}"
        echo "Usage: $0 [matmul|conv2d]"
        exit 1
        ;;
esac

# 日志输出目录
LOG_DIR=logs
mkdir -p "${LOG_DIR}"

echo "========================================================================"
echo " AP Fusion Test Suite"
echo " Test type  : ${TEST_TYPE}"
echo " Start time : $(date '+%Y-%m-%d %H:%M:%S')"
echo " Tests      : ${#TESTS[@]}"
echo " Batch sizes: ${BATCH_SIZES[*]}"
echo " Log dir    : ${LOG_DIR}"
echo "========================================================================"
echo

suite_start=$(date +%s)

for test in "${TESTS[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
        log_file="${LOG_DIR}/${test%.py}_bs${bs}.txt"
        echo "------------------------------------------------------------------------"
        echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] Running tests/${test} with batch_size=${bs}"
        echo ">>> Log: ${log_file}"
        case_start=$(date +%s)
        python tests/${test} --batch_size ${bs} > ${log_file} 2>&1
        status=$?
        case_end=$(date +%s)
        echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] Finished in $((case_end - case_start))s, exit code ${status}"
        echo
    done
done

suite_end=$(date +%s)

echo "========================================================================"
echo " End time      : $(date '+%Y-%m-%d %H:%M:%S')"
echo " Total elapsed : $((suite_end - suite_start))s"
echo "========================================================================"

echo
echo "========================================================================"
echo " Speedup Summary"
echo "========================================================================"
python tools/extract_perf.py "${LOG_DIR}"
