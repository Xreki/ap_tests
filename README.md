# AP 测试集

PaddlePaddle AP (Abstract Pass) 算子融合的功能与性能测试集。

每个用例构造一个 `matmul + 若干 elementwise/broadcast` 的子图，分别用以下两种方式执行并对比：

- **Eager**：PaddlePaddle 动态图逐算子执行；
- **AP**：通过 `paddle.incubate.cc.compile` 编译，将 matmul 的 epilogue（bias 及后续 elementwise/broadcast）融合进一个 CUTLASS kernel。

测试会校验两者的数值精度，并分别 benchmark 二者的 GPU 耗时、输出加速比。

## 目录结构

```
ap_tests/
├── run_test.sh                 # 批量运行脚本（遍历用例 + batch_size）
├── tests/
│   ├── test_ap_base.py         # 测试基类：编译、精度校验、benchmark、main 入口
│   ├── test_matmul_add_relu.py
│   ├── test_matmul_add_gelu.py
│   ├── test_matmul_add_multiply.py
│   ├── test_matmul_add_divide_multiply.py
│   └── test_matmul_add_divide_multiply_add.py
└── logs/                       # run_test.sh 生成的日志（每个用例 + batch_size 一个文件）
```

## 测试用例

各用例的融合子图（`x @ w + b` 之后的 epilogue）：

| 用例 | 融合计算 |  M | N | K |
| --- | --- | --- | --- | --- |
| `test_matmul_add_relu` | `relu(x @ w + b)` | 500 | 1024 | 256 |
| `test_matmul_add_gelu` | `gelu(x @ w + b)` | 1280 | 512 | 128 |
| `test_matmul_add_multiply` | `(x @ w + b) + z` | 784 | 192 | 768 |
| `test_matmul_add_divide_multiply` | `(x @ w + b) / e1 * e2` | 3136 | 96 | 384 |
| `test_matmul_add_divide_multiply_add` | `(x @ w + b) / e1 * e2 + e3` | 784 | 192 | 768 |

> 注：含除法的用例会对除数 `e1` 做 `abs(e1) + 1.0` 处理，避免 float16 下除数接近 0 导致溢出为 `inf`，进而造成精度比对失败。

### 正确性检查

正确性由 `APTestBase.check_accuracy(fused_func, foo, tensor_args)` 校验，分两步保证“确实发生了融合”且“融合结果正确”：

1. **融合校验**：编译后（`get_compiled_func`）检查生成的 PIR program 中是否包含 `pd_op.ap_variadic` 算子，若不存在则判定 AP 融合失败、用例直接报错。
2. **数值比对**：用同一组输入张量分别执行 Eager（`foo`）与 AP（`fused_func`），对每个输出用 `np.testing.assert_allclose(ap_out, eager_out, atol=1e-2, rtol=1e-2)` 逐元素比对。

### 性能统计方式

性能由 `test_ap_base.bench_device_time(fn, input_args)` 统计，Eager 与 AP 各跑一次，最终上报 GPU 耗时（单位 ms）。流程如下：

1. **预热（warmup）**：先执行 `warmup_iters=10` 次，再 `paddle.device.synchronize()`，排除首次编译 / 显存分配 / kernel 加载等冷启动开销。
2. **正式计时（repeat）**：连续执行 `repeat_iters=100` 次。
3. **优先取 GPU kernel 时间**：用 `paddle.profiler.Profiler`（同时采集 CPU 与 GPU）包裹正式计时区间，从生成的 “Overview Summary” 表中解析 `Kernel` 行的 GPU Time。该值为纯 GPU 计算耗时，**不含** CPU 调度与 Python 开销，是默认上报值。
4. **回退到端到端计时**：若 profiler 未能解析出 kernel 时间，则回退为 CUDA Event 计时——逐次用 `cuda.Event` 的 `start/end` 包裹单次执行并 `synchronize`，取 100 次平均；且每次计时前调用 `_flush_l2_cache()`（分配一块 64MB 的 buffer）冲刷 L2 cache，避免缓存命中导致访存耗时被低估。

加速比定义：

```
Speedup = EagerTime / APTime
```

## 测试结果
### CUDA A100 测试结果

依赖环境：

- 编译了 CUDA、CINN 与 AP 功能的 PaddlePaddle；
- 可用的 NVIDIA GPU；
- CUTLASS 源码（通过环境变量 `AP_CUTLASS_DIR` 指定）。

以下为 `logs/cuda/` 目录下日志的汇总（dtype 为 float16，时间为 benchmark 得到的 GPU 耗时，越小越好；`Speedup = EagerTime / APTime`）。各用例的基础 shape（`B` 由 `--batch_size` 覆盖）：

| 用例 | batch_size | 精度 | EagerTime (ms) | APTime (ms) | Speedup |
| --- | --- | --- | --- | --- | --- |
| `test_matmul_add_relu` | 1 | PASS | 1.48 | 0.82 | 1.80x |
| `test_matmul_add_relu` | 8 | PASS | 4.49 | 2.59 | 1.73x |
| `test_matmul_add_relu` | 32 | PASS | 15.30 | 7.53 | 2.03x |
| `test_matmul_add_gelu` | 1 | PASS | 1.52 | 0.88 | 1.73x |
| `test_matmul_add_gelu` | 8 | PASS | 5.05 | 2.94 | 1.72x |
| `test_matmul_add_gelu` | 32 | PASS | 20.14 | 11.01 | 1.83x |
| `test_matmul_add_multiply` | 1 | PASS | 1.40 | 0.88 | 1.59x |
| `test_matmul_add_multiply` | 8 | PASS | 2.82 | 2.40 | 1.18x |
| `test_matmul_add_multiply` | 32 | PASS | 8.16 | 7.74 | 1.05x |
| `test_matmul_add_divide_multiply` | 1 | PASS | 1.77 | 1.10 | 1.61x |
| `test_matmul_add_divide_multiply` | 8 | PASS | 5.08 | 4.20 | 1.21x |
| `test_matmul_add_divide_multiply` | 32 | PASS | 18.65 | 16.86 | 1.11x |
| `test_matmul_add_divide_multiply_add` | 1 | PASS | 2.08 | 0.98 | 2.12x |
| `test_matmul_add_divide_multiply_add` | 8 | PASS | 4.21 | 3.17 | 1.33x |
| `test_matmul_add_divide_multiply_add` | 32 | PASS | 11.86 | 11.19 | 1.06x |

- 全部用例精度比对均通过。
- AP 融合在所有用例上均较 Eager 有加速；epilogue 计算简单、访存占比高的用例（如 `relu`、`gelu`）加速更明显，而 `multiply`/`divide` 类用例在 batch_size 增大后加速比回落至约 1.05~1.3x。

## 运行方式

批量运行（推荐）：

```bash
bash run_test.sh
```

`run_test.sh` 会遍历 5 个用例与多个 `batch_size`，结果分别写入 `logs/<用例名>_bs<batch_size>.txt`。
可在脚本内修改以下变量：

- `TESTS`：要运行的用例列表；
- `BATCH_SIZES`：要遍历的 batch_size 列表；
- `CUDA_VISIBLE_DEVICES`：使用的 GPU 卡号。

单独运行某个用例：

```bash
cd tests
python test_matmul_add_relu.py                  # 使用用例内默认的 batch_size
python test_matmul_add_relu.py --batch_size 8   # 覆盖 batch_size
python test_matmul_add_relu.py -v               # unittest 原生参数依然可用
```

## 关键环境变量

`run_test.sh` 中设置：

```bash
export CUDA_VISIBLE_DEVICES=1                           # 使用的 GPU
export AP_CUTLASS_DIR=/work/Paddle/third_party/cutlass  # CUTLASS 源码路径
export FLAGS_prim_all=True                              # 开启组合算子
export FLAGS_prim_enable_dynamic=true                   # 动态 shape 下的组合算子
export FLAGS_use_cinn=1                                 # 启用 CINN
```

此外 `test_ap_base.py` 会设置 AP 编译产物的工作目录：

```python
os.environ["AP_WORKSPACE_DIR"] = "/tmp/paddle_ap_workspace"
```

生成的融合 kernel 源码（`matmul_variadic_kernel.cu`）与编译产物可在该目录下查看。
