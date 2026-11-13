# AP 测试集

PaddlePaddle AP (Abstract Pass) 算子融合的功能与性能测试集。

每个用例构造一个 `matmul / conv2d / flash_attn + elementwise/broadcast` 的子图，分别用以下两种方式执行并对比：

- **Eager**：PaddlePaddle 动态图逐算子执行；
- **AP**：通过 `paddle.incubate.cc.compile` 编译，将 matmul / conv2d / attention 的 epilogue（bias 及后续 elementwise/broadcast）融合进一个 kernel（matmul / conv2d 编译为 CUTLASS kernel，attention 编译为 FlashAttention kernel）。

测试会校验两者的数值精度，并分别 benchmark 二者的 GPU 耗时、输出加速比。

## 目录结构

```
ap_tests/
├── run_matmul_test.sh          # matmul 批量运行脚本（遍历用例 + batch_size）
├── run_conv2d_test.sh          # conv2d 批量运行脚本（遍历用例 + batch_size）
├── run_attention_test.sh       # attention 批量运行脚本（遍历用例 + batch_size）
├── tools/
│   └── extract_perf.py         # 从日志提取 EagerTime / APTime / Speedup 并汇总
├── tests/
│   ├── test_ap_base.py         # 测试基类：编译、精度校验、benchmark、main 入口
│   ├── test_matmul_add_relu.py
│   ├── test_matmul_add_gelu.py
│   ├── test_matmul_add_multiply.py
│   ├── test_matmul_add_divide_multiply.py
│   ├── test_matmul_add_divide_multiply_add.py
│   ├── test_conv2d_add_relu.py
│   ├── test_conv2d_add_bias_residual_relu.py
│   ├── test_conv2d_leaky_relu.py
│   ├── test_attention_add.py
│   ├── test_attention_add_relu.py
│   ├── test_attention_add_bias_residual_relu.py
│   ├── test_attention_add_gelu.py
│   └── test_attention_scale_bias.py
└── logs/                       # 运行脚本生成的日志（每个用例 + batch_size 一个文件）
```

## 测试用例

### matmul 用例

各用例的融合子图（`x @ w + b` 之后的 epilogue）：

| 用例 | 融合计算 |  M | N | K |
| --- | --- | --- | --- | --- |
| `test_matmul_add_relu` | `relu(x @ w + b)` | 500 | 1024 | 256 |
| `test_matmul_add_gelu` | `gelu(x @ w + b)` | 1280 | 512 | 128 |
| `test_matmul_add_multiply` | `(x @ w + b) + z` | 784 | 192 | 768 |
| `test_matmul_add_divide_multiply` | `(x @ w + b) / e1 * e2` | 3136 | 96 | 384 |
| `test_matmul_add_divide_multiply_add` | `(x @ w + b) / e1 * e2 + e3` | 784 | 192 | 768 |

> 注：含除法的用例会对除数 `e1` 做 `abs(e1) + 1.0` 处理，避免 float16 下除数接近 0 导致溢出为 `inf`，进而造成精度比对失败。

### conv2d 用例

卷积采用 NHWC 布局，`stride=1`（`1x1` 卷积 `padding=0`，`3x3` 卷积 `padding=1`，输出空间尺寸均与输入一致）。各用例融合 `conv2d(x, w)` 之后的 epilogue（`B` 为 batch_size，由 `--batch_size` 覆盖）：

| 用例 | 卷积 | 融合计算 | 输入 x (NHWC) | 卷积核 w (OCHW) | 输出 (NHWC) |
| --- | --- | --- | --- | --- | --- |
| `test_conv2d_add_relu` | 3x3 | `relu(conv2d(x, w) + b)` | `[B, 56, 56, 64]` | `[128, 64, 3, 3]` | `[B, 56, 56, 128]` |
| `test_conv2d_add_bias_residual_relu` | 1x1 | `relu(conv2d(x, w) + bias + residual)` | `[B, 56, 56, 64]` | `[256, 64, 1, 1]` | `[B, 56, 56, 256]` |
| `test_conv2d_leaky_relu` | 3x3 | `leaky_relu(conv2d(x, w), 0.1)` | `[B, 52, 52, 128]` | `[256, 128, 3, 3]` | `[B, 52, 52, 256]` |

> 注：`add_relu` 中的 `b` 为与输出同形状的张量 `[B, 56, 56, 128]`；`add_bias_residual_relu` 中 `bias` 为按通道广播的 `[256]`、`residual` 为与输出同形状的 `[B, 56, 56, 256]`；`leaky_relu` 的 `negative_slope=0.1` 为标量常量，无额外张量输入。

### attention 用例

注意力使用 `paddle.nn.functional.flash_attention`（`training=False`，即 `is_test=True`），对应 AP 的 `FlashAttnEpilogueFusion` pass（锚定 `pd_op.flash_attn`），epilogue 在 FlashAttention kernel 内完成。Q/K/V/O 均为 BMHK 布局：Q `[B, M, H, D]`，K/V `[B, N, H, D]`，O `[B, M, H, D]`，其中 `D` 为编译期常量（`Dh == Dv`）。各用例采用典型模型的 shape（hidden = H × D），`B` 可用 `--batch_size` 覆盖：

| 模型预设 | B | M = N | H | D | hidden |
| --- | --- | --- | --- | --- | --- |
| BERT-base | 8 | 512 | 12 | 64 | 768 |
| GPT-2 small | 4 | 1024 | 12 | 64 | 768 |
| LLaMA-7B | 1 | 2048 | 32 | 128 | 4096 |
| ViT-B/16 | 8 | 197（196 patches + CLS） | 12 | 64 | 768 |

各用例融合 attention 之后的 epilogue：

| 用例 | 模型预设 | 融合计算 | 额外输入 |
| --- | --- | --- | --- |
| `test_attention_add` | BERT-base | `O + residual` | `residual: [B,M,H,D]` |
| `test_attention_add_relu` | ViT-B/16 | `relu(O + bias)` | `bias: [B,M,H,D]` |
| `test_attention_add_bias_residual_relu` | GPT-2 small | `relu(O + bias + residual)` | `bias: [D]`, `residual: [B,M,H,D]` |
| `test_attention_add_gelu` | LLaMA-7B | `gelu(O + bias)` | `bias: [D]`（`D=128`） |
| `test_attention_scale_bias` | GPT-2 small | `O * scale + bias` | `scale: [D]`, `bias: [D]` |

> 注：
> - attention 融合后端仅支持 **CUDA sm80（A100）**。
> - 设置 `AP_FLASHATTN_DIR` 时使用（已 patch 的）FlashAttention 后端做 in-epilogue 真融合；未设置时回退到 CUTLASS FMHA 后端。
> - 当前共享的广播化简 pass 只消除 “提升 rank” 的广播（如 `[D]` 广播到 4 维），尚不支持同 rank 的 4D 广播（如 `[1,1,H,1]` / `[1,1,1,D]`）；因此 per-channel bias 统一用 `[D]` 表达。需要覆盖 per-head / 同 rank 广播时，需先扩展 `topo_drr_pass.py` 中的 `CinnBroadcastUpSpiderAccessTopoPass`。

### 正确性检查

正确性由 `APTestBase.check_accuracy(foo, fused_func, tensor_args)` 校验，分两步保证“确实发生了融合”且“融合结果正确”：

1. **融合校验**：编译后（`get_compiled_func`）检查生成的 PIR program 中是否包含 `pd_op.ap_variadic` 算子，若不存在则判定 AP 融合失败、用例直接报错。
2. **数值比对**：用同一组输入张量分别执行 Eager（`foo`）与 AP（`fused_func`），对每个输出用 `np.testing.assert_allclose(ap_out, eager_out, atol=self.atol, rtol=self.rtol)` 逐元素比对。默认 `atol=rtol=1e-2`；attention 用例因 FlashAttention 融合 kernel 与 phi kernel 的 fp16 累加差异略大，在 `AttentionAPTestBase` 中覆盖为 `1e-1`。

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
- CUTLASS 源码（通过环境变量 `AP_CUTLASS_DIR` 指定）；
- FlashAttention 源码（通过环境变量 `AP_FLASHATTN_DIR` 指定，attention 用例的 in-epilogue 融合后端依赖它）。

以下为 `logs/cuda/` 目录下日志的汇总（dtype 为 float16，时间为 benchmark 得到的 GPU 耗时，越小越好；`Speedup = EagerTime / APTime`）。各用例的基础 shape（`B` 由 `--batch_size` 覆盖）：

#### matmul 用例

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

#### conv2d 用例

| 用例 | batch_size | 精度 | EagerTime (ms) | APTime (ms) | Speedup |
| --- | --- | --- | --- | --- | --- |
| `test_conv2d_add_relu` | 1 | PASS | 2.15 | 1.81 | 1.19x |
| `test_conv2d_add_relu` | 8 | PASS | 4.99 | 4.21 | 1.19x |
| `test_conv2d_add_relu` | 32 | PASS | 20.64 | 20.08 | 1.03x |
| `test_conv2d_add_bias_residual_relu` | 1 | PASS | 2.66 | 1.50 | 1.77x |
| `test_conv2d_add_bias_residual_relu` | 8 | PASS | 7.66 | 5.32 | 1.44x |
| `test_conv2d_add_bias_residual_relu` | 32 | PASS | 36.86 | 24.79 | 1.49x |
| `test_conv2d_leaky_relu` | 1 | PASS | 2.76 | 2.50 | 1.10x |
| `test_conv2d_leaky_relu` | 8 | PASS | 8.28 | 7.35 | 1.13x |
| `test_conv2d_leaky_relu` | 32 | PASS | 34.78 | 29.36 | 1.18x |

- 全部用例精度比对均通过。
- `add_bias_residual_relu`（1×1、输出 256 通道）融合的 epilogue（bias + residual + relu）访存收益最大，稳定在 1.44~1.77x。
- `add_relu`（3×3）在 batch_size 增大后加速比回落（bs32 仅 1.03x）：3×3 卷积本身计算占比高，epilogue 的访存收益相对有限。

#### attention 用例

以下为 `batch_size=1/8/32` 遍历结果（`M/N/H/D` 使用上表各用例的模型预设，`B` 由 `--batch_size` 覆盖）：

| 用例 | batch_size | 精度 | EagerTime (ms) | APTime (ms) | Speedup |
| --- | --- | --- | --- | --- | --- |
| `test_attention_add` | 1 | PASS | 2.40 | 2.45 | 0.98x |
| `test_attention_add` | 8 | PASS | 7.34 | 7.16 | 1.03x |
| `test_attention_add` | 32 | PASS | 26.36 | 23.75 | 1.11x |
| `test_attention_add_relu` | 1 | PASS | 1.96 | 1.82 | 1.08x |
| `test_attention_add_relu` | 8 | PASS | 3.02 | 2.47 | 1.22x |
| `test_attention_add_relu` | 32 | PASS | 10.21 | 9.15 | 1.12x |
| `test_attention_add_bias_residual_relu` | 1 | PASS | 4.98 | 4.29 | 1.16x |
| `test_attention_add_bias_residual_relu` | 8 | PASS | 27.31 | 24.34 | 1.12x |
| `test_attention_add_bias_residual_relu` | 32 | PASS | 100.57 | 75.24 | 1.34x |
| `test_attention_add_gelu` | 1 | PASS | 48.06 | 50.78 | 0.95x |
| `test_attention_add_gelu` | 8 | PASS | 318.67 | 299.69 | 1.06x |
| `test_attention_add_gelu` | 32 | PASS | 1169.31 | 1205.46 | 0.97x |
| `test_attention_scale_bias` | 1 | PASS | 4.71 | 4.10 | 1.15x |
| `test_attention_scale_bias` | 8 | PASS | 25.84 | 23.39 | 1.10x |
| `test_attention_scale_bias` | 32 | PASS | 91.85 | 81.19 | 1.13x |

- 全部用例精度比对均通过，且均成功匹配 `pd_op.ap_variadic`。
- 平均加速比约 1.10x（最高 1.34x、最低 0.95x）。
- attention 子图由 `FlashAttnEpilogueFusion` 锚定 `pd_op.flash_attn`，epilogue 在 FlashAttention kernel 内完成，输出只写回一次。由于典型模型 shape 下 attention 本体的算力/访存占比很高，epilogue 融合收益有限，加速比大多在 0.95~1.2x；epilogue 访存越多收益越明显，`bias + residual + relu`（bs32）可达 1.34x。

## 运行方式

批量运行（推荐）：

```bash
bash run_matmul_test.sh     # 运行 matmul 全部用例
bash run_conv2d_test.sh     # 运行 conv2d 全部用例
bash run_attention_test.sh  # 运行 attention 全部用例
```

运行脚本会遍历对应类型的用例与 `batch_size`，结果分别写入 `logs/<用例名>_bs<batch_size>.txt`。
全部用例跑完后，脚本会自动调用 `tools/extract_perf.py` 汇总**本次运行**生成的日志，在终端最后打印 `Eager / AP / Speedup` 表格与平均/最高/最低加速比。也可手动分析：

```bash
python tools/extract_perf.py logs                 # 汇总 logs/ 下全部日志
python tools/extract_perf.py logs -r              # 递归包含子目录
python tools/extract_perf.py logs --files logs/test_attention_add_bs1.txt -o perf.csv
```

可在脚本内修改以下变量：

- `TESTS`：该类型下要运行的用例列表；
- `BATCH_SIZES`：要遍历的 batch_size 列表（默认均为 `1 8 32`）；
- `CUDA_VISIBLE_DEVICES`：使用的 GPU 卡号。

单独运行某个用例：

```bash
cd tests
python test_matmul_add_relu.py                  # 使用用例内默认的 batch_size
python test_matmul_add_relu.py --batch_size 8   # 覆盖 batch_size
python test_matmul_add_relu.py -v               # unittest 原生参数依然可用
```

## 关键环境变量

`run_matmul_test.sh` / `run_conv2d_test.sh` / `run_attention_test.sh` 中设置：

```bash
export CUDA_VISIBLE_DEVICES=0                           # 使用的 GPU
export AP_CUTLASS_DIR=/work/Paddle/third_party/cutlass  # CUTLASS 源码路径
export AP_FLASHATTN_DIR=/work/Paddle/third_party/flashattn  # FlashAttention 源码路径（attention 用例）
export FLAGS_prim_all=True                              # 开启组合算子
export FLAGS_prim_enable_dynamic=true                   # 动态 shape 下的组合算子
export FLAGS_use_cinn=1                                 # 启用 CINN
```

此外 `test_ap_base.py` 会设置 AP 编译产物的工作目录：

```python
os.environ["AP_WORKSPACE_DIR"] = "/tmp/paddle_ap_workspace"
```

生成的融合 kernel 源码（matmul 为 `matmul_variadic_kernel.cu`，conv2d 为 `conv2d_variadic_kernel.cu`，attention 为 `attention_variadic_kernel.cu`）与编译产物可在该目录下查看。
