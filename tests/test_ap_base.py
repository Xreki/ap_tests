# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import sys
import argparse
import unittest
import collections
import numpy as np
from dataclasses import dataclass

import paddle
import paddle.profiler as profiler
import paddle.incubate.cc as pcc
import paddle.incubate.cc.typing as pct

os.environ["AP_WORKSPACE_DIR"] = "/tmp/paddle_ap_workspace"

# Optional CLI override for the matmul batch size (B). When set via main(),
# MatmulConfig.__post_init__ uses it instead of the value passed in each test.
_BATCH_SIZE_OVERRIDE = None


def get_backend_device():
    place = paddle.framework._current_expected_place()
    if isinstance(place, paddle.CUDAPlace):
        device_type = 'dcu' if paddle.is_compiled_with_rocm() else 'cuda'
    elif isinstance(place, paddle.CustomPlace):
        device_type = "custom_device"
    elif isinstance(place, paddle.CPUPlace):
        device_type = "cpu"
    else:
        device_type = "unknown"
    return device_type


def _get_pir_program(fused_func, tensor_args):
    dtypes = tuple(tensor.dtype for tensor in tensor_args)
    func = fused_func.func_overload_ctx.dtypes2func.get(dtypes, None)
    return str(func.infer_program.forward_program)


def _parse_kernel_time(log_text):
    in_overview = False
    for line in log_text.splitlines():
        if "Overview Summary" in line:
            in_overview = True
            continue
        if not in_overview:
            continue
        cols = re.split(r"\s{2,}", line.strip())
        if len(cols) >= 4 and cols[0] == "Kernel":
            return {
                "calls": int(cols[1]),
                "gpu_time": float(cols[2]),
            }
    return None


def _summary_profiler_kernel_time(prof):
    if prof.profiler_result:
        statistic_data = profiler.profiler_statistic.StatisticData(
            prof.profiler_result.get_data(),
            prof.profiler_result.get_extra_info(),
        )
        summary_text = profiler.profiler_statistic._build_table(statistic_data)
        print(summary_text)
        kernel_info = _parse_kernel_time(summary_text)
        if kernel_info:
            return kernel_info["gpu_time"]
    return None


def bench_device_time(fn, input_args, warmup_iters=10, repeat_iters=100, enable_profile=True):
    def _flush_l2_cache() -> None:
        buf = paddle.zeros([64 * 1024 * 1024 // 4], dtype="float32")
        del buf

    # warmup
    for _ in range(warmup_iters):
        fn(*input_args)
    paddle.device.synchronize()

    if enable_profile:
        with profiler.Profiler(
            targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],
            scheduler=[warmup_iters, warmup_iters + repeat_iters],
        ) as prof:
            for _ in range(warmup_iters + repeat_iters + 1):
                fn(*input_args)
                prof.step()
            # prof.summary()
            kernel_time = _summary_profiler_kernel_time(prof)
            if kernel_time:
                return kernel_time

    e2e_times = []
    for _ in range(repeat_iters):
        _flush_l2_cache()
        start_event = paddle.device.cuda.Event(enable_timing=True)
        end_event = paddle.device.cuda.Event(enable_timing=True)
        start_event.record()
        fn(*input_args)
        end_event.record()
        paddle.device.synchronize()
        e2e_times.append(start_event.elapsed_time(end_event))
    avg_e2e_time = sum(e2e_times) / len(e2e_times)
    return avg_e2e_time


@dataclass
class MatmulConfig:
    B: int
    M: int
    N: int
    K: int

    def __post_init__(self):
        # Allow a CLI override (set via main()) to replace the hardcoded
        # batch size in every test's setUp without touching each test.
        if _BATCH_SIZE_OVERRIDE is not None:
            self.B = _BATCH_SIZE_OVERRIDE


class APTestBase(unittest.TestCase):
    def init_tensors(self, dtype, tensor_shapes):
        tensors = []
        for shape in tensor_shapes:
            t = paddle.randn(shape, dtype=dtype)
            t.stop_gradient = False
            tensors.append(t)
        return tensors

    def get_compiled_func(self, func, tensor_args):
        backend_device = get_backend_device()
        fused_func = pcc.compile(
            func,
            backend_device=backend_device,
        )
        generated_pir_program = _get_pir_program(fused_func, tensor_args)
        self.assertTrue(
            'pd_op.ap_variadic' in generated_pir_program, "No pd_op.ap_variadic is found in compiled program. AP fusion failed!"
        )
        return fused_func

    def check_accuracy(self, func, fused_func, tensor_args):
        dy_outs = func(*tensor_args)
        ap_outs = fused_func(*tensor_args)
        for dy_out, ap_out in zip(dy_outs, ap_outs):
            np.testing.assert_allclose(dy_out, ap_out, atol=1e-2, rtol=1e-2)


def main():
    """Entry point for the test scripts.

    Parses test-specific args (currently --batch_size) and forwards the
    rest to unittest so its own flags (-v, test names, etc.) still work.
    """
    global _BATCH_SIZE_OVERRIDE

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override the matmul batch size (B) used in setUp.",
    )
    args, remaining = parser.parse_known_args()

    if args.batch_size is not None:
        _BATCH_SIZE_OVERRIDE = args.batch_size

    # Rebuild argv so unittest only sees the args meant for it.
    unittest.main(argv=[sys.argv[0]] + remaining)
