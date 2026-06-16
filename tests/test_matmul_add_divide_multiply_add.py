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
import unittest
import numpy as np

import paddle
import paddle.incubate.cc as pcc
import paddle.incubate.cc.typing as pct
import test_ap_base


class TestMatmulAddDivideMultiplyAdd(test_ap_base.APTestBase):
    def setUp(self):
        self.dtype = 'float16'

        matmul_config = test_ap_base.MatmulConfig(B=1, M=784, N=192, K=768)

        self.x_shape = [matmul_config.B, matmul_config.M, matmul_config.K]
        self.y_shape = [matmul_config.K, matmul_config.N]
        self.b_shape = [matmul_config.N]
        self.e1_shape = [matmul_config.B, matmul_config.M, matmul_config.N]
        self.e2_shape = [matmul_config.B, matmul_config.M, matmul_config.N]
        self.e3_shape = [matmul_config.B, matmul_config.M, matmul_config.N]

    def get_subgraph(self):
        B = pct.DimVar(self.x_shape[0])
        M = pct.DimVar(self.x_shape[1])
        K = pct.DimVar(self.x_shape[2])
        N = pct.DimVar(self.y_shape[1])
        DType = pct.DTypeVar("T", self.dtype)

        def foo(
            x: pct.Tensor([B, M, K], DType),
            w: pct.Tensor([K, N], DType),
            b: pct.Tensor([N], DType),
            e1: pct.Tensor([B, M, N], DType),
            e2: pct.Tensor([B, M, N], DType),
            e3: pct.Tensor([B, M, N], DType),
        ):
            mm_out = paddle.matmul(x, w)
            add_out = mm_out + b
            div_out = add_out / e1
            mul_out = div_out * e2
            out = mul_out + e3
            return out

        return foo

    def test_subgraph(self):
        tensor_args = self.init_tensors(self.dtype, [self.x_shape, self.y_shape, self.b_shape, self.e1_shape, self.e2_shape, self.e3_shape])
        e1 = paddle.abs(tensor_args[3]) + 1.0
        tensor_args[3] = e1

        foo = self.get_subgraph()
        fused_foo = self.get_compiled_func(foo, tensor_args)
        self.check_accuracy(fused_foo, foo, tensor_args)

        print("\nBenchmark_for_eager:\n")
        eager_time = test_ap_base.bench_device_time(foo, tensor_args)

        print("\nBenchmark_for_AP:\n")
        ap_time = test_ap_base.bench_device_time(fused_foo, tensor_args)

        speedup = eager_time / ap_time if eager_time and ap_time else None
        print(f"\nEagerTime: {eager_time:.4f} ms, APTime: {ap_time:.4f} ms, Speedup: {speedup:.2f}x\n")


if __name__ == "__main__":
    test_ap_base.main()
