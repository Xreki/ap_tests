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

import paddle
import paddle.incubate.cc.typing as pct
import test_ap_base


class TestConv2dAddRelu(test_ap_base.Conv2dAPTestBase):
    def setUp(self):
        self.conv_config = test_ap_base.Conv2dConfig(
            N=1, H=56, W=56, C=64, O=128, KH=3, KW=3, padding=1
        )
        super().setUp()

    def get_subgraph(self):
        N, H, W, C = test_ap_base.make_dim_vars(self.x_shape)
        O, KH, KW = test_ap_base.make_dim_vars(
            [self.w_shape[0], self.w_shape[2], self.w_shape[3]]
        )
        P, Q = test_ap_base.make_dim_vars(self.b_shape[1:3])
        DType = pct.DTypeVar("T", self.dtype)

        def foo(
            x: pct.Tensor([N, H, W, C], DType),
            w: pct.Tensor([O, C, KH, KW], DType),
            b: pct.Tensor([N, P, Q, O], DType),
        ):
            y = self.conv2d_nhwc(x, w)
            return paddle.nn.functional.relu(y + b)

        return foo

    def test_subgraph(self):
        tensor_args = self.init_tensors(
            self.dtype, [self.x_shape, self.w_shape, self.b_shape]
        )
        self.run_subgraph_test(self.get_subgraph(), tensor_args)


if __name__ == "__main__":
    test_ap_base.main()
