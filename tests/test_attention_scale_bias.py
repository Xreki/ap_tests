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

import paddle.incubate.cc.typing as pct
import test_ap_base


class TestAttentionScaleBias(test_ap_base.AttentionAPTestBase):
    """Attention + per-channel affine: `O * scale + bias` (GPT-2 small shapes).

    Both `scale` and `bias` are `[D]` (broadcast along B / M / H).
    """

    def setUp(self):
        self.attention_config = test_ap_base.make_attention_config("gpt2_small")
        super().setUp()
        self.scale_shape = [self.q_shape[3]]
        self.bias_shape = [self.q_shape[3]]

    def get_subgraph(self):
        B, M, H, D = test_ap_base.make_dim_vars(self.q_shape)
        N = pct.DimVar(self.k_shape[1])
        DType = pct.DTypeVar("T", self.dtype)

        def foo(
            q: pct.Tensor([B, M, H, D], DType),
            k: pct.Tensor([B, N, H, D], DType),
            v: pct.Tensor([B, N, H, D], DType),
            scale: pct.Tensor([D], DType),
            bias: pct.Tensor([D], DType),
        ):
            o = self.attention(q, k, v)
            return o * scale + bias

        return foo

    def test_subgraph(self):
        tensor_args = self.init_tensors(
            self.dtype,
            [
                self.q_shape,
                self.k_shape,
                self.v_shape,
                self.scale_shape,
                self.bias_shape,
            ],
        )
        self.run_subgraph_test(self.get_subgraph(), tensor_args)


if __name__ == "__main__":
    test_ap_base.main()
