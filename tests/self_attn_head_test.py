from dataclasses import dataclass
from typing import Tuple

import pytest
import torch
import torch.nn as nn
from assertpy import assert_that

from optimus.self_attention_head import SelfAttentionHead


class TestSelfAttention:
    def _get_tensor_with_shape(self, *shape):
        return torch.randn(*shape)

    def assert_tensor_has_shape(self, tensor: torch.Tensor, shape: Tuple[int]):
        assert_that(tensor).is_type_of(torch.Tensor)
        assert_that(tensor.shape).is_equal_to(shape)

    @pytest.mark.parametrize(
        "batch_size,max_seq_len,model_size,keyvalue_size",
        [(16, 25, 30, 12), (1, 25, 30, 12)],
    )
    def test_should_return_correct_shape(
        self, batch_size, max_seq_len, model_size, keyvalue_size
    ):
        data = self._get_tensor_with_shape(batch_size, max_seq_len, model_size)
        attn_head = SelfAttentionHead(
            model_size=model_size, keyvalue_size=keyvalue_size
        )

        result = attn_head(data)

        self.assert_tensor_has_shape(result, (batch_size, max_seq_len, keyvalue_size))

    @pytest.mark.parametrize(
        "batch_size,max_seq_len,model_size,keyvalue_size",
        [(16, 25, 30, 12), (1, 25, 30, 12)],
    )
    def test_should_have_grad_in_params_after_backprop(
        self, batch_size, max_seq_len, model_size, keyvalue_size
    ):
        x = self._get_tensor_with_shape(batch_size, max_seq_len, model_size)
        y = self._get_tensor_with_shape(batch_size, max_seq_len, keyvalue_size)

        attn_head = SelfAttentionHead(
            model_size=model_size, keyvalue_size=keyvalue_size
        )

        result = attn_head(x)

        criterion = nn.MSELoss()
        loss = criterion(y, result)
        loss.backward()

        assert attn_head.W_q.grad is not None
        assert attn_head.W_k.grad is not None
        assert attn_head.W_v.grad is not None
