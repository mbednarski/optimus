import pytest
import torch
import torch.nn as nn

from optimus.self_attention import SelfAttention
from optimus.testing import OptimusTestCase


class TestSelfAttention(OptimusTestCase):
    @pytest.mark.parametrize(
        "batch_size,max_seq_len,model_size,keyvalue_size,n_heads",
        [(16, 25, 30, 12, 4), (1, 25, 30, 12, 4)],
    )
    def test_should_return_correct_shape(
        self, batch_size, max_seq_len, model_size, keyvalue_size, n_heads
    ):

        data = self._get_tensor_with_shape(batch_size, max_seq_len, model_size)
        attn = SelfAttention(
            n_heads=n_heads, model_size=model_size, keyvalue_size=keyvalue_size
        )

        result = attn(data)

        self.assert_tensor_has_shape(result, (batch_size, max_seq_len, model_size))

    @pytest.mark.parametrize(
        "batch_size,max_seq_len,model_size,keyvalue_size,n_heads",
        [(16, 25, 30, 12, 4), (1, 25, 30, 12, 4)],
    )
    def test_should_have_grad_in_params_after_backprop(
        self, batch_size, max_seq_len, model_size, keyvalue_size, n_heads
    ):
        data = self._get_tensor_with_shape(batch_size, max_seq_len, model_size)
        y = self._get_tensor_with_shape(batch_size, max_seq_len, model_size)

        self_attn = SelfAttention(
            n_heads=n_heads, model_size=model_size, keyvalue_size=keyvalue_size
        )

        result = self_attn(data)
        criterion = nn.MSELoss()
        loss = criterion(y, result)
        loss.backward()

        self.assert_parameter_has_grad(self_attn.WO)
        for h in self_attn.heads:
            for _, param in h.named_parameters():
                self.assert_parameter_has_grad(param)
