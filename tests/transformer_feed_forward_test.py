import pytest
import torch.nn as nn

from optimus.testing import OptimusTestCase
from optimus.transformer_feed_forward import TransformerFeedForward


class TestTransformerFeedForward(OptimusTestCase):
    @pytest.mark.parametrize(
        "batch_size, max_seq_len, model_size, hidden_size",
        [
            (16, 25, 64, 128),
            (1, 25, 64, 128),
        ],
    )
    def test_should_return_correct_shape(
        self, batch_size, max_seq_len, model_size, hidden_size
    ):
        x = self._get_tensor_with_shape(batch_size, max_seq_len, model_size)

        ff = TransformerFeedForward(model_size, hidden_size)
        result = ff(x)

        self.assert_tensor_has_shape(result, (batch_size, max_seq_len, model_size))

    @pytest.mark.parametrize(
        "batch_size, max_seq_len, model_size, hidden_size",
        [
            (16, 25, 64, 256),
            (1, 25, 64, 256),
        ],
    )
    def test_should_have_grad_after_backprop(
        self, batch_size, max_seq_len, model_size, hidden_size
    ):
        x = self._get_tensor_with_shape(batch_size, max_seq_len, model_size)
        y = self._get_tensor_with_shape(batch_size, max_seq_len, model_size)

        ff = TransformerFeedForward(model_size, hidden_size)
        result = ff(x)

        criterion = nn.MSELoss()
        loss = criterion(y, result)
        loss.backward()

        self.assert_parameter_has_grad(ff.fc1.weight)
        self.assert_parameter_has_grad(ff.fc1.bias)
        self.assert_parameter_has_grad(ff.fc2.weight)
        self.assert_parameter_has_grad(ff.fc2.bias)
