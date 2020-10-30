import pytest
import torch.nn as nn

from optimus.add_and_normalize import AddAndNormalize
from optimus.testing import OptimusTestCase


class TestAddAndNormalize(OptimusTestCase):
    @pytest.mark.parametrize(
        "batch_size, max_seq_len, model_size",
        [
            (16, 25, 64),
            (1, 25, 64),
        ],
    )
    def test_should_return_correct_shape(self, batch_size, max_seq_len, model_size):
        x = self._get_tensor_with_shape(batch_size, max_seq_len, model_size)
        z = self._get_tensor_with_shape(batch_size, max_seq_len, model_size)

        add_and_norm = AddAndNormalize((max_seq_len, model_size))
        result = add_and_norm(x, z)

        self.assert_tensor_has_shape(result, (batch_size, max_seq_len, model_size))

    @pytest.mark.parametrize(
        "batch_size, max_seq_len, model_size",
        [
            (16, 25, 64),
            (1, 25, 64),
        ],
    )
    def test_should_have_grad_after_backprop(self, batch_size, max_seq_len, model_size):
        x = self._get_tensor_with_shape(batch_size, max_seq_len, model_size)
        z = self._get_tensor_with_shape(batch_size, max_seq_len, model_size)
        y = self._get_tensor_with_shape(batch_size, max_seq_len, model_size)
        add_and_norm = AddAndNormalize((max_seq_len, model_size))
        result = add_and_norm(x, z)

        criterion = nn.MSELoss()
        loss = criterion(y, result)
        loss.backward()

        self.assert_parameter_has_grad(add_and_norm.layer_norm.weight)
        self.assert_parameter_has_grad(add_and_norm.layer_norm.bias)
