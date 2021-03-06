from typing import Tuple

import torch
import torch.nn as nn
from assertpy import assert_that
from torch import Tensor


class OptimusTestCase:
    def _get_tensor_with_shape(self, *shape):
        return torch.randn(*shape)

    def _get_vocab_tensor_with_shape(
        self, batch_size: int, vocab_size: int, seq_len: int
    ):
        return torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))

    def assert_tensor_has_shape(self, tensor: torch.Tensor, shape: Tuple[int]):
        assert_that(tensor).is_type_of(torch.Tensor)
        assert_that(tensor.shape).is_equal_to(shape)

    def assert_parameter_has_grad(self, parameter: nn.Parameter):
        assert_that(parameter).is_type_of(nn.Parameter)
        assert_that(parameter.grad).is_not_none()
        assert_that(parameter.grad.sum().item()).is_not_zero()

    def assert_all_named_parameters_have_grad(self, module: nn.Module):
        assert_that(module).is_type_of(nn.Module)
        for _, param in module.named_parameters():
            assert_that(param.grad).is_not_none()
            assert_that(param.grad.sum().item()).is_not_zero()

    def assert_tensor_does_not_require_grad(self, tensor: Tensor):
        assert_that(tensor).is_type_of(Tensor)
        assert_that(tensor).is_not_none()
        assert_that(tensor.requires_grad).is_false()
