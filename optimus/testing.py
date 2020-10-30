from typing import Tuple

import torch
import torch.nn as nn
from assertpy import assert_that


class OptimusTestCase:
    def _get_tensor_with_shape(self, *shape):
        return torch.randn(*shape)

    def assert_tensor_has_shape(self, tensor: torch.Tensor, shape: Tuple[int]):
        assert_that(tensor).is_type_of(torch.Tensor)
        assert_that(tensor.shape).is_equal_to(shape)

    def assert_parameter_has_grad(self, parameter: nn.Parameter):
        assert_that(parameter).is_type_of(nn.Parameter)
        assert_that(parameter.grad).is_not_none()
        assert_that(parameter.grad.sum().item()).is_not_zero()
