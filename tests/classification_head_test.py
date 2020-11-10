import pytest
import torch.nn as nn

from optimus.heads.classification_head import ClassificationHead
from optimus.testing import OptimusTestCase


class TestClassificationHeadForward(OptimusTestCase):
    @pytest.mark.parametrize(
        "batch_size, max_seq_len, model_size, n_classes",
        [
            (16, 25, 64, 2),
            (1, 25, 64, 2),
            (16, 25, 64, 20),
            (1, 25, 64, 20),
        ],
    )
    def test_should_return_correct_shape(
        self, batch_size, max_seq_len, model_size, n_classes
    ):
        x = self._get_tensor_with_shape(batch_size, max_seq_len, model_size)

        clf = ClassificationHead(model_size, n_classes=n_classes)
        result = clf(x)

        self.assert_tensor_has_shape(result, (batch_size, n_classes))
