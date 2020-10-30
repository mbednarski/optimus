import pytest
import torch.nn as nn

from optimus.encoder_layer import EncoderLayer
from optimus.testing import OptimusTestCase


class TestTransformerFeedForward(OptimusTestCase):
    @pytest.mark.parametrize(
        "batch_size, max_seq_len, model_size, n_heads,keyvalue_size,ff_hidden_size",
        [
            (16, 25, 64, 6, 56, 256),
            (1, 25, 64, 6, 56, 256),
        ],
    )
    def test_should_return_correct_shape(
        self,
        batch_size,
        max_seq_len,
        model_size,
        n_heads,
        keyvalue_size,
        ff_hidden_size,
    ):
        x = self._get_tensor_with_shape(batch_size, max_seq_len, model_size)

        encoder_layer = EncoderLayer(
            n_attention_heads=n_heads,
            model_size=model_size,
            keyvalue_size=keyvalue_size,
            max_seq_len=max_seq_len,
            feed_forward_hidden_size=ff_hidden_size,
            dropout=0.1,
        )

        result = encoder_layer(x)

        self.assert_tensor_has_shape(result, (batch_size, max_seq_len, model_size))
