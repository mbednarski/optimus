import pytest
import torch.nn as nn

from optimus.optimus_embedding import OptimusEmbedding
from optimus.testing import OptimusTestCase


class OptimusEmbeddingForward(OptimusTestCase):
    @pytest.mark.parametrize(
        "batch_size, max_seq_len, model_size, vocab_size",
        [
            (16, 25, 64, 128),
            (1, 25, 64, 128),
        ],
    )
    def test_should_return_correct_shape(
        self, batch_size, max_seq_len, model_size, vocab_size
    ):
        x = self._get_vocab_tensor_with_shape(batch_size, vocab_size, max_seq_len)

        emb = OptimusEmbedding(vocab_size, model_size, max_seq_len)
        result = emb(x)

        self.assert_tensor_has_shape(result, (batch_size, max_seq_len, model_size))

    # @pytest.skip("asd")
    # @pytest.mark.parametrize(
    #     "batch_size, max_seq_len, model_size, vocab_size",
    #     [
    #         (16, 25, 64, 128),
    #         (1, 25, 64, 128),
    #     ],
    # )
    # def test_should_have_grad_after_backprop(
    #     self, batch_size, max_seq_len, model_size, vocab_size
    # ):
    #     x = self._get_vocab_tensor_with_shape(batch_size, vocab_size,max_seq_len)
    #     y = self._get_tensor_with_shape(batch_size, max_seq_len, model_size)

    #     emb = OptimusEmbedding(vocab_size, model_size, max_seq_len)
    #     result = emb(x)

    #     criterion = nn.MSELoss()
    #     loss = criterion(y, result)
    #     loss.backward()

    #     self.assert_all_named_parameters_have_grad(emb.embedding.weight)
