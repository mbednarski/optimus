from optimus.positional_embedding import PositionalEmbedding
from optimus.testing import OptimusTestCase


class TestPositionalEncoding(OptimusTestCase):
    def test_should_return_correct_shape(self):
        pos_embedding = PositionalEmbedding(50, 128)

        output = pos_embedding()

        self.assert_tensor_has_shape(output, (1, 50, 128))

    def test_should_not_require_grad(self):
        pos_embedding = PositionalEmbedding(50, 128)

        output = pos_embedding()

        self.assert_tensor_does_not_require_grad(output)
