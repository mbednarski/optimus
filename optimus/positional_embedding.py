import numpy as np
import torch
import torch.nn as nn


class PositionalEmbedding:
    """Optimus positional embedding

    Code from: # Code from https://www.tensorflow.org/tutorials/text/transformer
    """

    def __init__(self, sequence_len: int, model_size: int) -> None:
        self.sequence_len = sequence_len
        self.model_size = model_size

    def __call__(self):
        """Generates positional embedding

        Returns:
            Tensor: 1 x seq len x model size
        """
        return self._positional_encoding()

    def _get_angles(self, pos, i):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(self.model_size))
        return pos * angle_rates

    def _positional_encoding(self):
        angle_rads = self._get_angles(
            np.arange(self.sequence_len)[:, np.newaxis],
            np.arange(self.model_size)[np.newaxis, :],
        )

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return torch.FloatTensor(pos_encoding)
