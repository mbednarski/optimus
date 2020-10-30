import torch
import torch.nn as nn

from optimus.self_attention import SelfAttention


class AddAndNormalize(nn.Module):
    def __init__(self, shape) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(shape)

    def forward(self, x, z):
        summed = x + z
        normalized = self.layer_norm(summed)

        return normalized
