import math

import torch
import torch.nn as nn
from collections import namedtuple


SelfAttentionOutput = namedtuple('SelfAttentionOutput', [
    "z", "attention_map"
])

class SelfAttentionHead(nn.Module):

    def __init__(self, model_size: int, keyvalue_size: int) -> None:
        super().__init__()
        self.embedding_size = model_size
        self.value_size = keyvalue_size

        self.W_q = nn.Parameter(torch.randn(model_size, keyvalue_size))
        self.W_k = nn.Parameter(torch.randn(model_size, keyvalue_size))
        self.W_v = nn.Parameter(torch.randn(model_size, keyvalue_size))

    def forward(self, x) ->SelfAttentionOutput:
        """Forwards self attn head

        Args:
            x ([type]): shape [batch size x seq len x embed size]
        """

        q = torch.matmul(x, self.W_q)
        k: torch.Tensor = torch.matmul(x, self.W_k)
        v = torch.matmul(x, self.W_v)

        dot_prod = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.value_size)
        softmaxed = torch.softmax(dot_prod, dim=2)
        z = torch.matmul(softmaxed, v)

        return SelfAttentionOutput(z, softmaxed)
