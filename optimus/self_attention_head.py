import torch
import torch.nn as nn


class SelfAttentionHead(nn.Module):
    def __init__(self, embedding_size: int, value_size: int) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.value_size = value_size

        self.W_q = nn.Parameter(torch.randn(embedding_size, value_size))
        self.W_k = nn.Parameter(torch.randn(embedding_size, value_size))
        self.W_v = nn.Parameter(torch.randn(embedding_size, value_size))

    def forward(self, x):
        """Forwards self attn head

        Args:
            x ([type]): shape [batch size x seq len x embed size]
        """

        q = torch.matmul(x, self.W_q)
        k: torch.Tensor = torch.matmul(x, self.W_k)
        v = torch.matmul(x, self.W_v)

        dot_prod = torch.matmul(q, k.transpose(0, 1))
        softmaxed = torch.softmax(dot_prod, dim=1)
        z = torch.matmul(softmaxed, v)

        return z


class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, embedding_size: int, value_size: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList()
        for _ in range(n_heads):
            self.heads.append(SelfAttentionHead(embedding_size, value_size))

        self.WO = nn.Parameter(torch.randn(n_heads * value_size, embedding_size))

    def forward(self, x):
        # x shape seq len x embedding size
        # out shape seq len x embedding size

        head_values = []
        for h in self.heads:
            head_values.append(h(x))

        head_values = torch.cat(head_values, dim=1)

        z = torch.matmul(head_values, self.WO)

        return z
