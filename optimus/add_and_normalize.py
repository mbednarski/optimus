import torch
import torch.nn as nn

from optimus.self_attention_head import SelfAttention


class AddAndNormalize(nn.Module):
    def __init__(self, shape) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(shape)

    def forward(self, x, z):
        summed = x + z
        normalized = self.layer_norm(summed)

        return normalized


if __name__ == "__main__":
    bs = 16
    seq_len = 25
    es = 64

    x = torch.randn(bs, seq_len, es)

    aan = AddAndNormalize((seq_len, es))

    attn = SelfAttention(n_heads=2, embedding_size=es, value_size=8)
    attn_outputs = []
    for i in range(x.shape[0]):
        attn_out = attn(x[i])
        attn_outputs.append(attn_out)

    z = torch.stack(attn_outputs)

    output = aan.forward(x, z)

    pass

    # bs x seq len x embedding size
