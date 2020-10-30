import torch
import torch.nn as nn

from optimus.add_and_normalize import AddAndNormalize
from optimus.self_attention_head import SelfAttention
from optimus.transformer_feed_forward import TransformerFeedForward


class EncoderLayer(nn.Module):
    def __init__(
        self,
        n_attention_heads: int,
        model_size: int,
        attention_size: int,
        seq_len: int,
        feed_forward_hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.attention = SelfAttention(
            n_heads=n_attention_heads,
            embedding_size=model_size,
            value_size=attention_size,
        )
        self.add_and_normalize1 = AddAndNormalize((seq_len, model_size))
        self.add_and_normalize2 = AddAndNormalize((seq_len, model_size))

        self.feed_forward = TransformerFeedForward(
            model_size=model_size, hidden_size=feed_forward_hidden_size, dropout=dropout
        )

    def forward(self, x):
        z1 = self.attention(x)
        z2 = self.add_and_normalize1(x, z1)
        ff_out = self.feed_forward(z2)
        out = self.add_and_normalize2(ff_out, z2)

        return out
