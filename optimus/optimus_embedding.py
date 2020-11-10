import torch.nn as nn

from optimus.positional_embedding import PositionalEmbedding


class OptimusEmbedding(nn.Module):
    def __init__(
        self, vocab_size: int, model_size: int, seq_len: int, padding_idx: int = 0
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, model_size, padding_idx)
        self.pos_embedding = PositionalEmbedding(seq_len, model_size)

    def forward(self, x):
        # x: batch size x seq len

        # embedded: batch size x seq len x model_size
        embedded = self.embedding(x)
        # pos: 1 x seq len x model size
        pos = self.pos_embedding()

        # embedded with pos: batch size x seq len x model_size
        embedded_with_pos = embedded + pos

        return embedded_with_pos
