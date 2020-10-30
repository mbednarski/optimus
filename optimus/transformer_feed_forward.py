import torch
import torch.nn as nn


class TransformerFeedForward(nn.Module):
    def __init__(
        self,
        model_size,
        hidden_size,
        dropout=0.1,
    ):
        super().__init__()

        self.fc1 = nn.Linear(model_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_size, model_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    bs = 16
    seq_len = 20
    ms = 28

    x = torch.randn(bs, seq_len, ms)
    tff = TransformerFeedForward(ms, 1024)
    out = tff.forward(x)

    print(out.shape)
