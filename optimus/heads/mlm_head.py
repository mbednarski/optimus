import torch
import torch.nn as nn

class MlmHead(nn.Module):
    def __init__(self, model_size, vocab_size) -> None:
        super().__init__()
        self.ff = nn.Linear(model_size, vocab_size)

    def forward(self, x):
        out = self.ff(x)
        return out