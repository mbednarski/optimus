import torch
import torch.nn as nn

from optimus.self_attention_head import SelfAttentionHead


class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, model_size: int, keyvalue_size: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList()
        for _ in range(n_heads):
            self.heads.append(SelfAttentionHead(model_size, keyvalue_size))

        self.WO = nn.Parameter(torch.randn(n_heads * keyvalue_size, model_size))

    def forward(self, x):
        # x shape seq len x embedding size
        # out shape seq len x embedding size

        head_values = []
        head_maps = []
        for h in self.heads:
            out = h(x)
            head_values.append(out.z)
            head_maps.append(out.attention_map)

        head_values = torch.cat(head_values, dim=2)
        head_maps = torch.stack(head_maps).permute(1,0,2,3)

        z = torch.matmul(head_values, self.WO)

        return z, head_maps
