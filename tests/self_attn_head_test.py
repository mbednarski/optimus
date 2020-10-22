import torch
import torch.nn as nn

from optimus.self_attention_head import SelfAttention, SelfAttentionHead


def test_self_attn_head_has_correct_shape_for_no_batch():
    embed_size = 589
    val_size = 124
    seq_len = 512
    data = torch.randn(seq_len, embed_size)

    attn_head = SelfAttentionHead(embed_size, val_size)
    result = attn_head(data)

    assert result.shape == (seq_len, val_size)


def test_self_attn_head_is_training_params():
    embed_size = 128
    val_size = 64
    seq_len = 512
    data = torch.randn(seq_len, embed_size)
    y = torch.randn(seq_len, val_size)
    criterion = nn.MSELoss()

    attn_head = SelfAttentionHead(embed_size, val_size)
    result = attn_head(data)
    loss = criterion(y, result)
    loss.backward()

    assert attn_head.W_q.grad is not None
    assert attn_head.W_k.grad is not None
    assert attn_head.W_v.grad is not None


def test_self_attn_has_correct_shape_for_no_batch():
    embed_size = 32
    val_size = 12
    seq_len = 128
    n_heads = 6
    data = torch.randn(seq_len, embed_size)
    y = torch.randn(seq_len, embed_size)
    criterion = nn.MSELoss()

    self_attn = SelfAttention(n_heads, embed_size, val_size)
    result = self_attn(data)
    loss = criterion(y, result)
    loss.backward()

    assert self_attn.WO.grad is not None
    for h in self_attn.heads:
        for name, param in h.named_parameters():
            param.grad is not None
