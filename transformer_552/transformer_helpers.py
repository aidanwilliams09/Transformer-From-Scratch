import torch
import torch.nn.functional as F
from torch import nn


def scaled_dot_prod(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    temp = query.bmm(key.transpose(1, 2))
    scale = query.size(-1) ** 0.5
    softmax = F.softmax(temp / scale, dim=-1)
    return softmax.bmm(value)


def position_encoding(seq_len: int, dim_model: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    i = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = pos / (1e4 ** (torch.div(i, 2, rounding_mode='floor') / dim_model))

    return torch.where(i.long() % 2 == 0, torch.sin(phase), torch.cos(phase))


def feed_forward(dim_in: int = 512, dim_inner: int = 2048) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim_in, dim_inner),
        nn.ReLU(),
        nn.Linear(dim_inner, dim_in)
    )


class Residual(nn.Module):
    
    def __init__(self, sub_layer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dimension)
        self.sublayer = sub_layer
        self.dropout = nn.Dropout(dropout)


    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(tensors[0] + self.dropout(self.sublayer(tensors)))


class AttentionHead(nn.Module):
    
    def __init__(self, dim_in: int, dim_q: int, dim_k: int):
        super(AttentionHead, self).__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)


    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        query = self.q(query)
        key = self.k(key)
        value = self.v(value)
        return scaled_dot_prod(query, key, value)


class MultiHeadAttention(nn.Module):
    
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads*dim_k, dim_in)


    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return self.linear(
            torch.cat(
                [h(query, key, value) for h in self.heads],
                dim=-1
            )
        )