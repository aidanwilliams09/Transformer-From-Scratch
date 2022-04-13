import torch
from torch import nn
from transformer_552.transformer_helpers import (Residual,
                                 MultiHeadAttention,
                                 feed_forward,
                                 position_encoding)


class TransformerEncoderLayer(nn.Module):
    
    def __init__(self, dim_model: int, num_heads: int, dim_feedforward: int, dropout: float):
        super(TransformerEncoderLayer, self).__init__()
        dim_q = dim_k = max(dim_model // num_heads, 1)
        self.attention = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout,
        )


    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.attention(src, src, src)
        return self.feed_forward(src)


class TransformerDecoderLayer(nn.Module):
    
    def __init__(self, dim_model: int, num_heads: int, dim_feedforward: int, dropout: float):
        super(TransformerDecoderLayer, self).__init__()
        dim_q = dim_k = max(dim_model // num_heads, 1)

        self.attention_1 = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
            dimension=dim_model,
            dropout=dropout,
        )
        self.attention_2 = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout,
        )


    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        tgt = self.attention_1(tgt, tgt, tgt)
        tgt = self.attention_2(tgt, memory, memory)
        return self.feed_forward(tgt)


class TransformerEncoder(nn.Module):
    
    def __init__(self, num_layers: int, dim_model: int, num_heads: int, dim_feedforward: int, dropout: float):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )


    def forward(self, src: torch.Tensor) -> torch.Tensor:
        seq_len, dimension = src.size(1), src.size(2)
        src += position_encoding(seq_len, dimension)
        for layer in self.layers:
            src = layer(src)

        return src


class TranformerDecoder(nn.Module):
    
    def __init__(self, num_layers: int, dim_model: int, num_heads: int, dim_feedforward: int, dropout: float):
        super(TranformerDecoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(dim_model, num_heads, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )
        self.linear = nn.Linear(dim_model, dim_model)


    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        seq_len, dimension = tgt.size(1), tgt.size(2)
        tgt += position_encoding(seq_len, dimension)
        for layer in self.layers:
            tgt = layer(tgt, memory)

        return torch.softmax(self.linear(tgt), dim=-1)


class Transformer(nn.Module):
    
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super(Transformer, self).__init__()
        
        self.encoder = TransformerEncoder(
            num_encoder_layers,
            dim_model,
            num_heads,
            dim_feedforward,
            dropout
        )
        
        self.decoder = TranformerDecoder(
            num_decoder_layers,
            dim_model,
            num_heads,
            dim_feedforward,
            dropout
        )
    
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        return self.decoder(tgt, self.encoder(src))
