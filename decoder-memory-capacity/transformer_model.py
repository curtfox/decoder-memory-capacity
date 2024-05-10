import torch
import torch.nn as nn
import math


class DecoderOnlyTransformer(nn.Module):

    def __init__(self, omega, d, m, tao, device):
        super().__init__()
        self.omega = omega
        self.d = d
        self.m = m
        self.tao = tao
        self.device = device
        self.embed = nn.Embedding(omega, d)
        self.pos_encode = PositionalEncoding(d, tao)
        self.decoder_block = DecoderBlock(d, m, device)
        self.linear = nn.Linear(m, omega)

    def forward(self, x):
        embedded = self.pos_encode(self.embed(x))
        dec_in = embedded
        dec_out = self.decoder_block(dec_in)
        out = self.linear(dec_out)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, m, device):
        super(DecoderBlock, self).__init__()
        self.self_atten = Attention(embed_size, device)
        self.ff = PositionWiseFeedForward(embed_size, m)

    def forward(self, x):
        atten_out = self.self_atten(x, x, x)
        ff_out = self.ff(atten_out)
        return ff_out


class PositionWiseFeedForward(nn.Module):

    def __init__(self, embed_size, m):
        super(PositionWiseFeedForward, self).__init__()
        self.layer = nn.Linear(embed_size, m)
        self.act = nn.GELU()

    def forward(self, x):
        out = self.act(self.layer(x))
        return out


class PositionalEncoding(nn.Module):

    def __init__(self, embed_size, max_length=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_length, embed_size)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class Attention(nn.Module):

    def __init__(self, embed_size, device):
        super(Attention, self).__init__()
        self.embed_size = embed_size
        self.d_k = embed_size
        self.device = device
        self.W_q = nn.Linear(embed_size, embed_size)
        self.W_k = nn.Linear(embed_size, embed_size)
        self.W_v = nn.Linear(embed_size, embed_size)
        self.W_o = nn.Linear(embed_size, embed_size)

    def scaled_dot_product_attention(self, Q, K, V, mask=True):
        atten_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask:
            input_len = atten_scores.size(-1)
            atten_scores = atten_scores.masked_fill(
                torch.tril(torch.ones(input_len, input_len).to(self.device)) == 0,
                float("-inf"),
            )
        atten_probs = torch.softmax(atten_scores, dim=-1)
        output = torch.matmul(atten_probs, V)
        return output

    def forward(self, Q, K, V):
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        atten_output = self.scaled_dot_product_attention(Q, K, V)
        output = self.W_o(atten_output)
        return output
