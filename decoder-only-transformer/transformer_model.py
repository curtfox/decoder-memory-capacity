import torch
import torch.nn as nn
import math


class DecoderOnlyTransformer(nn.Module):

    def __init__(self, omega, d, num_heads, num_decode_layers, m, tao, device):
        super().__init__()
        self.omega = omega
        self.d = d
        self.num_heads = num_heads
        self.num_decode_layers = num_decode_layers
        self.m = m
        self.tao = tao
        self.embed = nn.Embedding(omega, d)
        self.pos_encode = PositionalEncoding(d, tao)
        self.device = device

        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(d, num_heads, m, device) for _ in range(num_decode_layers)]
        )

        self.linear = nn.Linear(m, omega)

    def forward(self, x):
        embedded = self.pos_encode(self.embed(x))
        dec_in = embedded
        for dec_block in self.decoder_blocks:
            dec_out = dec_block(dec_in)

        out = self.linear(dec_out)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, m, device):
        super(DecoderBlock, self).__init__()
        self.self_atten = MultiHeadAttention(embed_size, num_heads, device)
        self.ff = PositionWiseFeedForward(embed_size, m)

    def forward(self, x):
        atten_out = self.self_atten(x, x, x)
        ff_out = self.ff(atten_out)
        return ff_out


class PositionWiseFeedForward(nn.Module):

    def __init__(self, embed_size, m):
        super(PositionWiseFeedForward, self).__init__()
        self.layer1 = nn.Linear(embed_size, m)
        # self.layer2 = nn.Linear(dim_ff, embed_size)
        self.act = nn.Tanh()  ### may need to edit this to use different activation

    def forward(self, x):
        out = self.act(self.layer1(x))
        # out = self.layer2(out)
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


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, device):
        super(MultiHeadAttention, self).__init__()
        assert embed_size % num_heads == 0, "d_model must be divisible by num_heads"

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.d_k = embed_size // num_heads
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

    def split_heads(self, x):
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, _ = x.size()
        return (
            x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_size)
        )

    def forward(self, Q, K, V):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        atten_output = self.scaled_dot_product_attention(Q, K, V)
        output = self.W_o(self.combine_heads(atten_output))
        return output
