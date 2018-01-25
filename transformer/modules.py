
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_normal(self.linear.weight)

    def forward(self, inputs):
        return self.linear(inputs)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=.1):
        super(ScaledDotProductAttention, self).__init__()
        self.scale_factor = np.sqrt(d_k)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        # q: [b_size x len_q x d_k]
        # k: [b_size x len_k x d_k]
        # v: [b_size x len_v x d_v] note: (len_k == len_v)
        attn = torch.bmm(q, k.transpose(1, 2)) / self.scale_factor  # attn: [b_size x len_q x len_k]
        if attn_mask is not None:
            assert attn_mask.size() == attn.size()
            attn.data.masked_fill_(attn_mask, -float('inf'))

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        outputs = torch.bmm(attn, v) # outputs: [b_size x len_q x d_v]

        return outputs, attn


class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(d_hid), requires_grad=True)
        self.eps = eps

    def forward(self, z):
        mean = z.mean(dim=-1, keepdim=True,)
        std = z.std(dim=-1, keepdim=True,)
        ln_out = (z - mean.expand_as(z)) / (std.expand_as(z) + self.eps)
        ln_out = self.gamma.expand_as(ln_out) * ln_out + self.beta.expand_as(ln_out)

        return ln_out


class PosEncoding(nn.Module):
    def __init__(self, max_seq_len, d_word_vec):
        super(PosEncoding, self).__init__()
        pos_enc = np.array(
            [[pos / np.power(10000, 2.0 * (j // 2) / d_word_vec) for j in range(d_word_vec)]
            for pos in range(max_seq_len)])
        pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])
        pad_row = np.zeros([1, d_word_vec])
        pos_enc = np.concatenate([pad_row, pos_enc]).astype(np.float32)

        # additional single row for PAD idx
        self.pos_enc = nn.Embedding(max_seq_len + 1, d_word_vec)
        # fix positional encoding: exclude weight from grad computation
        self.pos_enc.weight = nn.Parameter(torch.from_numpy(pos_enc), requires_grad=False)

    def forward(self, input_len):
        max_len = torch.max(input_len)
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        input_pos = tensor([list(range(1, len+1)) + [0]*(max_len-len) for len in input_len])

        return self.pos_enc(input_pos)