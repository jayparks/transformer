
import torch
import torch.nn as nn
import torch.nn.init as init

from transformer.modules import Linear
from transformer.modules import ScaledDotProductAttention
from transformer.modules import LayerNormalization


class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads

        self.w_q = nn.Parameter(torch.FloatTensor(n_heads, d_model, d_k))
        self.w_k = nn.Parameter(torch.FloatTensor(n_heads, d_model, d_k))
        self.w_v = nn.Parameter(torch.FloatTensor(n_heads, d_model, d_v))

        self.attention = ScaledDotProductAttention(d_k, dropout)
        self.layer_norm = LayerNormalization(d_model)
        self.proj = Linear(n_heads * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        init.xavier_normal(self.w_q)
        init.xavier_normal(self.w_k)
        init.xavier_normal(self.w_v)

    def forward(self, q, k, v, attn_mask):
        (d_k, d_v, d_model, n_heads) = (self.d_k, self.d_v, self.d_model, self.n_heads)
        residual = q

        b_size, len_q, d_model = q.size()  # q: [b_size x len_q x d_model]
        b_size, len_k, d_model = k.size()  # k: [b_size x len_k x d_model]
        b_size, len_v, d_model = v.size()  # v: [b_size x len_v x d_model]

        q_s = q.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)  # [n_heads x b_size * len_q x d_model]
        k_s = k.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)  # [n_heads x b_size * len_k x d_model]
        v_s = v.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)  # [n_heads x b_size * len_v x d_model]

        q_s = torch.bmm(q_s, self.w_q).view(-1, len_q, d_k)  # [b_size * n_heads x len_q x d_k]
        k_s = torch.bmm(k_s, self.w_k).view(-1, len_k, d_k)  # [b_size * n_heads x len_k x d_k]
        v_s = torch.bmm(v_s, self.w_v).view(-1, len_v, d_v)  # [b_size * n_heads x len_v x d_v]

        # perform attention, result_size = [b_size * n_branches x len_q x d_v]
        outputs, attn = self.attention(q_s, k_s, v_s, attn_mask=attn_mask.repeat(n_heads, 1, 1))

        # back to original batch_size, result_size = [b_size x len_q x d_v * n_heads]
        outputs = torch.cat(torch.split(outputs, b_size, dim=0), dim=-1)

        # project back to residual size, result_size = [b_size x len_q x d_model]
        outputs = self.proj(outputs)
        outputs = self.dropout(outputs)

        return self.layer_norm(residual + outputs), attn


class MultiBranchAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_branches, dropout, is_decoder=False):
        super(MultiBranchAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_branches = n_branches
        self.is_decoder = is_decoder

        if is_decoder:
            self.w_l = nn.Parameter(torch.FloatTensor(n_branches, d_model, d_model))
        self.w_q = nn.Parameter(torch.FloatTensor(n_branches, d_model, d_k))
        self.w_k = nn.Parameter(torch.FloatTensor(n_branches, d_model, d_k))
        self.w_v = nn.Parameter(torch.FloatTensor(n_branches, d_model, d_v))

        # additional weights for BranchedAttention
        self.w_o = nn.Parameter(torch.FloatTensor(n_branches, d_v, d_model))
        self.w_kp = nn.Parameter(torch.rand(n_branches))
        self.kp_softmax = nn.Softmax(dim=0)
        self.w_a = nn.Parameter(torch.rand(n_branches))
        self.a_softmax = nn.Softmax(dim=0)

        self.attention = ScaledDotProductAttention(d_k, dropout)
        self.pos_ffn = nn.ModuleList([
            PoswiseFeedForwardNet(d_model, d_ff, dropout) for _ in range(n_branches)])
        self.layer_norm = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)

        if is_decoder:
            init.xavier_normal(self.w_l)
        init.xavier_normal(self.w_q)
        init.xavier_normal(self.w_k)
        init.xavier_normal(self.w_v)
        init.xavier_normal(self.w_o)

    def forward(self, q, k, v, attn_mask):
        (d_k, d_v, d_model, n_branches) = (self.d_k, self.d_v, self.d_model, self.n_branches)
        residual = q

        b_size, len_q, d_model = q.size()  # q: [b_size x len_q x d_model]
        b_size, len_k, d_model = k.size()  # k: [b_size x len_k x d_model]
        b_size, len_v, d_model = v.size()  # v: [b_size x len_v x d_model]

        q_s = q.repeat(n_branches, 1, 1).view(n_branches, -1, d_model)  # [n_branches x b_size * len_q x d_model]
        k_s = k.repeat(n_branches, 1, 1).view(n_branches, -1, d_model)  # [n_branches x b_size * len_k x d_model]
        v_s = v.repeat(n_branches, 1, 1).view(n_branches, -1, d_model)  # [n_branches x b_size * len_v x d_model]

        if self.is_decoder:
            q_s = torch.bmm(q_s, self.w_l)
        q_s = torch.bmm(q_s, self.w_q).view(-1, len_q, d_k)  # [b_size * n_branches x len_q x d_k]
        k_s = torch.bmm(k_s, self.w_k).view(-1, len_k, d_k)  # [b_size * n_branches x len_k x d_k]
        v_s = torch.bmm(v_s, self.w_v).view(-1, len_v, d_v)  # [b_size * n_branches x len_v x d_v]

        # perform attention, result_size = [b_size * n_branches x len_q x d_v]
        outputs, attn = self.attention(q_s, k_s, v_s, attn_mask=attn_mask.repeat(n_branches, 1, 1))
        outputs = torch.bmm(outputs.view(n_branches, -1, d_v), self.w_o) # [n_branches x b_size * len_q x d_model]
        #outputs = self.dropout(outputs)

        outputs = self.kp_softmax(self.w_kp).view(-1, 1, 1) * outputs
        outputs = [out.squeeze(0).view(-1, len_q, d_model) \
                   for out in torch.split(outputs, split_size=1, dim=0)] # [b_size x len_q x d_model] x n_branches
        outputs = torch.cat([pos_ffn(output) for output, pos_ffn in \
                             zip(outputs, self.pos_ffn)], dim=0).view(n_branches, -1, d_model)
        outputs = self.a_softmax(self.w_a).view(-1, 1, 1) * outputs
        outputs = torch.sum(outputs, dim=0).view(-1, len_q, d_model) # [b_size x len_q x d_model]
        outputs = self.dropout(outputs)

        return self.layer_norm(residual + outputs), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PoswiseFeedForwardNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)

        init.kaiming_normal(self.conv1.weight)
        init.xavier_normal(self.conv2.weight)

    def forward(self, inputs):
        residual = inputs # inputs: [b_size x len_q x d_model]
        outputs = self.relu(self.conv1(inputs.transpose(1, 2)))
        outputs = self.conv2(outputs).transpose(1, 2) # outputs: [b_size x len_q x d_model]
        outputs = self.dropout(outputs)

        return self.layer_norm(residual + outputs)


