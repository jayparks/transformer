
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from data import data_utils

from transformer.modules import Linear
from transformer.modules import PosEncoding
from transformer.layers import EncoderLayer, DecoderLayer, \
                               WeightedEncoderLayer, WeightedDecoderLayer


def get_attn_pad_mask(seq_q, seq_k):
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    b_size, len_q = seq_q.size()
    b_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(data_utils.PAD).unsqueeze(1)  # b_size x 1 x len_k
    pad_attn_mask = pad_attn_mask.expand(b_size, len_q, len_k) # b_size x len_q x len_k

    return pad_attn_mask


def get_attn_subsequent_mask(seq):
    assert seq.dim() == 2
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()

    return subsequent_mask


class Encoder(nn.Module):
    def __init__(self, n_layers, d_k, d_v, d_model, d_ff, n_heads,
                 max_seq_len, src_vocab_size, dropout=0.1, weighted=False):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=data_utils.PAD,)
        self.pos_emb = PosEncoding(max_seq_len * 10, d_model) # TODO: *10 fix
        self.dropout_emb = nn.Dropout(dropout)
        self.layer_type = EncoderLayer if not weighted else WeightedEncoderLayer
        self.layers = nn.ModuleList(
            [self.layer_type(d_k, d_v, d_model, d_ff, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, enc_inputs, enc_inputs_len, return_attn=False):
        enc_outputs = self.src_emb(enc_inputs) * (self.d_model ** 0.5)
        enc_outputs += self.pos_emb(enc_inputs_len) # Adding positional encoding TODO: note
        enc_outputs = self.dropout_emb(enc_outputs)

        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            if return_attn:
                enc_self_attns.append(enc_self_attn)

        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    def __init__(self, n_layers, d_k, d_v, d_model, d_ff, n_heads,
                 max_seq_len, tgt_vocab_size, dropout=0.1, weighted=False):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=data_utils.PAD, )
        self.pos_emb = PosEncoding(max_seq_len * 10, d_model) # TODO: *10 fix
        self.dropout_emb = nn.Dropout(dropout)
        self.layer_type = DecoderLayer if not weighted else WeightedDecoderLayer
        self.layers = nn.ModuleList(
            [self.layer_type(d_k, d_v, d_model, d_ff, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, dec_inputs, dec_inputs_len, enc_inputs, enc_outputs, return_attn=False):
        dec_outputs = self.tgt_emb(dec_inputs) * (self.d_model ** 0.5)
        dec_outputs += self.pos_emb(dec_inputs_len) # Adding positional encoding # TODO: note
        dec_outputs = self.dropout_emb(dec_outputs)

        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)

        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        dec_enc_attn_pad_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs,
                                                             self_attn_mask=dec_self_attn_mask,
                                                             enc_attn_mask=dec_enc_attn_pad_mask)
            if return_attn:
                dec_self_attns.append(dec_self_attn)
                dec_enc_attns.append(dec_enc_attn)

        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self, opt):
        super(Transformer, self).__init__()
        self.encoder = Encoder(opt.n_layers, opt.d_k, opt.d_v, opt.d_model, opt.d_ff, opt.n_heads,
                               opt.max_src_seq_len, opt.src_vocab_size, opt.dropout, opt.weighted_model)
        self.decoder = Decoder(opt.n_layers, opt.d_k, opt.d_v, opt.d_model, opt.d_ff, opt.n_heads,
                               opt.max_tgt_seq_len, opt.tgt_vocab_size, opt.dropout, opt.weighted_model)
        self.tgt_proj = Linear(opt.d_model, opt.tgt_vocab_size, bias=False)

        if opt.share_proj_weight:
            print('Sharing target embedding and projection..')

            self.tgt_proj.weight = self.decoder.tgt_emb.weight

        if opt.share_embs_weight:
            print('Sharing source and target embedding..')
            assert opt.src_vocab_size == opt.tgt_vocab_size, \
                'To share word embeddings, the vocabulary size of src/tgt should be the same'
            self.encoder.src_emb.weight = self.decoder.tgt_emb.weight

    def trainable_params(self):
        # Avoid updating the position encoding
        return filter(lambda p: p.requires_grad, self.parameters())

    def encode(self, enc_inputs, enc_inputs_len, return_attn=False):
        return self.encoder(enc_inputs, enc_inputs_len, return_attn)

    def decode(self, dec_inputs, dec_inputs_len, enc_inputs, enc_outputs, return_attn=False):
        return self.decoder(dec_inputs, dec_inputs_len, enc_inputs, enc_outputs, return_attn)

    def forward(self, enc_inputs, enc_inputs_len, dec_inputs, dec_inputs_len, return_attn=False):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs, enc_inputs_len, return_attn)
        dec_outputs, dec_self_attns, dec_enc_attns = \
            self.decoder(dec_inputs, dec_inputs_len, enc_inputs, enc_outputs, return_attn)
        dec_logits = self.tgt_proj(dec_outputs)

        return dec_logits.view(-1, dec_logits.size(-1)), \
               enc_self_attns, dec_self_attns, dec_enc_attns
