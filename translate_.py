# -*- coding: utf-8 -*-
import torch
import argparse
from data.data_utils import load_test_data
from data.data_utils import convert_idx2text

from transformer.models import Transformer
from data import data_utils

use_cuda = torch.cuda.is_available()


def main(opt):
    model_state = torch.load(opt.model_path)
    model_opt = model_state['opt']

    model = Transformer(model_opt.n_layers, model_opt.d_k, model_opt.d_v, model_opt.d_model, model_opt.d_ff, model_opt.d_word_vec, model_opt.n_heads,
                        model_opt.max_src_seq_len, model_opt.max_tgt_seq_len, model_opt.src_vocab_size, model_opt.tgt_vocab_size,
                        model_opt.dropout, model_opt.share_proj_weight, model_opt.share_embs_weight, model_opt.weighted_model)

    model.load_state_dict(model_state['model_params'])
    model.eval()
    if use_cuda:
        print('Using GPU..')
        model = model.cuda()

    _, _,  tgt_idx2word = torch.load(opt.tgt_vocab)
    _, test_iter = load_test_data(opt.decode_input, opt.src_vocab, opt.batch_size, use_cuda)

    lines = 0
    print ('Translated output will be written in {}'.format(opt.decode_output))

    with open(opt.decode_output, 'w') as output:
        for batch in test_iter:

            enc_inputs, enc_inputs_len = batch.src
            print('enc_inputs\n', enc_inputs)
            batch_size = enc_inputs.size(0)
            preds_prev = torch.zeros(batch_size, opt.max_decode_step).long()
            preds_prev[:, 0] += data_utils.BOS

            preds_prev = torch.autograd.Variable(preds_prev)
            preds = torch.autograd.Variable(torch.zeros(batch_size, opt.max_decode_step).long())

            enc_outputs, _ = model.encode(enc_inputs, enc_inputs_len)
            print('enc_outputs\n', enc_outputs)

            for t in range(opt.max_decode_step):
                print('pred_prev\n', preds_prev)
                lens = torch.ones(batch_size).fill_(t+1).long()
                dec_outputs, *_ = model.decode(preds_prev[:, :t+1], lens, enc_inputs, enc_outputs)
                logits = model.tgt_proj(dec_outputs)

                # outputs: [batch_size, max_decode_step]
                outputs = torch.max(logits, dim=-1)[1].view(batch_size, -1)
                print(outputs)
                preds[:, t] = outputs[:, t].data
                if t < opt.max_decode_step - 1:
                    preds_prev[:, t + 1] = outputs[:, t]

            for i in range(len(preds)):
                output.write(convert_idx2text(preds[i].data.tolist(), tgt_idx2word) + '\n')
                output.flush()

            lines += batch_size
            print ('  {} lines decoded'.format(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translation hyperparams')
    parser.add_argument('-model_path', required=True, type=str, help='Path to the test data')
    parser.add_argument('-src_vocab', required=True, type=str, help='Path to an existing source vocab')
    parser.add_argument('-tgt_vocab', required=True, type=str, help='Path to an existing target vocab')
    parser.add_argument('-decode_input', required=True, type=str, help='Path to the source file to translate')
    parser.add_argument('-decode_output', required=True, type=str, help='Path to write translated sequences' )
    parser.add_argument('-batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('-beam_size', type=int, default=5, help='Beam width')
    parser.add_argument('-n_best', type=int, default=1, help='Output the n_best decoded sentence')
    parser.add_argument('-max_decode_step', default=100, type=int, help='Maximum # of steps for decoding')

    opt = parser.parse_args()
    print(opt)
    main(opt)
    print ('Terminated')