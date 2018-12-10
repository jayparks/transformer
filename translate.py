# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import argparse

from data.data_utils import load_test_data
from data.data_utils import convert_idx2text
from transformer.translator import Translator

use_cuda = torch.cuda.is_available()


def main(opt):
    translator = Translator(opt, use_cuda)

    _, _, tgt_idx2word = torch.load(opt.vocab)['tgt_dict']
    _, test_iter = load_test_data(opt.decode_input, opt.vocab, opt.batch_size, use_cuda)

    lines = 0
    print('Translated output will be written in {}'.format(opt.decode_output))
    with open(opt.decode_output, 'w') as output:
        with torch.no_grad():
            for batch in test_iter:
                all_hyp, all_scores = translator.translate_batch(batch.src)
                for idx_seqs in all_hyp:
                    for idx_seq in idx_seqs:
                        pred_line = convert_idx2text(idx_seq, tgt_idx2word)
                        output.write(pred_line + '\n')
                lines += batch.src[0].size(0)
                print('  {} lines decoded'.format(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translation hyperparams')
    parser.add_argument('-model_path', required=True, type=str, help='Path to the test data')
    parser.add_argument('-vocab', required=True, type=str, help='Path to an existing vocabulary file')
    parser.add_argument('-decode_input', required=True, type=str, help='Path to the source file to translate')
    parser.add_argument('-decode_output', required=True, type=str, help='Path to write translated sequences' )
    parser.add_argument('-batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('-beam_size', type=int, default=5, help='Beam width')
    parser.add_argument('-n_best', type=int, default=1, help='Output the n_best decoded sentence')
    parser.add_argument('-max_decode_step', type=int, default=100, help='Maximum # of steps for decoding')

    opt = parser.parse_args()
    print(opt)
    main(opt)
    print('Terminated')