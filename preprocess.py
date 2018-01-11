# -*- coding: utf-8 -*-

from __future__ import print_function
import torch
import argparse

from data import data_utils
from data.data_utils import read_parallel_corpus
from data.data_utils import build_vocab
from data.data_utils import convert_text2idx


def main(opt):
    train_src, train_tgt = read_parallel_corpus(opt.train_src, opt.train_tgt, opt.max_len, opt.lower_case)
    dev_src, dev_tgt = read_parallel_corpus(opt.dev_src, opt.dev_tgt, None, opt.lower_case)

    if opt.vocab:
        src_counter, src_word2idx, src_idx2word, = torch.load(opt.vocab)['src_dict']
        tgt_counter, tgt_word2idx, tgt_idx2word, = torch.load(opt.vocab)['tgt_dict']
    else:
        if opt.share_vocab:
            print('Building shared vocabulary')
            vocab_size = min(opt.src_vocab_size, opt.tgt_vocab_size) \
                if (opt.src_vocab_size is not None and opt.tgt_vocab_size is not None) else None
            counter, word2idx, idx2word = build_vocab(train_src + train_tgt, vocab_size,
                                                      opt.min_word_count, data_utils.extra_tokens)
            src_counter, src_word2idx, src_idx2word = (counter, word2idx, idx2word)
            tgt_counter, tgt_word2idx, tgt_idx2word = (counter, word2idx, idx2word)
        else:
            src_counter, src_word2idx, src_idx2word = build_vocab(train_src, opt.src_vocab_size,
                                                                  opt.min_word_count, data_utils.extra_tokens)
            tgt_counter, tgt_word2idx, tgt_idx2word = build_vocab(train_tgt, opt.tgt_vocab_size,
                                                                  opt.min_word_count, data_utils.extra_tokens)
    train_src, train_tgt = \
        convert_text2idx(train_src, src_word2idx), convert_text2idx(train_tgt, tgt_word2idx)
    dev_src, dev_tgt = \
        convert_text2idx(dev_src, src_word2idx), convert_text2idx(dev_tgt, tgt_word2idx)

    # Save source/target vocabulary and train/dev data
    torch.save(
        {
            'src_dict'  : (src_counter, src_word2idx, src_idx2word),
            'tgt_dict'  : (tgt_counter, tgt_word2idx, tgt_idx2word),
            'src_path'  : opt.train_src,
            'tgt_path'  : opt.train_tgt,
            'lower_case': opt.lower_case
        }
        ,'{}.dict'.format(opt.save_data)
    )
    torch.save(
        {
            'train_src': train_src,     'train_tgt': train_tgt,
            'dev_src'  : dev_src,       'dev_tgt'  : dev_tgt,
            'src_dict' : src_word2idx,  'tgt_dict' : tgt_word2idx,
        }
        , '{}-train.t7'.format(opt.save_data)
    )
    print('Saved the vocabulary at {}.dict'.format(opt.save_data))
    print('Saved the preprocessed train/dev data at {}-train.t7'.format(opt.save_data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing')

    parser.add_argument('-train_src', required=True, type=str, help='Path to training source data')
    parser.add_argument('-train_tgt', required=True, type=str, help='Path to training target data')
    parser.add_argument('-dev_src', required=True, type=str, help='Path to devation source data')
    parser.add_argument('-dev_tgt', required=True, type=str, help='Path to devation target data')
    parser.add_argument('-vocab', type=str, help='Path to an existing vocabulary file')
    parser.add_argument('-src_vocab_size', type=int, help='Source vocabulary size')
    parser.add_argument('-tgt_vocab_size', type=int, help='Target vocabulary size')
    parser.add_argument('-min_word_count', type=int, default=1)
    parser.add_argument('-max_len', type=int, default=50, help='Maximum sequence length')
    parser.add_argument('-lower_case', action='store_true')
    parser.add_argument('-share_vocab', action='store_true')
    parser.add_argument('-save_data', required=True, type=str, help='Output file for the prepared data')

    opt = parser.parse_args()
    print(opt)
    main(opt)