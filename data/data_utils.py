# -*- coding: utf-8 -*-
import torch
import torchtext.data as data
from torchtext.data import Field, Iterator, BucketIterator
from data.dataset import ParallelDataset
from collections import Counter

# Extra vocabulary symbols
pad_token = "<pad>"
unk_token = "<unk>"
bos_token = "<bos>"
eos_token = "<eos>"

extra_tokens = [pad_token, unk_token, bos_token, eos_token]

PAD = extra_tokens.index(pad_token)
UNK = extra_tokens.index(unk_token)
BOS = extra_tokens.index(bos_token)
EOS = extra_tokens.index(eos_token)


def convert_text2idx(examples, word2idx):
    return [[word2idx[w] if w in word2idx else UNK
            for w in sent] for sent in examples]


def convert_idx2text(example, idx2word):
    words = []
    for i in example:
        if i == EOS:
            break
        words.append(idx2word[i])
    return ' '.join(words)


def read_corpus(src_path, max_len, lower_case=False):
    print('Reading examples from {}..'.format(src_path))
    src_sents = []
    empty_lines, exceed_lines = 0, 0
    with open(src_path) as src_file:
        for idx, src_line in enumerate(src_file):
            if idx % 10000 == 0:
                print('  reading {} lines..'.format(idx))
            if src_line.strip() == '':  # remove empty lines
                empty_lines += 1
                continue
            if lower_case:  # check lower_case
                src_line = src_line.lower()

            src_words = src_line.strip().split()
            if max_len is not None and len(src_words) > max_len:
                exceed_lines += 1
                continue
            src_sents.append(src_words)

    print('Removed {} empty lines'.format(empty_lines),
          'and {} lines exceeding the length {}'.format(exceed_lines, max_len))
    print('Result: {} lines remained'.format(len(src_sents)))
    return src_sents


def read_parallel_corpus(src_path, tgt_path, max_len, lower_case=False):
    print ('Reading examples from {} and {}..'.format(src_path, tgt_path))
    src_sents, tgt_sents = [], []
    empty_lines, exceed_lines = 0, 0
    with open(src_path) as src_file, open(tgt_path) as tgt_file:
        for idx, (src_line, tgt_line) in enumerate(zip(src_file, tgt_file)):
            if idx % 10000 == 0:
                print('  reading {} lines..'.format(idx))
            if src_line.strip() == '' or tgt_line.strip() == '': # remove empty lines
                empty_lines += 1
                continue
            if lower_case:  # check lower_case
                src_line = src_line.lower()
                tgt_line = tgt_line.lower()

            src_words = src_line.strip().split()
            tgt_words = tgt_line.strip().split()
            if max_len is not None and (len(src_words) > max_len or len(tgt_words) > max_len):
                exceed_lines += 1
                continue
            src_sents.append(src_words)
            tgt_sents.append(tgt_words)

    print ('Filtered {} empty lines'.format(empty_lines),
           'and {} lines exceeding the length {}'.format(exceed_lines, max_len))
    print ('Result: {} lines remained'.format(len(src_sents)))
    return src_sents, tgt_sents


def build_vocab(examples, max_size, min_freq, extra_tokens):
    print ('Creating vocabulary with max limit {}..'.format(max_size))
    counter = Counter()
    word2idx, idx2word = {}, []
    if extra_tokens:
        idx2word += extra_tokens
        word2idx = {word: idx for idx, word in enumerate(extra_tokens)}
    min_freq = max(min_freq, 1)
    max_size = max_size + len(idx2word) if max_size else None
    for sent in examples:
        for w in sent:
            counter.update([w])
    # first sort items in alphabetical order and then by frequency
    sorted_counter = sorted(counter.items(), key=lambda tup: tup[0])
    sorted_counter.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in sorted_counter:
        if freq < min_freq or (max_size and len(idx2word) == max_size):
            break
        idx2word.append(word)
        word2idx[word] = len(idx2word) - 1

    print ('Vocabulary of size {} has been created'.format(len(idx2word)))
    return counter, word2idx, idx2word


def load_train_data(data_path, batch_size, max_src_len, max_trg_len, use_cuda=False):
    # Note: sequential=False, use_vocab=False, since we use preprocessed inputs.
    src_field = Field(sequential=True, use_vocab=False, include_lengths=True, batch_first=True,
                      pad_token=PAD, unk_token=UNK, init_token=None, eos_token=None,)
    trg_field = Field(sequential=True, use_vocab=False, include_lengths=True, batch_first=True,
                      pad_token=PAD, unk_token=UNK, init_token=BOS, eos_token=EOS,)
    fields = (src_field, trg_field)
    device = None if use_cuda else -1

    def filter_pred(example):
        if len(example.src) <= max_src_len and len(example.trg) <= max_trg_len:
            return True
        return False

    dataset = torch.load(data_path)
    train_src, train_tgt = dataset['train_src'], dataset['train_tgt']
    dev_src, dev_tgt = dataset['dev_src'], dataset['dev_tgt']

    train_data = ParallelDataset(train_src, train_tgt, fields=fields, filter_pred=filter_pred,)
    train_iter = Iterator(dataset=train_data, batch_size=batch_size, train=True, # Variable(volatile=False)
                          sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg)),
                          repeat=False, shuffle=True, device=device)
    dev_data = ParallelDataset(dev_src, dev_tgt, fields=fields,)
    dev_iter = Iterator(dataset=dev_data, batch_size=batch_size, train=False,    # Variable(volatile=True)
                        repeat=False, device=device, shuffle=False, sort=False,)

    return src_field, trg_field, train_iter, dev_iter


def load_test_data(data_path, vocab_path, batch_size, use_cuda=False):
    # Note: sequential=False, use_vocab=False, since we use preprocessed inputs.
    src_field = Field(sequential=True, use_vocab=False, include_lengths=True, batch_first=True,
                      pad_token=PAD, unk_token=UNK, init_token=None, eos_token=None,)
    fields = (src_field, None)
    device = None if use_cuda else -1

    vocab = torch.load(vocab_path)
    _, src_word2idx, _ = vocab['src_dict']
    lower_case = vocab['lower_case']

    test_src = convert_text2idx(read_corpus(data_path, None, lower_case), src_word2idx)
    test_data = ParallelDataset(test_src, None, fields=fields,)
    test_iter = Iterator(dataset=test_data, batch_size=batch_size, train=False,  # Variable(volatile=True)
                         repeat=False, device=device, shuffle=False, sort=False)

    return src_field, test_iter

