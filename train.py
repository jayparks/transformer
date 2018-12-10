from __future__ import print_function
import os
import sys
import time
import math
import argparse

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm
import torch.optim as optim

from data import data_utils
from data.data_utils import load_train_data
from transformer.models import Transformer
from transformer.optimizer import ScheduledOptimizer

use_cuda = torch.cuda.is_available()


def create_model(opt):
    data = torch.load(opt.data_path)
    opt.src_vocab_size = len(data['src_dict'])
    opt.tgt_vocab_size = len(data['tgt_dict'])

    print('Creating new model parameters..')
    model = Transformer(opt)  # Initialize a model state.
    model_state = {'opt': opt, 'curr_epochs': 0, 'train_steps': 0}

    # If opt.model_path exists, load model parameters.
    if os.path.exists(opt.model_path):
        print('Reloading model parameters..')
        model_state = torch.load(opt.model_path)
        model.load_state_dict(model_state['model_params'])

    if use_cuda:
        print('Using GPU..')
        model = model.cuda()

    return model, model_state


def main(opt):
    print('Loading training and development data..')
    _, _, train_iter, dev_iter = load_train_data(opt.data_path, opt.batch_size,
                                                 opt.max_src_seq_len, opt.max_tgt_seq_len, use_cuda)
    # Create a new model or load an existing one.
    model, model_state = create_model(opt)
    init_epoch = model_state['curr_epochs']
    if init_epoch >= opt.max_epochs:
        print('Training is already complete.',
              'current_epoch:{}, max_epoch:{}'.format(init_epoch, opt.max_epochs))
        sys.exit(0)

    # Loss and Optimizer
    # If size_average=True (default): Loss for a mini-batch is averaged over non-ignore index targets.
    criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=data_utils.PAD)
    optimizer = ScheduledOptimizer(optim.Adam(model.trainable_params(), betas=(0.9, 0.98), eps=1e-9),
                                   opt.d_model, opt.n_layers, opt.n_warmup_steps)
    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_dev_file = opt.log + '.valid.log'
        if not os.path.exists(log_train_file) and os.path.exists(log_dev_file):
            with open(log_train_file, 'w') as log_tf, open(log_dev_file, 'w') as log_df:
                log_tf.write('epoch,ppl,sents_seen\n')
                log_df.write('epoch,ppl,sents_seen\n')
        print('Training and validation log will be written in {} and {}'
              .format(log_train_file, log_dev_file))

    for epoch in range(init_epoch + 1, opt.max_epochs + 1):
        # Execute training steps for 1 epoch.
        train_loss, train_sents = train(model, criterion, optimizer, train_iter, model_state)
        print('Epoch {}'.format(epoch), 'Train_ppl: {0:.2f}'.format(train_loss),
              'Sents seen: {}'.format(train_sents))

        # Execute a validation step.
        eval_loss, eval_sents = eval(model, criterion, dev_iter)
        print('Epoch {}'.format(epoch), 'Eval_ppl: {0:.2f}'.format(eval_loss),
              'Sents seen: {}'.format(eval_sents))

        # Save the model checkpoint in every 1 epoch.
        model_state['curr_epochs'] += 1
        model_state['model_params'] = model.state_dict()
        torch.save(model_state, opt.model_path)
        print('The model checkpoint file has been saved')

        if opt.log and log_train_file and log_dev_file:
            with open(log_train_file, 'a') as log_tf, open(log_dev_file, 'a') as log_df:
                log_tf.write('{epoch},{ppl:0.2f},{sents}\n'.format(
                    epoch=epoch, ppl=train_loss, sents=train_sents, ))
                log_df.write('{epoch},{ppl:0.2f},{sents}\n'.format(
                    epoch=epoch, ppl=eval_loss, sents=eval_sents, ))


def train(model, criterion, optimizer, train_iter, model_state):  # TODO: fix opt
    model.train()
    opt = model_state['opt']
    train_loss, train_loss_total = 0.0, 0.0
    n_words, n_words_total = 0, 0
    n_sents, n_sents_total = 0, 0
    start_time = time.time()

    for batch_idx, batch in enumerate(train_iter):
        enc_inputs, enc_inputs_len = batch.src
        dec_, dec_inputs_len = batch.trg
        dec_inputs = dec_[:, :-1]
        dec_targets = dec_[:, 1:]
        dec_inputs_len = dec_inputs_len - 1

        # Execute a single training step: forward
        optimizer.zero_grad()
        dec_logits, _, _, _ = model(enc_inputs, enc_inputs_len,
                                    dec_inputs, dec_inputs_len)
        step_loss = criterion(dec_logits, dec_targets.contiguous().view(-1))

        # Execute a single training step: backward
        step_loss.backward()
        if opt.max_grad_norm:
            clip_grad_norm(model.trainable_params(), float(opt.max_grad_norm))
        optimizer.step()
        optimizer.update_lr()
        model.proj_grad()  # works only for weighted transformer

        train_loss_total += float(step_loss.data[0])
        n_words_total += torch.sum(dec_inputs_len)
        n_sents_total += dec_inputs_len.size(0)  # batch_size
        model_state['train_steps'] += 1

        # Display training status
        if model_state['train_steps'] % opt.display_freq == 0:
            loss_int = (train_loss_total - train_loss)
            n_words_int = (n_words_total - n_words)
            n_sents_int = (n_sents_total - n_sents)

            loss_per_words = loss_int / n_words_int
            avg_ppl = math.exp(loss_per_words) if loss_per_words < 300 else float("inf")
            time_elapsed = (time.time() - start_time)
            step_time = time_elapsed / opt.display_freq

            n_words_sec = n_words_int / time_elapsed
            n_sents_sec = n_sents_int / time_elapsed

            print('Epoch {0:<3}'.format(model_state['curr_epochs']), 'Step {0:<10}'.format(model_state['train_steps']),
                  'Perplexity {0:<10.2f}'.format(avg_ppl), 'Step-time {0:<10.2f}'.format(step_time),
                  '{0:.2f} sents/s'.format(n_sents_sec), '{0:>10.2f} words/s'.format(n_words_sec))
            train_loss, n_words, n_sents = (train_loss_total, n_words_total, n_sents_total)
            start_time = time.time()

    # return per_word_loss over 1 epoch
    return math.exp(train_loss_total / n_words_total), n_sents_total


def eval(model, criterion, dev_iter):
    model.eval()
    eval_loss_total = 0.0
    n_words_total, n_sents_total = 0, 0

    print('Evaluation')
    with torch.no_grad():
        for batch_idx, batch in enumerate(dev_iter):
            enc_inputs, enc_inputs_len = batch.src
            dec_, dec_inputs_len = batch.trg
            dec_inputs = dec_[:, :-1]
            dec_targets = dec_[:, 1:]
            dec_inputs_len = dec_inputs_len - 1

            dec_logits, *_ = model(enc_inputs, enc_inputs_len, dec_inputs, dec_inputs_len)
            step_loss = criterion(dec_logits, dec_targets.contiguous().view(-1))
            eval_loss_total += float(step_loss.data[0])
            n_words_total += torch.sum(dec_inputs_len)
            n_sents_total += dec_inputs_len.size(0)
            print('  {} samples seen'.format(n_sents_total))

    # return per_word_loss
    return math.exp(eval_loss_total / n_words_total), n_sents_total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Hyperparams')
    # data loading params
    parser.add_argument('-data_path', required=True, help='Path to the preprocessed data')

    # network params
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)
    parser.add_argument('-d_ff', type=int, default=2048)
    parser.add_argument('-n_heads', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-share_proj_weight', action='store_true')
    parser.add_argument('-share_embs_weight', action='store_true')
    parser.add_argument('-weighted_model', action='store_true')

    # training params
    parser.add_argument('-lr', type=float, default=0.0002)
    parser.add_argument('-max_epochs', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-max_src_seq_len', type=int, default=50)
    parser.add_argument('-max_tgt_seq_len', type=int, default=50)
    parser.add_argument('-max_grad_norm', type=float, default=None)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)
    parser.add_argument('-display_freq', type=int, default=100)
    parser.add_argument('-log', default=None)
    parser.add_argument('-model_path', type=str, required=True)

    opt = parser.parse_args()
    print(opt)
    main(opt)
    print('Terminated')
