#!/usr/bin/env python3

# Copyright 2019-2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

from datetime import datetime
import logging

import numpy as np

import torch

import kaldi


def setup_logger(log_filename, log_level='info'):
    now = datetime.now()
    date_time = now.strftime('%Y-%m-%d-%H-%M-%S')
    log_filename = '{}-{}'.format(log_filename, date_time)
    formatter = '%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s'
    if log_level == 'debug':
        level = logging.DEBUG
    elif log_level == 'info':
        level = logging.INFO
    elif log_level == 'warning':
        level = logging.WARNING
    logging.basicConfig(filename=log_filename,
                        format=formatter,
                        level=level,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(formatter))
    logging.getLogger('').addHandler(console)


def load_checkpoint(filename, model):
    logging.info('load checkpoint from {}'.format(filename))

    checkpoint = torch.load(filename, map_location='cpu')

    keys = ['state_dict', 'epoch', 'learning_rate', 'objf']
    for k in keys:
        assert k in checkpoint

    model.load_state_dict(checkpoint['state_dict'])

    epoch = checkpoint['epoch']
    learning_rate = checkpoint['learning_rate']
    objf = checkpoint['objf']

    return epoch, learning_rate, objf


def save_checkpoint(filename, model, epoch, learning_rate, objf):
    logging.info('Save checkpoint to {filename}: epoch={epoch}, '
                 'learning_rate={learning_rate}, objf={objf}'.format(
                     filename=filename,
                     epoch=epoch,
                     learning_rate=learning_rate,
                     objf=objf))
    checkpoint = {
        'state_dict': model.state_dict(),
        'epoch': epoch,
        'learning_rate': learning_rate,
        'objf': objf
    }
    torch.save(checkpoint, filename)


def save_training_info(filename, model_path, current_epoch, learning_rate, objf,
                       best_objf, best_epoch):
    with open(filename, 'w') as f:
        f.write('model_path: {}\n'.format(model_path))
        f.write('epoch: {}\n'.format(current_epoch))
        f.write('learning rate: {}\n'.format(learning_rate))
        f.write('objf: {}\n'.format(objf))
        f.write('best objf: {}\n'.format(best_objf))
        f.write('best epoch: {}\n'.format(best_epoch))

    logging.info('write training info to {}'.format(filename))


def load_lda_mat(lda_mat_filename):
    lda_mat = kaldi.read_mat(lda_mat_filename).numpy()
    # y = Ax + b,
    # lda contains [A, b], x is feature
    # A.rows() == b.rows()
    # b.cols() == 1
    # lda.rows() == A.rows() == b.rows()
    # lda.cols() == A.cols() + 1
    assert lda_mat.shape[0] + 1 == lda_mat.shape[1]
    lda_A = torch.from_numpy(np.transpose(lda_mat[:, :-1])).float()
    lda_b = torch.from_numpy(np.transpose(lda_mat[:, -1:])).float()
    # transpose because we use x^T * A^T + b^T
    return lda_A, lda_b


def splice_feats(x):
    '''
    Example input:
        0 1
        2 3
        4 5
        6 7
    Example output:
        0 1 2 3 4 5
        2 3 4 5 6 7

    The purpose of this function is for LDA.
    '''
    x = torch.from_numpy(x)
    # x is [T, C] where T is seq_len, C is feat_dim
    x = x.unsqueeze(0)
    x = x.unsqueeze(0)
    # now x is [1, 1, T, C]
    # we use a fixed constant 3 since kaldi usually uses 3 for LDA
    x = torch.nn.functional.unfold(x, kernel_size=(3, x.shape[-1]))
    # now x is 3-D [1, C', T']
    x = x.permute(0, 2, 1)
    # now x is 3-D [1, T', C']
    x = x.squeeze(0)
    # now x is 2-D [T', C'], where T' = T - 2, C' = 3 * C
    return x.numpy()
