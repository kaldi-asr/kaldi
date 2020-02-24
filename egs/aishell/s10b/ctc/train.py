#!/usr/bin/env python3

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import logging
import os
import sys
import warnings

# disable warnings when loading tensorboard
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
import torch.nn.functional as F

import kaldi

from common import setup_logger
from ctc_loss import CTCLoss
from dataset import get_ctc_dataloader
from model import get_ctc_model
from options import get_args


def main():
    args = get_args()
    setup_logger('{}/log-train'.format(args.dir), args.log_level)
    logging.info(' '.join(sys.argv))

    if torch.cuda.is_available() == False:
        logging.error('No GPU detected!')
        sys.exit(-1)

    kaldi.SelectGpuDevice(device_id=args.device_id)

    device = torch.device('cuda', args.device_id)

    model = get_ctc_model(input_dim=args.input_dim,
                          output_dim=args.output_dim,
                          num_layers=args.num_layers,
                          hidden_dim=args.hidden_dim,
                          proj_dim=args.proj_dim)

    model.to(device)

    dataloader = get_ctc_dataloader(feats_scp=args.feats_scp,
                                    labels_scp=args.labels_scp,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=8)

    lr = args.learning_rate
    optimizer = optim.Adam(model.parameters(),
                           lr=lr,
                           weight_decay=args.l2_regularize)

    model.train()

    loss_func = CTCLoss(use_warp_ctc=True, blank=0, reduction='mean')

    for epoch in range(args.num_epochs):
        learning_rate = lr * pow(0.4, epoch)

        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        logging.info('epoch {}, learning rate {}'.format(epoch, learning_rate))

        for batch_idx, batch in enumerate(dataloader):
            uttidlist, feat, feat_len_list, label_list, label_len_list = batch

            feat = feat.to(device)
            log_probs = model(feat, feat_len_list)

            # at this point log_probs is of shape: [batch_size, seq_len, output_dim]
            # CTCLoss requires a layout: [seq_len, batch_size, output_dim]

            log_probs = log_probs.permute(1, 0, 2)
            # now log_probs is of shape [seq_len, batch_size, output_dim]

            targets = torch.tensor(label_list)

            input_lengths = torch.tensor(feat_len_list)

            target_lengths = torch.tensor(label_len_list)

            loss = loss_func(log_probs=log_probs,
                             targets=targets,
                             input_lengths=input_lengths,
                             target_lengths=target_lengths)

            optimizer.zero_grad()

            loss.backward()

            clip_grad_value_(model.parameters(), 5.0)

            optimizer.step()

            logging.info('batch {}, loss {}'.format(batch_idx, loss.item()))


if __name__ == '__main__':
    torch.manual_seed(20200221)
    main()
