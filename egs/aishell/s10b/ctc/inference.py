#!/usr/bin/env python3

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import logging
import os
import sys

import torch
import torch.nn.functional as F

import kaldi

from common import load_checkpoint
from common import setup_logger
from dataset import get_ctc_dataloader
from model import get_ctc_model
from options import get_args
from tdnnf_model import get_tdnnf_model


def main():
    args = get_args()

    setup_logger('{}/log-inference'.format(args.dir), args.log_level)
    logging.info(' '.join(sys.argv))

    if torch.cuda.is_available() == False:
        logging.warning('No GPU detected! Use CPU for inference.')
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args.device_id)

    model = get_tdnnf_model(
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        hidden_dim=args.hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
        prefinal_bottleneck_dim=args.prefinal_bottleneck_dim,
        kernel_size_list=args.kernel_size_list,
        subsampling_factor_list=args.subsampling_factor_list)

    load_checkpoint(args.checkpoint, model)

    model.to(device)
    model.eval()

    wspecifier = 'ark,scp:{filename}.ark,{filename}.scp'.format(
        filename=os.path.join(args.dir, 'nnet_output'))

    writer = kaldi.MatrixWriter(wspecifier)

    dataloader = get_ctc_dataloader(
        feats_scp=args.feats_scp,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        model_left_context=args.model_left_context,
        model_right_context=args.model_right_context)

    for batch_idx, batch in enumerate(dataloader):
        uttid_list, feat, feat_len_list, _, _ = batch

        feat = feat.to(device)

        with torch.no_grad():
            activations, feat_len_list = model(feat, feat_len_list)

        log_probs = F.log_softmax(activations, dim=-1)

        num = len(uttid_list)
        for i in range(num):
            uttid = uttid_list[i]
            feat_len = feat_len_list[i]
            value = log_probs[i, :feat_len, :]

            value = value.cpu()

            writer.Write(uttid, value.numpy())

        if batch_idx % 10 == 0:
            logging.info('Processed batch {}/{} ({:.3f}%)'.format(
                batch_idx, len(dataloader),
                float(batch_idx) / len(dataloader) * 100))

    writer.Close()
    logging.info('pseudo-log-likelihood is saved to {}'.format(
        os.path.join(args.dir, 'nnet_output.scp')))


if __name__ == '__main__':
    main()
