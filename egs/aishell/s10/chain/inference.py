#!/usr/bin/env python3

# Copyright 2019 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import logging
import os
import sys

import torch
from torch.utils.dlpack import to_dlpack

import kaldi

from common import load_checkpoint
from common import setup_logger
from feat_dataset import get_feat_dataloader
from model import get_chain_model
from options import get_args


def main():
    args = get_args()
    setup_logger('{}/log-inference'.format(args.dir), args.log_level)
    logging.info(' '.join(sys.argv))

    if torch.cuda.is_available() == False:
        logging.warning('No GPU detected! Use CPU for inference.')
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args.device_id)

    model = get_chain_model(feat_dim=args.feat_dim,
                            output_dim=args.output_dim,
                            lda_mat_filename=args.lda_mat_filename,
                            hidden_dim=args.hidden_dim,
                            kernel_size_list=args.kernel_size_list,
                            stride_list=args.stride_list)

    load_checkpoint(args.checkpoint, model)

    model.to(device)
    model.eval()

    specifier = 'ark,scp:{filename}.ark,{filename}.scp'.format(
        filename=os.path.join(args.dir, 'nnet_output'))

    if args.save_as_compressed:
        Writer = kaldi.CompressedMatrixWriter
        Matrix = kaldi.CompressedMatrix
    else:
        Writer = kaldi.MatrixWriter
        Matrix = kaldi.FloatMatrix

    writer = Writer(specifier)

    dataloader = get_feat_dataloader(
        feats_scp=args.feats_scp,
        model_left_context=args.model_left_context,
        model_right_context=args.model_right_context,
        batch_size=32)

    for batch_idx, batch in enumerate(dataloader):
        key_list, padded_feat, output_len_list = batch
        padded_feat = padded_feat.to(device)
        with torch.no_grad():
            nnet_output, _ = model(padded_feat)

        num = len(key_list)
        for i in range(num):
            key = key_list[i]
            output_len = output_len_list[i]
            value = nnet_output[i, :output_len, :]
            value = value.cpu()

            m = kaldi.SubMatrixFromDLPack(to_dlpack(value))
            m = Matrix(m)
            writer.Write(key, m)

        if batch_idx % 10 == 0:
            logging.info('Processed batch {}/{} ({:.6f}%)'.format(
                batch_idx, len(dataloader),
                float(batch_idx) / len(dataloader) * 100))

    writer.Close()
    logging.info('pseudo-log-likelihood is saved to {}'.format(
        os.path.join(args.dir, 'nnet_output.scp')))


if __name__ == '__main__':
    main()
