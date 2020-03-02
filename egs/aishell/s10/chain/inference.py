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
from device_utils import allocate_gpu_devices
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
        devices = allocate_gpu_devices(1)
        if len(devices) != 1:
            logging.error('Allocate GPU failed!')
            sys.exit(-1)
        device = torch.device('cuda', devices[0][0])

    model = get_chain_model(
        feat_dim=args.feat_dim,
        output_dim=args.output_dim,
        ivector_dim=args.ivector_dim,
        lda_mat_filename=args.lda_mat_filename,
        hidden_dim=args.hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
        prefinal_bottleneck_dim=args.prefinal_bottleneck_dim,
        kernel_size_list=args.kernel_size_list,
        subsampling_factor_list=args.subsampling_factor_list)

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
        ivector_scp=args.ivector_scp,
        model_left_context=args.model_left_context,
        model_right_context=args.model_right_context,
        batch_size=1,
        num_workers=0)
    subsampling_factor = 3
    subsampled_frames_per_chunk = args.frames_per_chunk // subsampling_factor
    for batch_idx, batch in enumerate(dataloader):
        key_list, padded_feat, output_len_list, padded_ivector, ivector_len_list = batch
        padded_feat = padded_feat.to(device)
        if ivector_len_list:
            padded_ivector = padded_ivector.to(device)
        with torch.no_grad():
            nnet_outputs = []
            input_num_frames = padded_feat.shape[1] + 2 \
                                - args.model_left_context - args.model_right_context
            for i in range(0, output_len_list[0], subsampled_frames_per_chunk):
                # 418 -> [0, 17, 34, 51, 68, 85, 102, 119, 136]
                first_output = i * subsampling_factor
                last_output = min(input_num_frames, \
                    first_output + (subsampled_frames_per_chunk-1) * subsampling_factor)
                first_input = first_output
                last_input = last_output + args.model_left_context + args.model_right_context
                input_x = padded_feat[:, first_input:last_input+1, :]
                ivector_index = (first_output + last_output) // 2 // args.ivector_period
                input_ivector = padded_ivector[:, ivector_index, :]
                feat = torch.cat((input_x, input_ivector.repeat((1, input_x.shape[1], 1))), dim=-1)
                nnet_output_temp, _ = model(feat)
                nnet_outputs.append(nnet_output_temp)
            nnet_output = torch.cat(nnet_outputs, dim=1)

        num = len(key_list)
        for i in range(num):
            key = key_list[i]
            output_len = output_len_list[i]
            value = nnet_output[i, :output_len, :]
            value = value.cpu()

            m = kaldi.SubMatrixFromDLPack(to_dlpack(value))
            m = Matrix(m)
            writer.Write(key, m)

        if batch_idx % 100 == 0:
            logging.info('Processed batch {}/{} ({:.6f}%)'.format(
                batch_idx, len(dataloader),
                float(batch_idx) / len(dataloader) * 100))

    writer.Close()
    logging.info('pseudo-log-likelihood is saved to {}'.format(
        os.path.join(args.dir, 'nnet_output.scp')))


if __name__ == '__main__':
    main()
