#!/usr/bin/env python3

# Copyright 2019-2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import logging
import os
import sys
import warnings

# disable warnings when loading tensorboard
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
from torch.utils.tensorboard import SummaryWriter

import kaldi
import kaldi_pybind.chain as chain
import kaldi_pybind.fst as fst

from chain_loss import KaldiChainObjfFunction
from common import load_checkpoint
from common import save_checkpoint
from common import save_training_info
from common import setup_logger
from egs_dataset import get_egs_dataloader
from model import get_chain_model
from options import get_args


def get_validation_objf(dataloader, model, device, criterion, opts, den_graph):
    total_objf = 0.
    total_weight = 0.
    total_frames = 0.  # for display only

    model.eval()

    for batch_idx, batch in enumerate(dataloader):
        key_list, feature_list, supervision_list = batch

        assert len(key_list) == len(feature_list) == len(supervision_list)
        batch_size = len(key_list)

        for n in range(batch_size):
            feats = feature_list[n]
            assert feats.ndim == 3

            # at this point, feats is [N, T, C]
            feats = feats.to(device)

            with torch.no_grad():
                nnet_output, xent_output = model(feats)

            # at this point, nnet_output is: [N, T, C]
            # refer to kaldi/src/chain/chain-training.h
            # the output should be organized as
            # [all sequences for frame 0]
            # [all sequences for frame 1]
            # [etc.]
            nnet_output = nnet_output.permute(1, 0, 2)
            # at this point, nnet_output is: [T, N, C]
            nnet_output = nnet_output.contiguous().view(-1,
                                                        nnet_output.shape[-1])

            # at this point, xent_output is: [N, T, C]
            xent_output = xent_output.permute(1, 0, 2)
            # at this point, xent_output is: [T, N, C]
            xent_output = xent_output.contiguous().view(-1,
                                                        xent_output.shape[-1])
            objf_l2_term_weight = criterion(opts, den_graph,
                                            supervision_list[n], nnet_output,
                                            xent_output)
            objf = objf_l2_term_weight[0]

            objf_l2_term_weight = objf_l2_term_weight.cpu()

            total_objf += objf_l2_term_weight[0].item()
            total_weight += objf_l2_term_weight[2].item()

            num_frames = nnet_output.shape[0]
            total_frames += num_frames

    return total_objf, total_weight, total_frames


def train_one_epoch(dataloader, valid_dataloader, model, device, optimizer,
                    criterion, current_epoch, opts, den_graph, tf_writer):
    model.train()

    total_objf = 0.
    total_weight = 0.
    total_frames = 0.  # for display only

    for batch_idx, batch in enumerate(dataloader):
        key_list, feature_list, supervision_list = batch
        assert len(key_list) == len(feature_list) == len(supervision_list)
        batch_size = len(key_list)
        for n in range(batch_size):
            feats = feature_list[n]
            assert feats.ndim == 3

            # at this point, feats is [N, T, C]
            feats = feats.to(device)
            nnet_output, xent_output = model(feats)

            # at this point, nnet_output is: [N, T, C]
            # refer to kaldi/src/chain/chain-training.h
            # the output should be organized as
            # [all sequences for frame 0]
            # [all sequences for frame 1]
            # [etc.]
            nnet_output = nnet_output.permute(1, 0, 2)
            # at this point, nnet_output is: [T, N, C]
            nnet_output = nnet_output.contiguous().view(-1,
                                                        nnet_output.shape[-1])

            # at this point, xent_output is: [N, T, C]
            xent_output = xent_output.permute(1, 0, 2)
            # at this point, xent_output is: [T, N, C]
            xent_output = xent_output.contiguous().view(-1,
                                                        xent_output.shape[-1])
            objf_l2_term_weight = criterion(opts, den_graph,
                                            supervision_list[n], nnet_output,
                                            xent_output)
            objf = objf_l2_term_weight[0]
            optimizer.zero_grad()
            objf.backward()

            clip_grad_value_(model.parameters(), 5.0)

            optimizer.step()

            objf_l2_term_weight = objf_l2_term_weight.detach().cpu()

            total_objf += objf_l2_term_weight[0].item()
            total_weight += objf_l2_term_weight[2].item()
            num_frames = nnet_output.shape[0]
            total_frames += num_frames

        if batch_idx % 100 == 0:
            logging.info(
                'Process {}/{}({:.6f}%) global average objf: {:.6f} over {} '
                'frames, current batch average objf: {:.6f} over {} frames, epoch {}'
                .format(
                    batch_idx, len(dataloader),
                    float(batch_idx) / len(dataloader) * 100,
                    total_objf / total_weight, total_frames,
                    objf_l2_term_weight[0].item() /
                    objf_l2_term_weight[2].item(), num_frames, current_epoch))

        if batch_idx % 500 == 0:
            total_valid_objf, total_valid_weight, total_valid_frames = get_validation_objf(
                dataloader=valid_dataloader,
                model=model,
                device=device,
                criterion=criterion,
                opts=opts,
                den_graph=den_graph)

            model.train()

            logging.info(
                'Validation average objf: {:.6f} over {} frames'.format(
                    total_valid_objf / total_valid_weight, total_valid_frames))

            tf_writer.add_scalar('train/global_valid_average_objf',
                                 total_valid_objf / total_valid_weight,
                                 batch_idx + current_epoch * len(dataloader))

        if batch_idx % 100 == 0:
            tf_writer.add_scalar('train/global_average_objf',
                                 total_objf / total_weight,
                                 batch_idx + current_epoch * len(dataloader))
            tf_writer.add_scalar(
                'train/current_batch_average_objf',
                objf_l2_term_weight[0].item() / objf_l2_term_weight[2].item(),
                batch_idx + current_epoch * len(dataloader))

            state_dict = model.state_dict()
            for key, value in state_dict.items():
                # skip batchnorm parameters
                if value.dtype != torch.float32:
                    continue
                if 'running_mean' in key or 'running_var' in key:
                    continue

                with torch.no_grad():
                    frobenius_norm = torch.norm(value, p='fro')

                tf_writer.add_scalar(
                    'train/parameters/{}'.format(key), frobenius_norm,
                    batch_idx + current_epoch * len(dataloader))

    return total_objf / total_weight


def main():
    args = get_args()
    setup_logger('{}/log-train'.format(args.dir), args.log_level)
    logging.info(' '.join(sys.argv))

    if torch.cuda.is_available() == False:
        logging.error('No GPU detected!')
        sys.exit(-1)

    # WARNING(fangjun): we have to select GPU at the very
    # beginning; otherwise you will get trouble later
    kaldi.SelectGpuDevice(device_id=args.device_id)
    kaldi.CuDeviceAllowMultithreading()
    device = torch.device('cuda', args.device_id)

    den_fst = fst.StdVectorFst.Read(args.den_fst_filename)

    opts = chain.ChainTrainingOptions()
    opts.l2_regularize = args.l2_regularize
    opts.xent_regularize = args.xent_regularize
    opts.leaky_hmm_coefficient = args.leaky_hmm_coefficient

    den_graph = chain.DenominatorGraph(fst=den_fst, num_pdfs=args.output_dim)

    model = get_chain_model(
        feat_dim=args.feat_dim,
        output_dim=args.output_dim,
        lda_mat_filename=args.lda_mat_filename,
        hidden_dim=args.hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
        prefinal_bottleneck_dim=args.prefinal_bottleneck_dim,
        kernel_size_list=args.kernel_size_list,
        subsampling_factor_list=args.subsampling_factor_list)

    start_epoch = 0
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    best_objf = -100000

    if args.checkpoint:
        start_epoch, learning_rate, best_objf = load_checkpoint(
            args.checkpoint, model)
        logging.info(
            'loaded from checkpoint: start epoch {start_epoch}, '
            'learning rate {learning_rate}, best objf {best_objf}'.format(
                start_epoch=start_epoch,
                learning_rate=learning_rate,
                best_objf=best_objf))

    model.to(device)

    dataloader = get_egs_dataloader(egs_dir_or_scp=args.cegs_dir,
                                    egs_left_context=args.egs_left_context,
                                    egs_right_context=args.egs_right_context,
                                    shuffle=True)

    valid_dataloader = get_egs_dataloader(
        egs_dir_or_scp=args.valid_cegs_scp,
        egs_left_context=args.egs_left_context,
        egs_right_context=args.egs_right_context,
        shuffle=False)

    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           weight_decay=5e-4)

    criterion = KaldiChainObjfFunction.apply

    tf_writer = SummaryWriter(log_dir='{}/tensorboard'.format(args.dir))

    best_epoch = start_epoch
    best_model_path = os.path.join(args.dir, 'best_model.pt')
    best_epoch_info_filename = os.path.join(args.dir, 'best-epoch-info')
    try:
        for epoch in range(start_epoch, args.num_epochs):
            learning_rate = 1e-3 * pow(0.4, epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            logging.info('epoch {}, learning rate {}'.format(
                epoch, learning_rate))

            tf_writer.add_scalar('learning_rate', learning_rate, epoch)

            objf = train_one_epoch(dataloader=dataloader,
                                   valid_dataloader=valid_dataloader,
                                   model=model,
                                   device=device,
                                   optimizer=optimizer,
                                   criterion=criterion,
                                   current_epoch=epoch,
                                   opts=opts,
                                   den_graph=den_graph,
                                   tf_writer=tf_writer)

            if best_objf is None:
                best_objf = objf
                best_epoch = epoch

            # the higher, the better
            if objf > best_objf:
                best_objf = objf
                best_epoch = epoch
                save_checkpoint(filename=best_model_path,
                                model=model,
                                epoch=epoch,
                                learning_rate=learning_rate,
                                objf=objf)
                save_training_info(filename=best_epoch_info_filename,
                                   model_path=best_model_path,
                                   current_epoch=epoch,
                                   learning_rate=learning_rate,
                                   objf=best_objf,
                                   best_objf=best_objf,
                                   best_epoch=best_epoch)

            # we always save the model for every epoch
            model_path = os.path.join(args.dir, 'epoch-{}.pt'.format(epoch))
            save_checkpoint(filename=model_path,
                            model=model,
                            epoch=epoch,
                            learning_rate=learning_rate,
                            objf=objf)

            epoch_info_filename = os.path.join(args.dir,
                                               'epoch-{}-info'.format(epoch))
            save_training_info(filename=epoch_info_filename,
                               model_path=model_path,
                               current_epoch=epoch,
                               learning_rate=learning_rate,
                               objf=objf,
                               best_objf=best_objf,
                               best_epoch=best_epoch)

    except KeyboardInterrupt:
        # save the model when ctrl-c is pressed
        model_path = os.path.join(args.dir,
                                  'epoch-{}-interrupted.pt'.format(epoch))
        # use a very small objf for interrupted model
        objf = -100000
        save_checkpoint(model_path,
                        model=model,
                        epoch=epoch,
                        learning_rate=learning_rate,
                        objf=objf)

        epoch_info_filename = os.path.join(
            args.dir, 'epoch-{}-interrupted-info'.format(epoch))
        save_training_info(filename=epoch_info_filename,
                           model_path=model_path,
                           current_epoch=epoch,
                           learning_rate=learning_rate,
                           objf=objf,
                           best_objf=best_objf,
                           best_epoch=best_epoch)

    tf_writer.close()
    logging.warning('Done')


if __name__ == '__main__':
    torch.manual_seed(20191227)
    main()
