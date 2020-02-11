#!/usr/bin/env python3

# Copyright 2019-2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Copyright 2019-2020 JD AI, Beijing, China (author: Lu Fan)
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
from torch.optim.lr_scheduler import MultiStepLR
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


def train_one_epoch(dataloader, model, device, optimizer, criterion,
                    current_epoch, opts, den_graph, tf_writer):
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

            # TODO(fangjun): how to choose this value or do we need this ?
            clip_grad_value_(model.parameters(), 5.0)
            optimizer.step()

            objf_l2_term_weight = objf_l2_term_weight.detach().cpu()

            total_objf += objf_l2_term_weight[0].item()
            total_weight += objf_l2_term_weight[2].item()
            num_frames = nnet_output.shape[0]
            total_frames += num_frames

        if np.random.choice(4) == 0:
            with torch.no_grad():
                model.constrain_orthonormal()

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
            log_norm = ["{}: {:.4f}".format(name, torch.norm(parms)) \ 
                        for name, parms in model.named_parameters() \
                        if "affine" in name or "linear" in name]
            logging.info("Process {}/{}({:.6f}%) l2-norm is:[ {} ]".format(batch_idx, 
                        len(dataloader), float(batch_idx) / len(dataloader) * 100), 
                        " ".join(log_norm))

        if batch_idx % 100 == 0:
            current_iter = batch_idx + current_epoch * len(dataloader)
            tf_writer.add_scalar('train/global_average_objf',
                                 total_objf / total_weight,
                                 current_iter)
            tf_writer.add_scalar(
                'train/current_batch_average_objf',
                objf_l2_term_weight[0].item() / objf_l2_term_weight[2].item(),
                current_iter)
            for name, parms in model.named_parameters():
                tf_writer.add_histogram(f'train/norm/{name}', 
                    parms.clone().cpu().data.numpy(),
                    current_iter)
                tf_writer.add_scalar(f'train/l2_norm/{name}', torch.norm(parms),
                    current_iter)

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

    # TODO(fangjun): pass these options from commandline
    opts = chain.ChainTrainingOptions()
    opts.l2_regularize = 5e-4
    opts.leaky_hmm_coefficient = 0.1

    den_graph = chain.DenominatorGraph(fst=den_fst, num_pdfs=args.output_dim)

    model = get_chain_model(feat_dim=args.feat_dim,
                            output_dim=args.output_dim,
                            lda_mat_filename=args.lda_mat_filename,
                            hidden_dim=args.hidden_dim,
                            bottleneck_dim=args.bottleneck_dim,
                            time_stride_list=args.time_stride_list)

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

    dataloader = get_egs_dataloader(egs_dir=args.cegs_dir,
                                    egs_left_context=args.egs_left_context,
                                    egs_right_context=args.egs_right_context)

    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           weight_decay=args.l2_regularize)

    scheduler = MultiStepLR(optimizer, milestones=[1, 2, 3, 4, 5], gamma=0.5)
    criterion = KaldiChainObjfFunction.apply

    tf_writer = SummaryWriter(log_dir='{}/tensorboard'.format(args.dir))

    best_epoch = start_epoch
    best_model_path = os.path.join(args.dir, 'best_model.pt')
    best_epoch_info_filename = os.path.join(args.dir, 'best-epoch-info')
    try:
        for epoch in range(start_epoch, args.num_epochs):
            learning_rate = scheduler.get_lr()[0]
            logging.info('epoch {}, learning rate {}'.format(
                epoch, learning_rate))
            tf_writer.add_scalar('learning_rate', learning_rate, epoch)

            objf = train_one_epoch(dataloader=dataloader,
                                   model=model,
                                   device=device,
                                   optimizer=optimizer,
                                   criterion=criterion,
                                   current_epoch=epoch,
                                   opts=opts,
                                   den_graph=den_graph,
                                   tf_writer=tf_writer)
            scheduler.step()

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
