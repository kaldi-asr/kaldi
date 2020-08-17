#!/usr/bin/env python3

# Copyright 2019-2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import logging
import os
import sys
import warnings
from multiprocessing import Process
# disable warnings when loading tensorboard
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_value_
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter 

import kaldi
import kaldi_pybind.chain as chain
import kaldi_pybind.fst as fst

from chain_loss import KaldiChainObjfFunction
from common import load_checkpoint
from common import save_checkpoint
from common import save_training_info
from common import setup_logger
from device_utils import allocate_gpu_devices
from egs_dataloader import get_egs_dataloader
from libs.nnet3.train.dropout_schedule import _get_dropout_proportions
from model import get_chain_model
from options import get_args
#from sgd_max_change import SgdMaxChange

def get_objf(batch, model, device, criterion, opts, den_graph, training, optimizer=None, dropout=0.):
    feature, supervision = batch
    assert feature.ndim == 3

    # at this point, feature is [N, T, C]
    feature = feature.to(device)
    if training:
        nnet_output, xent_output = model(feature, dropout=dropout)
    else:
        with torch.no_grad():
            nnet_output, xent_output = model(feature)

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
                                supervision, nnet_output,
                                xent_output)
    objf = objf_l2_term_weight[0]
    if training:
        optimizer.zero_grad()
        objf.backward()
        clip_grad_value_(model.parameters(), 5.0)
        optimizer.step()

    objf_l2_term_weight = objf_l2_term_weight.detach().cpu()

    total_objf = objf_l2_term_weight[0].item()
    total_weight = objf_l2_term_weight[2].item()
    total_frames = nnet_output.shape[0]

    return total_objf, total_weight, total_frames


def get_validation_objf(dataloader, model, device, criterion, opts, den_graph):
    total_objf = 0.
    total_weight = 0.
    total_frames = 0.  # for display only
 
    model.eval()

    for batch_idx, (pseudo_epoch, batch) in enumerate(dataloader):
        objf, weight, frames = get_objf(
            batch, model, device, criterion, opts, den_graph, False) 
        total_objf += objf
        total_weight += weight
        total_frames += frames

    return total_objf, total_weight, total_frames


def train_one_epoch(dataloader, valid_dataloader, model, device, optimizer, criterion, 
                    current_epoch, num_epochs, opts, den_graph, tf_writer, rank, dropout_schedule):
    total_objf = 0.
    total_weight = 0.
    total_frames = 0.  # for display only

    model.train()
    # iterates over one training scp file in one `pseudo_epoch`, 
    # so one `pseudo_epoch` may contain many `batch`.
    for batch_idx, (pseudo_epoch, batch) in enumerate(dataloader):
        # `len(dataloader)` returns the number of `pseudo_epoch`
        # in the current worker, that is the number of scp files
        # we will process in this worker.
        data_fraction = (pseudo_epoch + 1 + current_epoch *
                         len(dataloader)) / (len(dataloader) * num_epochs)
        _, dropout = _get_dropout_proportions(
            dropout_schedule, data_fraction)[0]
        curr_batch_objf, curr_batch_weight, curr_batch_frames = get_objf(
            batch, model, device, criterion, opts, den_graph, True, optimizer, dropout=dropout)

        total_objf += curr_batch_objf
        total_weight += curr_batch_weight
        total_frames += curr_batch_frames

        if batch_idx % 100 == 0:
            logging.info(
                'Device ({}) processing batch {}, current pseudo-epoch is {}/{}({:.6f}%), '
                'global average objf: {:.6f} over {} '
                'frames, current batch average objf: {:.6f} over {} frames, epoch {}'
                .format(
                    device.index, batch_idx, pseudo_epoch, len(dataloader),
                    float(pseudo_epoch) / len(dataloader) * 100,
                    total_objf / total_weight, total_frames,
                    curr_batch_objf / curr_batch_weight, 
                    curr_batch_frames, current_epoch))

        if valid_dataloader and batch_idx % 1000 == 0:
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
            if tf_writer:
                tf_writer.add_scalar('train/global_valid_average_objf',
                                 total_valid_objf / total_valid_weight,
                                 pseudo_epoch + current_epoch * len(dataloader))
        # rank == None means we are not using ddp
        if (rank is None or rank == 0) and batch_idx % 100 == 0 and tf_writer is not None:
            tf_writer.add_scalar('train/global_average_objf',
                                 total_objf / total_weight,
                                 pseudo_epoch + current_epoch * len(dataloader))
            tf_writer.add_scalar(
                'train/current_batch_average_objf',
                curr_batch_objf / curr_batch_weight,
                pseudo_epoch + current_epoch * len(dataloader))
            
            tf_writer.add_scalar(
                'train/current_dropout',
                dropout,
                pseudo_epoch + current_epoch * len(dataloader))

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
                    pseudo_epoch + current_epoch * len(dataloader))
    return total_objf / total_weight

def main():
    args = get_args()

    if args.use_ddp:
        learning_rate = args.learning_rate * args.world_size
        if args.multiple_machine:
            # Suppose we have submitted multiple jobs with SGE (Sun Grid Engine)
            local_rank = int(os.environ['SGE_TASK_ID']) - 1
            process_job(learning_rate, local_rank=local_rank)
        else:
            proc = []
            if args.device_ids is not None:
                assert len(args.device_ids) >= args.world_size
            for i in range(args.world_size):
                device_id = None if args.device_ids is None else args.device_ids[i]
                p = Process(target=process_job, args=(learning_rate, device_id, i))
                proc.append(p)
                p.start()
            for p in proc:
                p.join()
    else:
        device_id = None if args.device_ids is None else args.device_ids[0]
        process_job(args.learning_rate, device_id)


def process_job(learning_rate, device_id=None, local_rank=None):
    args = get_args()
    if local_rank is not None:    
        setup_logger('{}/logs/log-train-rank-{}'.format(args.dir, local_rank),
                 args.log_level)
    else:
        setup_logger('{}/logs/log-train-single-GPU'.format(args.dir), args.log_level)

    logging.info(' '.join(sys.argv))

    if not torch.cuda.is_available():
        logging.error('No GPU detected!')
        sys.exit(-1)

    if device_id is None:
        devices = allocate_gpu_devices(1)
        if len(devices) < 1:
            logging.error('Allocate GPU failed!')
            sys.exit(-1)
        device_id = devices[0][0]
    
    logging.info('device: {}'.format(device_id))

    if args.use_ddp:
        os.environ["NCCL_IB_DISABLE"]="1"  
        dist.init_process_group('nccl',
                            init_method=args.init_method,
                            rank=local_rank,
                            world_size=args.world_size)

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(20191227)
    # WARNING(fangjun): we have to select GPU at the very
    # beginning; otherwise you will get trouble later
    kaldi.SelectGpuDevice(device_id=device_id)
    kaldi.CuDeviceAllowMultithreading()
    device = torch.device('cuda', device_id)

    den_fst = fst.StdVectorFst.Read(args.den_fst_filename)

    opts = chain.ChainTrainingOptions()
    opts.l2_regularize = args.l2_regularize
    opts.xent_regularize = args.xent_regularize
    opts.leaky_hmm_coefficient = args.leaky_hmm_coefficient

    den_graph = chain.DenominatorGraph(fst=den_fst, num_pdfs=args.output_dim)
    
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

    start_epoch = 0
    num_epochs = args.num_epochs
    best_objf = -100000

    if args.checkpoint:
        start_epoch, curr_learning_rate, best_objf = load_checkpoint(
            args.checkpoint, model)
        logging.info(
            'Device ({device_id}) loaded from checkpoint: start epoch {start_epoch}, '
            'learning rate {learning_rate}, best objf {best_objf}'.format(
                device_id=device_id,
                start_epoch=start_epoch,
                learning_rate=curr_learning_rate,
                best_objf=best_objf))

    model.to(device)
    
    if args.use_ddp:
        model = DDP(model, device_ids=[device_id])

    dataloader = get_egs_dataloader(egs_dir_or_scp=args.cegs_dir,
                                    egs_left_context=args.egs_left_context,
                                    egs_right_context=args.egs_right_context,
                                    world_size=args.world_size,
                                    local_rank=local_rank)

    if not args.use_ddp or local_rank == 0:
        valid_dataloader = get_egs_dataloader(
            egs_dir_or_scp=args.valid_cegs_scp,
            egs_left_context=args.egs_left_context,
            egs_right_context=args.egs_right_context)
    else:
        valid_dataloader = None

    #optimizer = SgdMaxChange(model.parameters(),
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           weight_decay=5e-4)

    criterion = KaldiChainObjfFunction.apply

    if not args.use_ddp or local_rank == 0:
        tf_writer = SummaryWriter(log_dir='{}/tensorboard'.format(args.dir))
    else:
        tf_writer = None

    best_epoch = start_epoch
    best_model_path = os.path.join(args.dir, 'best_model.pt')
    best_epoch_info_filename = os.path.join(args.dir, 'best-epoch-info')
    
    if args.use_ddp:
        dist.barrier()

    try:
        for epoch in range(start_epoch, args.num_epochs):
            curr_learning_rate =  learning_rate * pow(0.4, epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = curr_learning_rate

            logging.info('epoch {}, learning rate {}'.format(
                epoch, curr_learning_rate))

            if tf_writer:
                tf_writer.add_scalar('learning_rate', curr_learning_rate, epoch)
            
            if dataloader.sampler and isinstance(dataloader.sampler, DistributedSampler):
                dataloader.sampler.set_epoch(epoch)

            objf = train_one_epoch(dataloader=dataloader,
                                   valid_dataloader=valid_dataloader,
                                   model=model,
                                   device=device,
                                   optimizer=optimizer,
                                   criterion=criterion,
                                   current_epoch=epoch,
                                   num_epochs=num_epochs,
                                   opts=opts,
                                   den_graph=den_graph,
                                   tf_writer=tf_writer,
                                   rank=local_rank,
                                   dropout_schedule=args.dropout_schedule)

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
                                learning_rate=curr_learning_rate,
                                objf=objf,
                                local_rank=local_rank)
                save_training_info(filename=best_epoch_info_filename,
                                   model_path=best_model_path,
                                   current_epoch=epoch,
                                   learning_rate=curr_learning_rate,
                                   objf=best_objf,
                                   best_objf=best_objf,
                                   best_epoch=best_epoch,
                                   local_rank=local_rank)

            # we always save the model for every epoch
            model_path = os.path.join(args.dir, 'epoch-{}.pt'.format(epoch))
            save_checkpoint(filename=model_path,
                            model=model,
                            epoch=epoch,
                            learning_rate=curr_learning_rate,
                            objf=objf,
                            local_rank=local_rank)
            epoch_info_filename = os.path.join(args.dir,
                                               'epoch-{}-info'.format(epoch))
            save_training_info(filename=epoch_info_filename,
                               model_path=model_path,
                               current_epoch=epoch,
                               learning_rate=curr_learning_rate,
                               objf=objf,
                               best_objf=best_objf,
                               best_epoch=best_epoch,
                               local_rank=local_rank)
    except KeyboardInterrupt:
        # save the model when ctrl-c is pressed
        model_path = os.path.join(args.dir,
                                  'epoch-{}-interrupted.pt'.format(epoch))
        # use a very small objf for interrupted model
        objf = -100000
        save_checkpoint(model_path,
                        model=model,
                        epoch=epoch,
                        learning_rate=curr_learning_rate,
                        objf=objf,
                        local_rank=local_rank)

        epoch_info_filename = os.path.join(
            args.dir, 'epoch-{}-interrupted-info'.format(epoch))
        save_training_info(filename=epoch_info_filename,
                           model_path=model_path,
                           current_epoch=epoch,
                           learning_rate=curr_learning_rate,
                           objf=objf,
                           best_objf=best_objf,
                           best_epoch=best_epoch,
                           local_rank=local_rank)

    if tf_writer:
        tf_writer.close()
    logging.warning('Done')


if __name__ == '__main__':
    main()
