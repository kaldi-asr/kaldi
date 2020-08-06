#!/usr/bin/env python3

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import logging
import math
import os
import sys
import warnings

# disable warnings when loading tensorboard
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_value_
from torch.utils.tensorboard import SummaryWriter

import kaldi

from common import load_checkpoint
from common import save_checkpoint
from common import save_training_info
from common import setup_logger
from ctc_loss import CTCLoss
from dataset import get_ctc_dataloader
from model import get_ctc_model
from options import get_args


def train_one_epoch(dataloader, model, device, optimizer, loss_func,
                    current_epoch, tf_writer):
    total_loss = 0.
    num = 0.

    # TODO(fangjun): remove `num_repeat`. It's used only for testing.
    num_repeat = 100
    for kk in range(num_repeat):
        for batch_idx, batch in enumerate(dataloader):
            unused_uttid_list, feat, feat_len_list, label_list, label_len_list = batch

            feat = feat.to(device)

            activations, feat_len_list = model(feat, feat_len_list)

            # at this point activations is of shape: [batch_size, seq_len, output_dim]
            # CTCLoss requires a layout: [seq_len, batch_size, output_dim]

            activations = activations.permute(1, 0, 2)
            # now activations is of shape [seq_len, batch_size, output_dim]

            targets = torch.tensor(label_list)

            if not isinstance(feat_len_list, torch.Tensor):
                input_lengths = torch.tensor(feat_len_list)
            else:
                input_lengths = feat_len_list

            target_lengths = torch.tensor(label_len_list)

            loss = loss_func(activations=activations,
                             targets=targets,
                             input_lengths=input_lengths,
                             target_lengths=target_lengths)

            optimizer.zero_grad()
            if math.isnan(loss.item()):
                print(loss)
                logging.warn('loss is nan for batch {} at epoch {}\n'
                             'feat_len_list: {}\n'
                             'label_len_list: {}\n'.format(
                                 batch_idx, current_epoch, feat_len_list,
                                 label_len_list))
                import sys
                sys.exit(1)

            loss.backward()

            #  clip_grad_value_(model.parameters(), 5.0)

            optimizer.step()

            total_loss += loss.item()
            num += 1
            if batch_idx % 100 == 0:
                logging.info(
                    'Device ({}) batch {}/{} ({:.2f}%) ({}/{}), loss {:.5f}, average {:.5f}'
                    .format(device.index, batch_idx, len(dataloader),
                            float(batch_idx) / len(dataloader) * 100, kk,
                            num_repeat, loss.item(), total_loss / num))

            if tf_writer and batch_idx % 100 == 0:
                tf_writer.add_scalar(
                    'train/current_batch_average_loss', loss.item(),
                    batch_idx + kk * len(dataloader) +
                    num_repeat * len(dataloader) * current_epoch)

                tf_writer.add_scalar(
                    'train/global_average_loss', total_loss / num,
                    batch_idx + kk * len(dataloader) +
                    num_repeat * len(dataloader) * current_epoch)

    return total_loss / num


def main():
    args = get_args()
    setup_logger('{}/log-train-device-{}'.format(args.dir, args.device_id),
                 args.log_level)
    logging.info(' '.join(sys.argv))

    if torch.cuda.is_available() == False:
        logging.error('No GPU detected!')
        sys.exit(-1)

    dist.init_process_group('nccl',
                            rank=args.device_id,
                            world_size=args.world_size)

    kaldi.SelectGpuDevice(device_id=args.device_id)

    device = torch.device('cuda', args.device_id)

    model = get_ctc_model(input_dim=args.input_dim,
                          output_dim=args.output_dim,
                          num_layers=args.num_layers,
                          hidden_dim=args.hidden_dim,
                          proj_dim=args.proj_dim)

    start_epoch = 0
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    best_loss = None

    if args.checkpoint:
        start_epoch, learning_rate, best_loss = load_checkpoint(
            args.checkpoint, model)
        logging.info(
            'loaded from checkpoint: start epoch {start_epoch}, '
            'learning rate {learning_rate}, best loss {best_loss}'.format(
                start_epoch=start_epoch,
                learning_rate=learning_rate,
                best_loss=best_loss))

    model.to(device)

    model = DDP(model, device_ids=[args.device_id])

    dataloader = get_ctc_dataloader(
        feats_scp=args.feats_scp,
        labels_scp=args.labels_scp,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        model_left_context=args.model_left_context,
        model_right_context=args.model_right_context,
        world_size=args.world_size,
        local_rank=args.device_id)

    lr = learning_rate
    optimizer = optim.Adam(model.parameters(),
                           lr=lr,
                           weight_decay=args.l2_regularize)

    if device.index == 0:
        tf_writer = SummaryWriter(log_dir='{}/tensorboard'.format(args.dir))
    else:
        tf_writer = None

    model.train()

    loss_func = CTCLoss(use_warp_ctc=False, blank=0, reduction='mean')

    best_epoch = 0
    best_model_path = os.path.join(args.dir, 'best_model.pt')
    best_epoch_info_filename = os.path.join(args.dir, 'best-epoch-info')

    dist.barrier()

    try:
        for epoch in range(start_epoch, num_epochs):
            learning_rate = lr * pow(0.8, epoch)
            #  learning_rate = lr

            if tf_writer:
                tf_writer.add_scalar('learning_rate', learning_rate, epoch)

            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            logging.info('Device ({}) epoch {}, learning rate {}'.format(
                device.index, epoch, learning_rate))

            loss = train_one_epoch(dataloader=dataloader,
                                   model=model,
                                   device=device,
                                   optimizer=optimizer,
                                   loss_func=loss_func,
                                   current_epoch=epoch,
                                   tf_writer=tf_writer)

            # the lower, the better
            if best_loss is None or best_loss > loss:
                best_loss = loss
                best_epoch = epoch
                save_checkpoint(filename=best_model_path,
                                model=model,
                                epoch=epoch,
                                learning_rate=learning_rate,
                                loss=loss,
                                local_rank=args.device_id)
                save_training_info(filename=best_epoch_info_filename,
                                   model_path=best_model_path,
                                   current_epoch=epoch,
                                   learning_rate=learning_rate,
                                   loss=loss,
                                   best_loss=best_loss,
                                   best_epoch=best_epoch,
                                   local_rank=args.device_id)

            # we always save the model for every epoch
            model_path = os.path.join(args.dir, 'epoch-{}.pt'.format(epoch))
            save_checkpoint(filename=model_path,
                            model=model,
                            epoch=epoch,
                            learning_rate=learning_rate,
                            loss=loss,
                            local_rank=args.device_id)

            epoch_info_filename = os.path.join(args.dir,
                                               'epoch-{}-info'.format(epoch))
            save_training_info(filename=epoch_info_filename,
                               model_path=model_path,
                               current_epoch=epoch,
                               learning_rate=learning_rate,
                               loss=loss,
                               best_loss=best_loss,
                               best_epoch=best_epoch,
                               local_rank=args.device_id)
    except KeyboardInterrupt:
        # save the model when ctrl-c is pressed
        model_path = os.path.join(args.dir,
                                  'epoch-{}-interrupted.pt'.format(epoch))
        # use a very large loss for the interrupted model
        loss = 100000000
        save_checkpoint(model_path,
                        model=model,
                        epoch=epoch,
                        learning_rate=learning_rate,
                        loss=loss,
                        local_rank=args.device_id)

        epoch_info_filename = os.path.join(
            args.dir, 'epoch-{}-interrupted-info'.format(epoch))
        save_training_info(filename=epoch_info_filename,
                           model_path=model_path,
                           current_epoch=epoch,
                           learning_rate=learning_rate,
                           loss=loss,
                           best_loss=best_loss,
                           best_epoch=best_epoch,
                           local_rank=args.device_id)

    if tf_writer:
        tf_writer.close()
    logging.warning('Device ({}) Training done!'.format(args.device_id))


if __name__ == '__main__':
    torch.manual_seed(20200221)
    main()
