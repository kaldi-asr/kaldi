#!/usr/bin/env python3

# Copyright 2019 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_value_

import kaldi
import kaldi_pybind.fst as fst
import kaldi_pybind.chain as chain

from options import get_args
from model import get_chain_model
from dataset import get_dataloader
from chain_loss import KaldiChainObjfFunction


def train_one_epoch(dataloader, model, device, optimizer, criterion,
                    current_epoch, opts, den_graph):
    model.train()

    total_objf = 0.
    total_weight = 0.
    total_frames = 0.

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
        if batch_idx % 10 == 0:
            print('Process {}/{}({:.6f}%) global average objf: {:.6f} over {} \
                    frames, current batch average objf: {:.6f} over {} frames, epoch {}'
                  .format(
                      batch_idx, len(dataloader),
                      float(batch_idx) / len(dataloader),
                      total_objf / total_weight, total_frames,
                      float(objf_l2_term_weight[0].item()) /
                      objf_l2_term_weight[2].item(), num_frames, current_epoch))


def main():
    args = get_args()

    # we always use GPU
    kaldi.SelectGpuDevice(device_id=args.device_id)
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
                            kernel_size_list=args.kernel_size_list,
                            dilation_list=args.dilation_list)

    model.to(device)

    dataloader = get_dataloader(egs_dir=args.cegs_dir,
                                egs_left_context=args.egs_left_context,
                                egs_right_context=args.egs_right_context)
    params = [{
        'params': [
            param for name, param in model.named_parameters()
            if 'xent' not in name
        ]
    }, {
        'params':
        [param for name, param in model.named_parameters() if 'xent' in name],
        'lr':
        args.learning_rate * 5
    }]

    optimizer = optim.Adam(params,
                           lr=args.learning_rate,
                           weight_decay=args.l2_regularize)

    scheduler = StepLR(optimizer, gamma=0.5, step_size=2)
    criterion = KaldiChainObjfFunction.apply

    for epoch in range(args.num_epochs):
        learning_rate = scheduler.get_lr()[0]
        print('epoch {}, learning rate {}'.format(epoch, learning_rate))
        train_one_epoch(dataloader=dataloader,
                        model=model,
                        device=device,
                        optimizer=optimizer,
                        criterion=criterion,
                        current_epoch=epoch,
                        opts=opts,
                        den_graph=den_graph)
    print('Done')


if __name__ == '__main__':
    main()

# TODO(fangjun)
# 1. support load/save checkpoint
# 2. support inference only mode
# 3. support tensorboard
