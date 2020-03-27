#!/usr/bin/env python3

# Copyright 2020 Xiaomi Corporation, Beijing, China (author: Haowen Qiu)
# Apache 2.0

from multiprocessing import Process
import datetime
import glob
import os

import numpy as np
import torch
import torch.distributed as dist

from torch.utils.data import Dataset

from kaldi import SequentialNnetChainExampleReader
import kaldi
import kaldi_pybind.nnet3 as nnet3

from common import splice_feats

def get_egs_dataloader(egs_dir_or_scp,
                       egs_left_context,
                       egs_right_context,
                       frame_subsampling_factor=3,
                       world_size=None,
                       local_rank=None):
    '''
    world_size and local_rank is for DistributedDataParallel training.
    '''
    dataset = NnetChainExampleScpDataset(egs_dir_or_scp)

    collate_fn = NnetChainExampleCollateFunc(
        egs_left_context=egs_left_context,
        egs_right_context=egs_right_context,
        frame_subsampling_factor=frame_subsampling_factor)

    if local_rank is not None:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    else:
        #sampler = torch.utils.data.SequentialSampler(dataset)
        sampler = torch.utils.data.RandomSampler(dataset)
    
    dataloader = NnetChainExampleDataLoader(dataset,
                            sampler=sampler,
                            collate_fn=collate_fn)
    return dataloader


class NnetChainExampleScpDataset(Dataset):

    def __init__(self, egs_dir_or_scp):
        '''
        If egs_dir_or_scp is a directory, we assume that there exist many cegs.*.scp files
        inside it.
        '''
        if os.path.isdir(egs_dir_or_scp):
            self.scps = glob.glob('{}/cegs.*.scp'.format(egs_dir_or_scp))
        else:
            self.scps = [egs_dir_or_scp]

        assert len(self.scps) > 0

    def __len__(self):
        return len(self.scps)

    def __getitem__(self, i):
        return self.scps[i]

    def __str__(self):
        s = 'num egs scp files: {}\n'.format(len(self.scps))
        return s


class NnetChainExampleDataLoader(object):
    '''
    Nnet chain example data loader, provides an iterable over the given scp files.

    Arguments:
        dataset (Dataset): scp files from which to load the egs.
        sampler (Sampler): defines the strategy to draw samples
            from the dataset.
        collate_fn (callable): creates a batch from mergerd eg.

    '''

    def __init__(self, dataset, sampler, collate_fn):

        self.dataset = dataset
        self.sampler = sampler
        self.collate_fn = collate_fn

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        # iterates over one scp file in a `pseudo_epoch`
        for pseudo_epoch, sample_idx in enumerate(self.sampler):
            # one sample is one scp file
            egs_rspecifier = 'scp:' + self.dataset[sample_idx]
            with SequentialNnetChainExampleReader(egs_rspecifier) as example_reader:
                for key, eg in example_reader:
                    batch = self.collate_fn(eg)
                    yield pseudo_epoch, batch


class NnetChainExampleCollateFunc:

    def __init__(self, egs_left_context, egs_right_context,
                 frame_subsampling_factor=3):

        '''
        egs_left_context is from egs/info/left_context
        egs_right_context is from egs/info/right_context
        '''
        assert egs_left_context >= 0
        assert egs_left_context >= 0

        # currently support either no subsampling or
        # subsampling factor to be 3
        assert frame_subsampling_factor in [1, 3]

        self.egs_left_context = egs_left_context
        self.egs_right_context = egs_right_context
        self.frame_subsampling_factor = frame_subsampling_factor
        
    def __call__(self, eg):
        '''
        eg is a batch as it has been merged
        '''
        assert eg.inputs[0].name == 'input'
        assert len(eg.outputs) == 1
        assert eg.outputs[0].name == 'output'


        supervision = eg.outputs[0].supervision

        batch_size = supervision.num_sequences
        frames_per_sequence = (supervision.frames_per_sequence *
                               self.frame_subsampling_factor) + \
            self.egs_left_context + self.egs_right_context


        _feats = kaldi.FloatMatrix()
        eg.inputs[0].features.GetMatrix(_feats)
        feats = _feats.numpy()

        if len(eg.inputs) > 1:
            _ivectors = kaldi.FloatMatrix()
            eg.inputs[1].features.GetMatrix(_ivectors)
            ivectors = _ivectors.numpy()

        assert feats.shape[0] == batch_size * frames_per_sequence

        feat_list = []
        for i in range(batch_size):
            start_index = i * frames_per_sequence
            if self.frame_subsampling_factor == 3:
                shift = np.random.choice([-1, 0, 1], 1)[0]
                start_index += shift

            end_index = start_index + frames_per_sequence
            start_index += 2  # remove the leftmost frame added for frame shift
            end_index -= 2  # remove the rightmost frame added for frame shift
            feat = feats[start_index:end_index:, :]
            if len(eg.inputs) > 1:
                repeat_ivector = torch.from_numpy(
                    ivectors[i]).repeat(feat.shape[0], 1)
                feat = torch.cat(
                    (torch.from_numpy(feat), repeat_ivector), dim=1).numpy()
            feat_list.append(feat)

        batched_feat = np.stack(feat_list, axis=0)
        assert batched_feat.shape[0] == batch_size

        assert batched_feat.shape[1] == frames_per_sequence - 4
        if len(eg.inputs) > 1:
            assert batched_feat.shape[2] == feats.shape[-1] + ivectors.shape[-1]
        else:
            assert batched_feat.shape[2] == feats.shape[-1]

        torch_feat = torch.from_numpy(batched_feat).float()

        return torch_feat, supervision


def _test_nnet_chain_example_dataloader():
    scp_dir = 'exp/chain_pybind/tdnn_sp/egs_chain2'
    _test_dataloader_iter(scp_dir)

def _test_dataloader_iter(scp_dir_or_file):
    egs_left_context = 29
    egs_right_context = 29
    frame_subsampling_factor = 3

    dataloader = get_egs_dataloader(
        scp_dir_or_file,
        egs_left_context,
        egs_right_context,
        frame_subsampling_factor)
    
    for i in range(2):
        batch_idx = 0
        for pseudo_epoch, batch in dataloader:
            print('{}: epoch {}, pseudo_epoch {}, batch_idx {}'.format(
                datetime.datetime.now(), i, pseudo_epoch, batch_idx))
            batch_idx = batch_idx + 1
            feature, supervision = batch
            assert feature.shape == (128, 204, 120) \
                or feature.shape == (128, 144, 120) \
                or feature.shape == (128, 165, 120)
            assert supervision.weight == 1
            assert supervision.num_sequences == 128  # minibach size is 128


if __name__ == '__main__':
    _test_nnet_chain_example_dataloader()
