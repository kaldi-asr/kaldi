#!/usr/bin/env python3

# Copyright 2019 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import os

import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import kaldi_pybind.nnet3 as nnet3
import kaldi

from common import splice_feats
from model import get_chain_model


def get_feat_dataloader(feats_scp,
                        model_left_context,
                        model_right_context,
                        batch_size=16,
                        num_workers=10):
    dataset = FeatDataset(feats_scp=feats_scp)

    collate_fn = FeatDatasetCollateFunc(model_left_context=model_left_context,
                                        model_right_context=model_right_context,
                                        frame_subsampling_factor=3)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=False,
                            collate_fn=collate_fn)
    return dataloader


def _add_model_left_right_context(x, left_context, right_context):
    padded = x
    if left_context > 0:
        first_frame = x[0, :]
        left_padding = [first_frame] * left_context
        padded = np.vstack([left_padding, x])

    if right_context > 0:
        last_frame = x[-1, :]
        right_padding = [last_frame] * right_context
        padded = np.vstack([padded, right_padding])

    return padded


class FeatDataset(Dataset):

    def __init__(self, feats_scp):
        assert os.path.isfile(feats_scp)

        self.feats_scp = feats_scp

        # items is a list of [key, rxfilename]
        items = list()

        with open(feats_scp, 'r') as f:
            for line in f:
                split = line.split()
                assert len(split) == 2
                items.append(split)

        self.items = items

        self.num_items = len(self.items)

    def __len__(self):
        return self.num_items

    def __getitem__(self, i):
        return self.items[i]

    def __str__(self):
        s = 'feats scp: {}\n'.format(self.feats_scp)
        s += 'num utt: {}\n'.format(self.num_items)
        return s


class FeatDatasetCollateFunc:

    def __init__(self,
                 model_left_context,
                 model_right_context,
                 frame_subsampling_factor=3):
        '''
        We need `frame_subsampling_factor` because we want to know
        the number of output frames of different waves in the same batch
        '''
        self.model_left_context = model_left_context
        self.model_right_context = model_right_context
        self.frame_subsampling_factor = frame_subsampling_factor

    def __call__(self, batch):
        '''
        batch is a list of [key, rxfilename]
        '''
        key_list = []
        feat_list = []
        output_len_list = []
        for b in batch:
            key, rxfilename = b
            key_list.append(key)
            feat = kaldi.read_mat(rxfilename).numpy()
            feat_len = feat.shape[0]
            output_len = (feat_len + self.frame_subsampling_factor -
                          1) // self.frame_subsampling_factor
            output_len_list.append(output_len)
            # now add model left and right context
            feat = _add_model_left_right_context(feat, self.model_left_context,
                                                 self.model_right_context)
            feat = splice_feats(feat)
            feat_list.append(feat)
            # no need to sort the feat by length

        # the user should sort utterances by length offline
        # to avoid unnecessary padding
        padded_feat = pad_sequence(
            [torch.from_numpy(feat).float() for feat in feat_list],
            batch_first=True)
        return key_list, padded_feat, output_len_list
