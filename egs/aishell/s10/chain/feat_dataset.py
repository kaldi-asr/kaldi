#!/usr/bin/env python3

# Copyright 2019 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import os
import math
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
                        frames_per_chunk=51,
                        ivector_scp=None,
                        ivector_period=10,
                        batch_size=16,
                        num_workers=10):
    dataset = FeatDataset(feats_scp=feats_scp, ivector_scp=ivector_scp)

    collate_fn = FeatDatasetCollateFunc(model_left_context=model_left_context,
                                        model_right_context=model_right_context,
                                        frame_subsampling_factor=3,
                                        frames_per_chunk=frames_per_chunk,
                                        ivector_period=ivector_period)

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

    def __init__(self, feats_scp, ivector_scp=None):
        assert os.path.isfile(feats_scp)
        if ivector_scp:
            assert os.path.isfile(ivector_scp)

        self.feats_scp = feats_scp

        # items is a dict of [uttid, feat_rxfilename, None]
        # or [uttid, feat_rxfilename, ivector_rxfilename] if ivector_scp is not None
        items = dict()

        with open(feats_scp, 'r') as f:
            for line in f:
                split = line.split()
                assert len(split) == 2
                uttid, rxfilename = split
                assert uttid not in items
                items[uttid] = [uttid, rxfilename, None]
        self.ivector_scp = None
        if ivector_scp:
            self.ivector_scp = ivector_scp
            expected_count = len(items)
            n = 0
            with open(ivector_scp, 'r') as f:
                for line in f:
                    uttid_rxfilename = line.split()
                    assert len(uttid_rxfilename) == 2
                    uttid, rxfilename = uttid_rxfilename
                    assert uttid in items
                    items[uttid][-1] = rxfilename
                    n += 1
            assert n == expected_count

        self.items = list(items.values())

        self.num_items = len(self.items)

    def __len__(self):
        return self.num_items

    def __getitem__(self, i):
        return self.items[i]

    def __str__(self):
        s = 'feats scp: {}\n'.format(self.feats_scp)
        if self.ivector_scp:
            s += 'ivector_scp scp: {}\n'.format(self.ivector_scp)
        s += 'num utt: {}\n'.format(self.num_items)
        return s


class FeatDatasetCollateFunc:

    def __init__(self,
                 model_left_context,
                 model_right_context,
                 frame_subsampling_factor=3,
                 frames_per_chunk=51,
                 ivector_period=10):
        '''
        We need `frame_subsampling_factor` because we want to know
        the number of output frames of different waves in the same batch
        '''
        self.model_left_context = model_left_context
        self.model_right_context = model_right_context
        self.frame_subsampling_factor = frame_subsampling_factor
        self.frames_per_chunk = frames_per_chunk
        self.ivector_period = ivector_period

    def __call__(self, batch):
        '''
        batch is a list of [key, rxfilename]
        '''
        key_list = []
        feat_list = []
        ivector_list = []
        ivector_len_list = []
        output_len_list = []
        subsampled_frames_per_chunk = (self.frames_per_chunk //
                                       self.frame_subsampling_factor)
        for b in batch:
            key, rxfilename, ivector_rxfilename = b
            key_list.append(key)
            feat = kaldi.read_mat(rxfilename).numpy()
            if ivector_rxfilename:
                ivector = kaldi.read_mat(
                    ivector_rxfilename).numpy()  # L // 10 * C
            feat_len = feat.shape[0]
            output_len = (feat_len + self.frame_subsampling_factor -
                          1) // self.frame_subsampling_factor
            output_len_list.append(output_len)
            # now add model left and right context
            feat = _add_model_left_right_context(feat, self.model_left_context,
                                                 self.model_right_context)

            # now we split feat to chunk, then we can do decode by chunk
            input_num_frames = feat.shape[0] - self.model_left_context - self.model_right_context
            for i in range(0, output_len, subsampled_frames_per_chunk):
                # input len:418 -> output len:140 -> output chunk:[0, 17, 34, 51, 68, 85, 102, 119, 136]
                first_output = i * self.frame_subsampling_factor
                last_output = min(input_num_frames,
                                  first_output + (subsampled_frames_per_chunk-1) * self.frame_subsampling_factor)
                first_input = first_output
                last_input = last_output + self.model_left_context + self.model_right_context
                input_x = feat[first_input:last_input+1, :]
                if ivector_rxfilename:
                    ivector_index = (
                        first_output + last_output) // 2 // self.ivector_period
                    input_ivector = ivector[ivector_index, :].reshape(1, -1)
                    feat_list.append(np.concatenate((input_x,
                                                     np.repeat(input_ivector, input_x.shape[0], axis=0)),
                                                    axis=-1))
                else:
                    feat_list.append(input_x)

        padded_feat = pad_sequence(
            [torch.from_numpy(feat).float() for feat in feat_list],
            batch_first=True)

        assert sum([math.ceil(l / subsampled_frames_per_chunk) for l in output_len_list]) \
            == padded_feat.shape[0]

        return key_list, padded_feat, output_len_list
