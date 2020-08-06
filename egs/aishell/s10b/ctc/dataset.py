#!/usr/bin/env python3

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import os
import logging

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import kaldi


def get_ctc_dataloader(feats_scp,
                       labels_scp=None,
                       batch_size=1,
                       shuffle=False,
                       num_workers=0,
                       model_left_context=0,
                       model_right_context=0,
                       world_size=None,
                       local_rank=None):

    dataset = CtcDataset(feats_scp=feats_scp, labels_scp=labels_scp)

    collate_fn = CtcDatasetCollateFunc(model_left_context=model_left_context,
                                       model_right_context=model_right_context)

    if world_size:
        logging.info('world_size: {}'.format(world_size))
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=local_rank, shuffle=shuffle)
        # sampler and shuffle are mutually exclusive;
        # it will raise an exception if you set both
        shuffle = False

    else:
        sampler = None

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            collate_fn=collate_fn,
                            sampler=sampler)

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


class CtcDataset(Dataset):

    def __init__(self, feats_scp, labels_scp=None):
        '''
        Args:
            feats_scp: filename for feats.scp
            labels_scp: if provided, it is the filename of labels.scp
        '''
        assert os.path.isfile(feats_scp)
        if labels_scp:
            assert os.path.isfile(labels_scp)
            logging.info('labels scp: {}'.format(labels_scp))
        else:
            logging.warn('No labels scp is given.')

        # items is a dict of [uttid, feat_rxfilename, None]
        # or [uttid, feat_rxfilename, label_rxfilename] if labels_scp is not None
        items = dict()

        with open(feats_scp, 'r') as f:
            for line in f:
                # every line has the following format:
                # uttid feat_rxfilename
                uttid_rxfilename = line.split()
                assert len(uttid_rxfilename) == 2

                uttid, rxfilename = uttid_rxfilename

                assert uttid not in items

                items[uttid] = [uttid, rxfilename, None]

        if labels_scp:
            expected_count = len(items)
            n = 0
            with open(labels_scp, 'r') as f:
                for line in f:
                    # every line has the following format:
                    # uttid rxfilename
                    uttid_rxfilename = line.split()

                    assert len(uttid_rxfilename) == 2

                    uttid, rxfilename = uttid_rxfilename

                    assert uttid in items

                    items[uttid][-1] = rxfilename

                    n += 1

            # every utterance should have a label if
            # labels_scp is given
            assert n == expected_count

        self.items = list(items.values())
        self.num_items = len(self.items)
        self.feats_scp = feats_scp
        self.labels_scp = labels_scp

    def __len__(self):
        return self.num_items

    def __getitem__(self, i):
        '''
        Returns:
            a list [key, feat_rxfilename, label_rxfilename]
            Note that label_rxfilename may be None.
        '''
        return self.items[i]

    def __str__(self):
        s = 'feats scp: {}\n'.format(self.feats_scp)

        if self.labels_scp:
            s += 'labels scp: {}\n'.format(self.labels_scp)

        s += 'num utterances: {}\n'.format(self.num_items)

        return s


class CtcDatasetCollateFunc:

    def __init__(self, model_left_context=0, model_right_context=0):
        self.model_left_context = model_left_context
        self.model_right_context = model_right_context

    def __call__(self, batch):
        '''
        Args:
            batch: a list of [uttid, feat_rxfilename, label_rxfilename].
                   Note that label_rxfilename may be None.

        Returns:
            uttid_list: a list of utterance id

            feat: a 3-D float tensor of shape [batch_size, seq_len, feat_dim]

            feat_len_list: number of frames of each utterance before padding

            label_list: a list of labels of each utterance; It may be None.

            label_len_list: label length of each utterance; It is None if label_list is None.
        '''
        uttid_list = []  # utterance id of each utterance
        feat_len_list = []  # number of frames of each utterance
        label_list = []  # label of each utterance
        label_len_list = []  # label length of each utterance

        feat_list = []

        for b in batch:
            uttid, feat_rxfilename, label_rxfilename = b

            uttid_list.append(uttid)

            feat = kaldi.read_mat(feat_rxfilename).numpy()

            # use the length before padding
            feat_len_list.append(feat.shape[0])

            feat = _add_model_left_right_context(feat, self.model_left_context,
                                                 self.model_right_context)

            feat = torch.from_numpy(feat).float()

            feat_list.append(feat)

            if label_rxfilename:
                label = kaldi.read_vec_int(label_rxfilename)
                assert 0 not in label

                # we will use frame subsampling factor == 3
                assert len(label) < feat_len_list[-1] / 3

                label_list.extend(label)
                label_len_list.append(len(label))

        feat = pad_sequence(feat_list, batch_first=True)

        if not label_list:
            label_list = None
            label_len_list = None

        return uttid_list, feat, feat_len_list, label_list, label_len_list


def _test_dataset():
    feats_scp = 'data/train_sp/feats.scp'
    labels_scp = 'data/train_sp/labels.scp'

    dataset = CtcDataset(feats_scp=feats_scp, labels_scp=labels_scp)

    print(dataset)


def _test_dataloader():
    feats_scp = 'data/test/feats.scp'
    labels_scp = 'data/test/labels.scp'

    dataset = CtcDataset(feats_scp=feats_scp, labels_scp=labels_scp)

    dataloader = DataLoader(dataset,
                            batch_size=2,
                            num_workers=10,
                            shuffle=True,
                            collate_fn=CtcDatasetCollateFunc())
    i = 0
    for batch in dataloader:
        uttid_list, feat, feat_len_list, label_list, label_len_list = batch
        print(uttid_list, feat.shape, feat_len_list, label_len_list)
        i += 1
        if i > 10:
            break


if __name__ == '__main__':
    #  _test_dataset()
    _test_dataloader()
