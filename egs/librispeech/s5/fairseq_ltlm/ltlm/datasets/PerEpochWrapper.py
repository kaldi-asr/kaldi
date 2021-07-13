# Copyright 2021 STC-Innovation LTD (Author: Anton Mitrofanov)
import argparse
import logging

from fairseq.data import FairseqDataset
from ltlm.datasets import LatsOracleAlignDataSet
logger = logging.getLogger(__name__)


class PerEpochWrapper(FairseqDataset):
    def __init__(self, args, tokenizer, data_config, dataset_cls=LatsOracleAlignDataSet):
        # data_config like [ { "epoch": 1, "lats": "exp/model/decode/lt_egs", "utt_suff": "", "use_once": True},
        #             { "epoch": 2, "lats": "exp/model2/decode/lt_egs", "utt_suff": "_2", "use_once": False},
        #             { "epoch": 2, "lats": "exp/model/decode/lt_egs", "utt_suff": "", "use_once": False}
        #           ],
        self.args = args
        self.tokenizer = tokenizer
        self.data_config = data_config
        self.clip_one_path=args.clip_one_path
        self.epoch2data = dict()
        self.dataset_cls = dataset_cls
        for d in self.data_config:
            e = d.get('epoch', 0)
            if e not in self.epoch2data.keys():
                self.epoch2data[e] = []
            self.epoch2data[e].append(d)
        logger.info(f"Epoch to data map is {self.epoch2data}")
        self.loaded_epoch = 0
        self.__ds = self.load_epoch(self.loaded_epoch)

    def load_epoch(self, epoch):
        assert epoch in self.epoch2data.keys(), RuntimeError(f"Cannot find data config for {epoch} epoch")
        data_conf = self.epoch2data[epoch]
        ds = self.dataset_cls(self.tokenizer)
        ds.get_data_from_disc(data_conf)
        ds.cliped_data(self.args.max_len, clip_one_path=self.clip_one_path)
        return ds

    def set_epoch(self, epoch):
        data_epoch = epoch - 1
        if data_epoch not in self.epoch2data.keys():
            epoch_id = data_epoch % len(self.epoch2data.keys())
            logger.info(f"REAL DATA EPOCH IS {data_epoch // len(self.epoch2data.keys())}")
            data_epoch = list(sorted(self.epoch2data.keys()))[epoch_id]
        if self.loaded_epoch == data_epoch:
            logger.info(f"Data for epoch {epoch} already loaded")
            return
        logger.info(f"Mapping epoch {epoch} to {data_epoch} data epoch.")
        self.__ds = self.load_epoch(data_epoch)
        self.loaded_epoch = data_epoch

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False

    def __getitem__(self, index):
        return self.__ds.__getitem__(index)

    def __len__(self):
        return self.__ds.__len__()

    def collater(self, samples):
        return self.__ds.collater(samples)

    def num_tokens(self, index):
        return self.__ds.collater(index)

    def size(self, index):
        return self.__ds.size(index)

    def ordered_indices(self):
        return self.__ds.ordered_indices()

    @property
    def supports_prefetch(self):
        return self.__ds.supports_prefetch

    def attr(self, attr: str, index: int):
        return self.__ds.attr(attr, index)

    def prefetch(self, indices):
        return self.__ds.prefetch(indices)

    def get_batch_shapes(self):
        return self.__ds.get_batch_shapes()

    def batch_by_size(
            self,
            indices,
            max_tokens=None,
            max_sentences=None,
            required_batch_size_multiple=1,
    ):
        return self.__ds.batch_by_size(indices,
                                       max_tokens=max_tokens,
                                       max_sentences=max_sentences,
                                       required_batch_size_multiple=required_batch_size_multiple)

    def filter_indices_by_size(self, indices, max_sizes):
        return self.__ds.filter_indices_by_size(indices, max_sizes)

    @property
    def supports_fetch_outside_dataloader(self):
        return self.__ds.supports_fetch_outside_dataloader
