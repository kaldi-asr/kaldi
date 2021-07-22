# Copyright 2021 STC-Innovation LTD (Author: Anton Mitrofanov)
import torch
import argparse
from collections import defaultdict

import os


from fairseq.tasks import FairseqTask, register_task
from ltlm.datasets import LatsOracleAlignDataSet, PerEpochWrapper

from ltlm.pyutils.data_utils import parse_lats_json
from ltlm.Tokenizer import WordTokenizer
import logging
from ltlm.eval import compute_model_wer, RESCORE_STRATEGIES
logger = logging.getLogger(__name__)

#DATASETS = ['valid']
TRAINING_TYPES = frozenset(['oracle_path'])


def add_training_type_args(parser):
    parser.add_argument("--training_type", choices=TRAINING_TYPES, default='oracle_path',
                        help=f"oracle_path - training 'is arc in oracle path'.")

def get_data_cls(args):
    if args.training_type == 'oracle_path':
        return LatsOracleAlignDataSet
    RuntimeError(f'Bad training_type {args.training_type}')


@register_task('rescoring_task')
class RescoringTask(FairseqTask):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser, add_def_opts=True, add_data_opts=True):
        WordTokenizer.add_args(parser)
        if add_data_opts:
            parser.add_argument("--data_json", type=str, required=True, help="Data egs config in json format")
        add_training_type_args(parser)
        parser.add_argument("--model_weight", type=float, default=19,
                            help="Rescoring Final weight = (hclg_score*lmwt + acoustic_score*acwt) + model_scores * model_weight")
        parser.add_argument('--dataloader_nj', type=int, default=12,
                            help='DataLoaders number of threads')
        parser.add_argument('--infer_btz', type=int, default=8,
                            help='Batch size for evaluating model')
        parser.add_argument('--hyp_filter', type=str, default='cat',
                            help="Filter pipe for preprocess hyps before scoring")
        parser.add_argument(f'--lmwt', type=float, default=1,
                            help=f'lmwt for sum Acoustic score and lm(hcl) score')
        parser.add_argument("--not_clip_one_path", action='store_true', default=False, 
                           help="Don't remove lattices which contains only one path")
        parser.add_argument('--strategy', choices=RESCORE_STRATEGIES, default='base',
                            help="Rescoring strategy. base - simple forward algorithm. \n"
                                 "norm - normalize lt probs. \n"
                                 "only_forks - set 1 for arcs without alternatives. \n")
        if add_def_opts:
            parser.add_argument(f'--max_len', type=int, default=None,
                                help=f"Default Max len for lattice")
            parser.add_argument(f"--all_oracle_targets", action='store_true',
                                help=f'All oracle paths contains in training target')

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg)
        self.tokenizer = WordTokenizer.build_from_args(self.cfg)
        cfg.clip_one_path = not cfg.not_clip_one_path

        self.data_config = None
        self.train_conf = None
        self.val_test_conf = None
        self.datasets_extras = None
        self.criterion = None
        self.data_cls = get_data_cls(cfg)

    def load_data_config(self):
        self.data_config = parse_lats_json(self.cfg.data_json)
        self.train_conf = self.data_config['train']
        self.val_test_conf = {"valid": self.data_config.get('valid', []),
                              "test": self.data_config.get('test', [])}
        self.datasets_extras = {"valid": [], 'test': []}

    def load_dataset(self, split, combine=False, epoch=0, **kwargs):
        if self.data_config is None:
            self.load_data_config()
        if split in self.datasets.keys():
            logger.warning(f"load_dataset {split} already loaded")
            return
        if split == 'train':
            self.datasets[split] = PerEpochWrapper(self.cfg, self.tokenizer, self.train_conf,
                                                   dataset_cls=self.data_cls)
            return

        for set_data in self.val_test_conf[split]:
            lats_data, ref_text = set_data['lats'], set_data['ref']
            assert ref_text is not None, RuntimeError(f"ref_text for {split}:({lats_data}) required!")
            ds = self.data_cls.build_from_kwargs(lats_data=lats_data,
                                                 tokenizer=self.tokenizer,
                                                 max_len=self.cfg.max_len,
                                                 ref_text_fname=ref_text,
                                                 data_type='dump',
                                                 clip=True)
            logger.info(f"For split {split}:({lats_data}) ORACLE WER is {round(ds.oracle_wer()[0], 2)}")
            wer = compute_model_wer(None,
                                    ds,
                                    ref_text,
                                    btz=self.cfg.infer_btz,
                                    model_weight=0,
                                    lmwt=self.cfg.lmwt,
                                    progress_bar=False,
                                    dataloader_nj=self.cfg.dataloader_nj,
                                    hyp_filter=self.cfg.hyp_filter)
            logger.info(f"For split {split}:({lats_data}) wer without rescoring is {wer}")
            if self.datasets.get(split, None):
                self.datasets_extras[split].append(ds)
            else:
                self.datasets[split] = ds

    def begin_epoch(self, epoch, model):
        self.criterion.epoch = epoch

    def begin_valid_epoch(self, epoch, model):
        if 'valid' not in self.datasets.keys():
            logger.info("Valid is empty. Skip compute wer")
            return
        for ds in [self.datasets['valid'], *self.datasets_extras.get('valid', [])]:
            valid_wer = compute_model_wer(model,
                                          ds,
                                          ds.ref_text_fname,
                                          btz=self.cfg.infer_btz,
                                          model_weight=self.cfg.model_weight,
                                          lmwt=self.cfg.lmwt,
                                          progress_bar=False,
                                          device=model.device,
                                          dataloader_nj=self.cfg.dataloader_nj,
                                          hyp_filter=self.cfg.hyp_filter)
            logger.info(f"Epoch {epoch}. Valid wer is {valid_wer}")

    def max_positions(self):
        return self.cfg.max_positions

    @property
    def source_dictionary(self):
        return self.tokenizer

    @property
    def target_dictionary(self):
        # kostil for fairseq
        return self.tokenizer

    def build_criterion(self, cfg):
        self.criterion = super().build_criterion(cfg)
        return self.criterion
