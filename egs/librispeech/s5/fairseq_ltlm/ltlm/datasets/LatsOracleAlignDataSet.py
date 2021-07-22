# Copyright 2021 STC-Innovation LTD (Author: Anton Mitrofanov)
import torch
import numpy as np
import os
import argparse
import logging
import itertools

from ltlm.pyutils.lattice_utils import oracle_path, topsort_lat
from ltlm.Tokenizer import WordTokenizer
from ltlm.datasets import LatsDataSet
from ltlm.pyutils.logging_utils import setup_logger
from ltlm.pyutils.lattice_utils import padding
from ltlm.pyutils.data_utils import parse_lats_data_str
logger = logging.getLogger(__name__)


class LatsOracleAlignDataSet(LatsDataSet):
    """ LatsDataset with extracted oracle paths """
    @staticmethod
    def add_args(parser: argparse.ArgumentParser, prefix='',  add_scale_opts=True, add_def_opts=False):
        LatsDataSet.add_args(parser, prefix, add_scale_opts)
        parser.add_argument(f'--{prefix}ref_text_fname', type=str, default=None,
                            help=f"{prefix} Reference kaldi text. Needs only if lat_t format.")
        parser.add_argument(f'--{prefix}cache_fname', type=str, default=None,
                            help=f"{prefix} preprocessed dataset dump. Not used by default.")
        if add_def_opts:
            parser.add_argument(f"--all_oracle_targets", action='store_true', 
                                help=f'All oracle paths contains in training target')
        return parser

    @classmethod
    def build_kwargs(cls, args, tokenizer, prefix=''):
        kwargs = LatsDataSet.build_kwargs(args, tokenizer, prefix=prefix)
        kwargs['ref_text_fname'] = getattr(args, f'{prefix}ref_text_fname', None)
        kwargs['cache_fname'] = getattr(args, f'{prefix}cache_fname', None)
        kwargs['all_oracle_targets'] = getattr(args, f'all_oracle_targets', False)
        return kwargs

    @classmethod
    def build_from_kwargs(cls, **kwargs):
        """ Собирает класс из аргументов
           :param lats_data: lattice data string
           :param tokenizer:  tokenizer obj
           :param ref_text_fname: reference text.
           :param all_oracle_targets: targets all oracle paths.
           :param data_type: data type. dump or lat_t.
           :param max_len: lattice max len.
           :param clip: clip data.
           :return: cls object
           """
        if 'all_oracle_targets' not in kwargs.keys():
            kwargs['all_oracle_targets'] = False
        obj = cls(tokenizer=kwargs['tokenizer'],  ref_text_fname=kwargs['ref_text_fname'],
                  all_oracle_targets=kwargs['all_oracle_targets'])
        obj.get_data_from_disc(parse_lats_data_str(kwargs['lats_data']), kwargs['data_type'])
        if kwargs['data_type'] == 'lat_t':
            assert kwargs['ref_text_fname'] is not None, \
                RuntimeError(f"For data type lat_t --ref_text_fname required!")
            assert os.path.exists(kwargs['ref_text_fname']), RuntimeError(f"{kwargs['ref_text_fname']} not exist!")
            obj.load_ref(kwargs['ref_text_fname'])
            obj.compute_oracle_ali()
            logger.info(f"Oracle wer is {round(obj.oracle_err_sum/obj.num_ref_words*100,2)}%")
        if kwargs['clip']:
            obj.cliped_data(kwargs['max_len'])
        return obj

    @classmethod
    def build_from_args(cls, args, tokenizer, prefix='', clip=True):
        kwargs = cls.build_kwargs(args, tokenizer, prefix=prefix)
        return cls.build_from_kwargs(**kwargs, clip=clip)

    def __init__(self, tokenizer, ref_text_fname=None,  all_oracle_targets=True):
        super().__init__(tokenizer)
        self.ref_text_fname=ref_text_fname
        self.all_oracle_targets = all_oracle_targets
        self.utt2ref = {}
        self.oracle_err_sum = 0
        self.num_ref_words = 0
        self.utt2ohyp = {}
        self.utt2ali = {}

    def load_ref(self, ref_text_fname):
        self.ref_text_fname = ref_text_fname
        self.utt2ref = super().load_ref(ref_text_fname)

    def add_data_from_dicts(self, data_dicts, recompute_utt2id=True, suffs=['']):
        super().add_data_from_dicts(data_dicts, recompute_utt2id, suffs)

        logger.info(f"utt2ref")
        self.utt2ref = {k+suff: v for data_dict, suff in zip([{'utt2ref': self.utt2ref}] + data_dicts, [''] + suffs) \
                                                             for k, v in data_dict['utt2ref'].items() }
        logger.info(f"utt2ohyp")
        self.utt2ohyp = {k+suff: v for data_dict, suff in zip([{'utt2ohyp': self.utt2ohyp}] + data_dicts, [''] + suffs) \
                                                             for k, v in data_dict['utt2ohyp'].items() }
        logger.info(f"utt2ali")
        self.utt2ali = {k+suff: v for data_dict, suff in zip([{'utt2ali': self.utt2ali}] + data_dicts, [''] + suffs) \
                                                             for k,v in data_dict['utt2ali'].items() }
        self.oracle_err_sum += sum(d['oracle_err_sum'] for d in data_dicts)
        self.num_ref_words += sum(d['num_ref_words'] for d in data_dicts)
        logger.info(f"done")

    def add_data_from_dict(self, data_dict, recompute_utt2id=True, suff=''):
        raise RuntimeError("add_data_from_dict is deprecated")

    def data_to_dict(self):
        out = super().data_to_dict()
        out['utt2ref'] = self.utt2ref
        out['oracle_err_sum'] = self.oracle_err_sum
        out['num_ref_words'] = self.num_ref_words
        out['utt2ohyp'] = self.utt2ohyp
        out['utt2ali'] = self.utt2ali
        out['all_oracle_targets'] = self.all_oracle_targets
        return out

    def __getitem__(self, item):
        lat_item = super().__getitem__(item)
        lat = lat_item['net_input']['src_tokens']
        utt_id = lat_item['utt_id']
        lat_item['ref'] = self.utt2ref[utt_id]
        lat_item['ali'] = self.utt2ali[utt_id]

        y = torch.zeros(lat.shape[0])
        y[lat_item['ali']] = 1
        lat_item['target'] = y
        #logger.info(f"lat item {lat_item}")
        return lat_item

    def cliped_data(self, max_len, clip_one_path=True):
        super().cliped_data(max_len)
        self.orig_clipid2len_id = self.clipid2len_id
        if clip_one_path:
            self.clipid2len_id = [(l, i) for l, i in self.clipid2len_id if len(self.utt2ali[self.id2utt[i]]) < len(self.id2lat[i])]
            all_true_lats = [i for l, i in self.orig_clipid2len_id if len(self.utt2ali[self.id2utt[i]]) >= len(self.id2lat[i])]
            num_clipped = len(all_true_lats)
            self.too_big_lats.extend(all_true_lats)
            logger.info(f'LatsOracleAlignDataSet: Clipping also remove {num_clipped} utts with only oracle paths.'
                        f'({round(num_clipped / len(self.id2lat) * 100, 2)}%).')
        self.print_statistic()

    def collater(self, samples):
        #logger.info(f"Collate {samples}")
        source = [d['net_input']['src_tokens'] for d in samples]
        if len(source) == 0:
            logger.warning(f"collate: Empty source collection. samples={samples}")
            samples = [self[0]]
            source = [d['net_input']['src_tokens'] for d in samples]
 
        #weights = [[d['weights'][i] for d in samples] for i in range(len(samples[0]['weights']))]
        weights = [d['weights'] for d in samples] # B x L x 2
        alis = [d['ali'] for d in samples]
        targets = [d['target'] for d in samples]
        utt_ids = [d['utt_id'] for d in samples]

        batch_x, *other = padding(source, weights, targets)

        batch_w, batch_y = other[0], other[-1]
        ntokens = sum([d['ntokens'] for d in samples])
        return {'net_input': {'src_tokens': batch_x},
                'weights': batch_w, # B x L x 2
                'target': batch_y,
                'ali': alis,
                'ntokens': ntokens,
                'utt_id': utt_ids}
    
    def compute_oracle_ali(self, print_interval=100):
        logger.info("Getting oracle ali.")
        global_err = 0
        global_len = 0
        hyps = {}
        alis = {}
        skip_words = set(self.tokenizer.get_disambig_words_ids())

        total_count = len(self.utt2ref)
        for i, (utt_id, ref) in enumerate(self.utt2ref.items()):
            lat = self.id2lat[self.utt2id[utt_id]]
            if len(lat) > self.max_len:
                logger.info(f"Skip {utt_id}. len ({len(lat)}) > max len ({self.max_len})")
            #logger.debug(f"Process {utt_id}")
            err, oracle_arcs = oracle_path(lat, ref, final_word_id=self.tokenizer.get_eos_id(), skip_words=skip_words,
                                                keep_all_oracle_paths=self.all_oracle_targets)

            global_err += err
            global_len += len(ref) - 2  # except eos and bos
            alis[utt_id] = np.array(oracle_arcs)
            if not self.all_oracle_targets:
                hyp = [self.tokenizer.id2word[lat[i][0]] for i in oracle_arcs[1:-1]]
                hyps[utt_id] = hyp
            if (i+1) % print_interval == 0:
                logger.info(f"Oracle wer processed {i} utts ({round(i/total_count*100, 2)}%).")
        self.oracle_err_sum, self.num_ref_words, self.utt2ohyp, self.utt2ali = global_err, global_len, hyps, alis

    def oracle_wer(self, *args, **kwargs):
        return self.oracle_err_sum/self.num_ref_words*100, self.utt2ohyp, self.utt2ali

    def get_statistic(self):
        stats = super().get_statistic()
        stats['Target arcs'] = sum([len(a) for a in self.utt2ali.values()]) / len(self.id2utt)
        stats['Non-target arcs'] = sum([len(self.id2lat[i]) - len(self.utt2ali[utt]) \
                                        for i, utt in enumerate(self.id2utt)]) / len(self.id2utt)
        return stats


if __name__ == "__main__":
    pass
