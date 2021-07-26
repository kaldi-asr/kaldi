# Copyright 2021 STC-Innovation LTD (Author: Anton Mitrofanov)
import torch.utils.data
import torch
import numpy as np
import os
import sys
from glob import glob
import pickle
from fairseq.data import FairseqDataset
from collections import Counter
import tqdm
import argparse
import time
import logging


from ltlm.pyutils.data_utils import parse_lats_data_str
from ltlm.pyutils.lattice_utils import (topsort_lat,
                                        oracle_path,
                                        padding,
                                        parse_lats,
                                        WORD_ID)
logger = logging.getLogger(__name__)


class LatsDataSet(FairseqDataset):
    """ Kaldi lattices """
    @staticmethod
    def add_args(parser: argparse.ArgumentParser, prefix='', add_scale_opts=True):
        parser.add_argument(f'--{prefix}data', type=str, default=None,
                            help=f"{prefix} data archive(s). Comma separable list of glob patterns. for each dump can be added suffix after :. "
                            f"Example exp/model/decode/lt_egs/lat.*.dump,exp/model/decode/lt_egs_beam8/lat.*.dump:_beam8")
        parser.add_argument(f'--{prefix}data_type', type=str, choices=['dump', 'lat_t'],  default='dump',
                            help=f"lat_t - kaldi ark,t format. Dump - pickle data dump(s).")
        if not add_scale_opts:
            return
        parser.add_argument(f'--{prefix}max_len', type=int, default=600,
                            help=f"{prefix} Max len for lattice. If len greater than max_len lat will be removed."
                            f"Default: Same as --max_len")
        return parser

    @classmethod
    def build_kwargs(cls, args, tokenizer, prefix=''):
        max_len = getattr(args, f'{prefix}max_len', None)
        if max_len is None:
            max_len = getattr(args, 'max_len', float('inf'))
        return {'lats_data': getattr(args, f'{prefix}data'),
                'data_type': getattr(args, f'{prefix}data_type'),
                'tokenizer': tokenizer,
                'max_len': max_len}

    @classmethod
    def build_from_args(cls, args, tokenizer, prefix=''):
        kwargs = cls.build_kwargs(args, tokenizer, prefix=prefix)
        return cls.build_from_kwargs(**kwargs)

    @classmethod
    def build_from_kwargs(cls, lats_data, tokenizer, data_type, max_len):
        obj = cls(tokenizer=tokenizer)

        data_dict_list = parse_lats_data_str(lats_data)
        obj.get_data_from_disc(data_dict_list, data_type)
        obj.cliped_data(max_len)
        return obj

    def save_to_file(self, fname):
        """ Save dataset to disk"""
        logger.info(f'Saving dataset to disk: {fname}')
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_file(fname):
        """ Load saved dataset from disk"""
        logger.info(f"Loading cached dataset {fname}")
        with open(fname, 'rb') as f:
            obj = pickle.load(f)
        return obj

    def __init__(self, tokenizer):
        self.data_pattern = ''
        self.id2utt = []
        self.utt2id = {}
        self.id2lat = []
        self.id2weights = []
        self.id2p_ali = []
        self.tokenizer = tokenizer
        self.clipid2len_id = []
        self.too_big_lats = []
        self.max_len = float('inf')

    def get_data_from_disc(self, data_dict_list, data_type='dump'):
        assert data_type in ['lat_t', 'dump'], RuntimeError(f'Wrong data type {data_type}')
        logger.info(f" data dict list is {data_dict_list}")
        self.data_pattern = data_dict_list
        # all_data = self.parse_lats_data_str(lats_data)

        data_dicts = []
        suffs = []
        for data_dict in data_dict_list:
            curr_data = data_dict['lats']
            suff = data_dict['utt_suff']
            # epoch = data_dict['epoch']
            logger.debug(f"Loading data from {curr_data}. Suffix={suff}.")
            # Read data from disk
            if data_type == 'lat_t':
                if curr_data == '-':
                    logger.info(f"Reading lattice arc from std input.")
                    self.add_data_from_latt(sys.stdin.readlines(), recompute_utt2id=False, suff=suff)
                else:
                    if os.path.isdir(curr_data):
                        logger.info(f"{curr_data} is directory. Loading {curr_data}/lat.*.t")
                        curr_data = os.path.join(curr_data, 'lat.*.t')
                    for fn in glob(curr_data):
                        logger.debug(f"Reading {fn}. Type = {data_type}")
                        with open(fn, 'r', encoding='utf-8') as f:
                            self.add_data_from_latt(f.readlines(), recompute_utt2id=False, suff=suff)
            elif data_type == 'dump':
                if os.path.isdir(curr_data):
                    logger.info(f"{curr_data} is directory. Loading {curr_data}/lat.*.dump")
                    curr_data = os.path.join(curr_data, 'lat.*.dump')
                for fn in glob(curr_data):
                    logger.debug(f"Reading {fn}. Type = {data_type}")
                    with open(fn, 'rb') as f:
                        data_dict = pickle.load(f)
                    data_dicts.append(data_dict)
                    suffs.append(suff)
            else:
                raise RuntimeError(f'Wrong data type {data_type}')
        if len(data_dicts) > 0:
            logger.info(f"Combining dicts to dataset")
            self.add_data_from_dicts(data_dicts, recompute_utt2id=False, suffs=suffs)
        assert len(self.id2utt) > 0, RuntimeError(f"{data_dict_list} is empty.")
        logger.info(f"Loaded {len(self.id2utt)} utterance from {data_dict_list}")
        self.utt2id = {utt: i for i, utt in enumerate(self.id2utt)}

    def add_data_from_latt(self, lines, recompute_utt2id=True, suff=''):
        utt2lat = parse_lats(lines)
        if suff:
            utt2lat = {k+suff:v for k,v in utt2lat.items()}
        id2utt = utt2lat.keys()
        self.id2utt.extend(id2utt)
        converted_comp_lat = [self.compact_lat_to_lat(utt2lat[utt]) for utt in id2utt]
        lats = [topsort_lat(l) for l, _, _ in converted_comp_lat]
        weights = [np.array(w) for _, w, _ in converted_comp_lat]
        phone_ali = [a for _, _, a in converted_comp_lat]
        self.id2lat.extend(lats)
        self.id2weights.extend(weights)
        self.id2p_ali.extend(phone_ali)
        if recompute_utt2id:
            self.utt2id = {utt: i for i, utt in enumerate(self.id2utt)}
    
    def add_data_from_dicts(self, data_dicts, recompute_utt2id=True, suffs=['']):
        for i, (d, s) in  enumerate(zip(data_dicts, suffs)):
            self.add_list_data_from_dict(d, recompute_utt2id=(recompute_utt2id and i == len(data_dicts) - 1), suff=s)


    def add_list_data_from_dict(self, data_dict, recompute_utt2id=True, suff=''):
        if suff:
            id2utt = [u+suff for u in data_dict['id2utt']]
        else:
            id2utt = data_dict['id2utt']
        self.id2utt.extend(id2utt)
        self.id2lat.extend(data_dict['id2lat'])
        self.id2weights.extend(data_dict['id2weights'])
        self.id2p_ali.extend(data_dict['id2p_ali'])
        if recompute_utt2id:
            self.utt2id = {utt: i for i, utt in enumerate(self.id2utt)}

    def data_to_dict(self):
        out = {
            "id2utt": self.id2utt,
            "id2lat": self.id2lat,
            "id2weights": self.id2weights,
            "id2p_ali": self.id2p_ali,
        }
        return out

    def load_ref(self, ref_text_fname):
        #self.ref_text_fname = ref_text_fname
        utts = set(self.id2utt)
        with open(ref_text_fname, 'r', encoding='utf-8') as f:
            utt2ref = {utt_id: self.tokenizer.encode([line], bos=True, eos=True)[0] for utt_id, line in
                       map(lambda x: x.split(' ', 1) if x.find(' ') != -1 else [x.strip(), ''], f) if utt_id in utts}
        logger.info(f"LatsDataSet: loaded {len(utt2ref)} ref.")
        assert len(utt2ref) == len(self.id2utt), f'Utterance name {set(self.id2utt) - set(utt2ref.keys())} not found in file {ref_text_fname}'
        return utt2ref

    def cliped_data(self, max_len):
        # Clip max len
        self.max_len = max_len
        self.clipid2len_id = sorted([(len(lat), i) for i, lat in enumerate(self.id2lat) if len(lat) < max_len])
        self.too_big_lats = [i for i, lat in enumerate(self.id2lat) if len(lat) >= max_len]
        num_clipped = len(self.id2lat) - len(self.clipid2len_id)
        logger.info(f'LatsDataSet: Clipping with max_len={max_len} remove {num_clipped} utts '
                    f'({round(num_clipped / len(self.id2lat) * 100, 2)}%).')

    def get_removed_utts(self):
        lats = [self.id2lat[i] for i in self.too_big_lats]
        weights = [self.id2weights[i] for i in self.too_big_lats]
        utts = [self.id2utt[i] for i in self.too_big_lats]
        return lats, weights, utts

    def __getitem__(self, item):
        #if type(item) == int:
        if isinstance(item, str):
            i = self.utt2id[item]
            utt_id = item
        else:
            i = self.clipid2len_id[item][1]
            utt_id = self.id2utt[i]

            #raise RuntimeError(f'LatsDataSet:__getitem__ Bad item {item}')
        weights = torch.Tensor(self.id2weights[i]) # L X 2
        #logger.info(f'W shape: {weights.shape}')
        lat = topsort_lat(self.id2lat[i])
        return {'net_input': {'src_tokens': torch.LongTensor(lat), },
                'weights': weights,
                'utt_id': utt_id,
                'ntokens': weights.shape[0]}

    def __len__(self):
        return len(self.clipid2len_id)

    def size(self, index: int):
        #assert isinstance(index, int), NotImplementedError(f"size implemented only for int index, "
        #                                               f"got {index} type {type(index)}")
        return self.clipid2len_id[index][0]

    def collater(self, samples):
        source = [d['net_input']['src_tokens'] for d in samples]
        #weights = [[d['weights'][i] for d in samples] for i in range(len(samples[0]['weights']))]
        weights = [d['weights'] for d in samples]
        utt_ids = [d['utt_id'] for d in samples]
        batch_x, batched_ws = padding(source, weights)
        #logger.info(f"W_batch shape: {batched_ws.shape}")
        ntokens = sum([d['ntokens'] for d in samples])
        return {'net_input': {'src_tokens': batch_x},
                'weights': batched_ws, # B x L x  x 
                'ntokens': ntokens,
                'utt_id': utt_ids}

    def num_tokens(self, index):
        return self.size(index)

    def compact_lat_to_lat(self, compact_lat):
        """ Converting kaldi compact lattice in to my lat format"""
        out_lat = []
        out_weights = []
        out_phone_ali = []

        new_id_shift = 2
        eos_arcs = set()
        max_state_id = -1

        # Add BOS symbol
        out_lat.append((self.tokenizer.get_bos_id(), 1, 2))
        out_weights.append((0, 0))
        out_phone_ali.append('')

        # Add words
        for arc in compact_lat:
            if len(arc) == 4:
                # EOS arc
                eos_arcs.add(arc)
            elif len(arc) == 6:
                # arc with word
                state_from, state_to, word_id, w_hcl, w_am, ali = arc
                new_state_from, new_state_to = state_from + new_id_shift, state_to + new_id_shift
                out_lat.append((word_id, new_state_from, new_state_to))
                out_weights.append((w_hcl, w_am))
                out_phone_ali.append(ali)
                if new_state_to > max_state_id:
                    max_state_id = new_state_to
            else:
                raise RuntimeError("unknown arc len in compact_lat_to_lat. Need to debug")
        # Add EOS symbols
        word_id = self.tokenizer.get_eos_id()
        new_state_to = max_state_id + 1
        for arc in eos_arcs:
            state_from, w_hcl, w_am, ali = arc
            new_state_from = state_from + new_id_shift
            out_lat.append((word_id, new_state_from, new_state_to))
            out_weights.append((w_hcl, w_am))
            out_phone_ali.append(ali)
        return out_lat, out_weights, out_phone_ali

    def get_compact_lattices(self):
        comp_lats = {}
        new_id_shift = 2
        for i, (utt_id, lat, weights, ali) in enumerate(zip(self.id2utt, self.id2lat, self.id2weights, self.id2p_ali)):
            comp_lat = []
            for (word_id, state_from, state_to), (w_hcl, w_ac), p_ali in zip(lat, weights, ali):
                state_from, state_to = state_from - new_id_shift, state_to - new_id_shift
                if word_id == self.tokenizer.get_bos_id():
                    continue
                if word_id == self.tokenizer.get_eos_id():
                    comp_arc = f'{state_from} {w_hcl},{w_ac},{p_ali}'
                    comp_lat.append(comp_arc)
                    continue
                comp_arc = f'{state_from} {state_to} {word_id} {w_hcl},{w_ac},{p_ali}'
                comp_lat.append(comp_arc)
            comp_lats[utt_id] = comp_lat
        return comp_lats

    def write_compact_lattices(self, fname):
        comp_lats = self.get_compact_lattices()
        with open(fname, 'w') as f:
            for utt, cl in sorted(comp_lats.items()):
                f.write(f"{utt}\n")
                f.write('\n'.join(cl))
                f.write('\n\n')
            f.flush()

    def get_utt_id(self, i):
        return self.id2utt[i]

    def get_utt_weights(self, utt):
        return self.id2weights[self.utt2id[utt]]

    def get_utt_lat(self, utt):
        return self.id2lat[self.utt2id[utt]]

    def set_utt_weights(self, utt, lm_weight, ac_weight):
        weights = self.id2weights[self.utt2id[utt]]
        weights[0] = lm_weight
        weights[1] = ac_weight

    def print_lat(self, *args, **kwargs):
        self.tokenizer.print_lat(*args, **kwargs)

    def get_name(self):
        return self.data_pattern

    def get_statistic(self):
        stats = {}
        stats['num_lat'] = len(self.id2lat)
        stats['tokenizer_num_words'] = len(self.tokenizer)
        stats['avg_lat_len'] = sum([len(l) for l in self.id2lat])/len(self.id2lat)
        return stats

    def print_statistic(self, out=logger.info):
        out("Dataset Statistic")
        for k, v in self.get_statistic().items():
            out(f"{k} = {v}")

    def compare(self, other, normalize=True, progress_bar=True, compare_clipped=False):
        bag_stats = []
        bag_uniq_stats = []
        composition_stat = []
        num_arcs_compare = []
        if compare_clipped:
            iterator = map(lambda li: li[1] , self.clipid2len_id)
        else:
            iterator = range(len(self.id2utt))
        if progress_bar:
            iterator = tqdm.tqdm(iterator, total=len(self.id2utt))
        for i in iterator:
            utt = self.id2utt[i]
            items = (self[utt], other[utt])
            # print(items)
            delta_num_arcs = items[0]['ntokens'] - items[1]['ntokens']
            num_arcs_compare.append(delta_num_arcs)

            bags = [item['net_input']['src_tokens'][:, WORD_ID] for item in items]
            # print(bags)
            uniq_bags = [set(b) for b in bags]
            # print(uniq_bags)
            intersection = set.intersection(*uniq_bags)
            union = set.union(*uniq_bags)
            delta_uniq_bags = len(intersection) / len(union)
            bag_uniq_stats.append(delta_uniq_bags)

            counters = [Counter(b) for b in bags]
            counters_intersection = counters[0] & counters[1]
            counters_union = counters[0] | counters[1]
            delta_bags = sum(counters_intersection.values()) / sum(counters_union.values())
            bag_stats.append(delta_bags)

        if not normalize:
            return {
                "utts": self.id2utt,
                'Bag of words stats': bag_stats,
                'Bag of uniq words stats': bag_uniq_stats,
                'diff between num arcs': num_arcs_compare,
                'composition_stat': composition_stat
            }
        return {
            "utts": self.id2utt,
            'Average bag of words correlation': sum(bag_stats) / len(bag_stats),
            'Average bag of uniq words correlation': sum(bag_uniq_stats) / len(bag_uniq_stats),
            'Average different between num arcs in first and second lat': sum(num_arcs_compare) / len(num_arcs_compare),
            'Average absolute different between num arcs': sum(map(abs, num_arcs_compare)) / len(num_arcs_compare),
            'composition_stat': 0
        }

    def oracle_wer(self, ref_text_fname, print_interval=5000):
        global_err = 0
        global_len = 0
        hyps = {}
        alis = {}
        skip_words = set(self.tokenizer.get_disambig_words_ids())

        with open(ref_text_fname, 'r', encoding='utf-8') as f:
            utt2ref = {utt_id: self.tokenizer.encode([line], bos=True, eos=True)[0]
                       for utt_id, line in
                       map(lambda x: x.split(' ', 1) if x.find(' ') != -1 else [x.strip(), ''], f.readlines())}
        total_count = len(utt2ref)
        for i, (utt_id, ref) in enumerate(utt2ref.items()):
            lat = self.id2lat[self.utt2id[utt_id]]
            err, ali = oracle_path(lat, ref, final_word_id=self.tokenizer.get_eos_id(), skip_words=skip_words,
                                   keep_all_oracle_paths=False)
            global_err += err
            global_len += len(ref) - 2  # except eos and bos
            alis[utt_id] = ali
            hyp = [self.tokenizer.id2word[lat[i][0]] for i in ali[1:-1]]
            hyps[utt_id] = hyp
            if i % print_interval == 0:
                logging.info(f"Oracle wer processed {round(i / total_count * 100, 2)}%. ")
        return global_err / global_len * 100, hyps, alis


if __name__ == "__main__":
    pass
