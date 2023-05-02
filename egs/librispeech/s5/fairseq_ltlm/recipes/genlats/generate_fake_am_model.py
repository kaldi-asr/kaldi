# Copyright 2021 STC-Innovation LTD (Author: Anton Mitrofanov)
import kaldi_io
import argparse
import numpy as np
import pickle
import os
from collections import defaultdict
import logging
import glob
from tqdm import tqdm
import sys
from scipy.special import softmax


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers.clear()

# default output
c_handler = logging.StreamHandler(sys.stderr)
c_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)



class AliStretchModel:
    @staticmethod
    def load_from_file(fname):
        with open(fname, 'rb') as f:
            obj = pickle.load(f)
        return obj

    def save_to_file(self, fname=None):
        if fname is None:
            fname = self.model_path
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument("--stretch_model_path", required=True, type=str,
                            help='Model path for saving or loading stretch aligment model')

    @staticmethod
    def build_kwargs(args):
        kwargs = {'model_path': args.stretch_model_path,
                  'max_pdf_id': args.max_pdf_id}
        return kwargs

    @classmethod
    def build_from_disk(cls, args):
        logger.info(f"Loading saved model from {args.stretch_model_path}")
        return cls.load_from_file(args.stretch_model_path)

    @classmethod
    def build(cls, args, load_from_cashe=True):
        if os.path.exists(args.stretch_model_path):
            if not load_from_cashe:
                raise RuntimeError(f"Model {args.stretch_model_path} already exists")
            cls.build_from_disk(args)
        assert args.max_pdf_id is not None, RuntimeError("--max_pdf_id  required!")
        kwargs = cls.build_kwargs(args)
        return cls(**kwargs)

    def __init__(self, model_path, max_pdf_id):
        self.model_path = model_path
        self.max_pdf_id = max_pdf_id
        self.id2count = np.zeros((max_pdf_id,))
        self.id2seq_count = [defaultdict(int) for _ in range(self.max_pdf_id)]

    def add_utts(self, ali):
        assert len(ali.shape) == 1, RuntimeError(f"Wrong shape in add_utts")
        prev_id = None
        seq_len = 0
        for t_id in ali:
            self.id2count[t_id] += 1
            if t_id != prev_id:
                if prev_id is not None:
                    self.id2seq_count[prev_id][seq_len] += 1
                    seq_len = 0
                prev_id = t_id
            seq_len += 1

    def compute(self):
        logger.info("Starting AliStretchModel Compute")
        if np.any(self.id2count == 0):
            bad_ids = np.where(self.id2count == 0)[0]
            logger.warning(f"Not all pdf ids found in train data. bad pdf_ids = {bad_ids}. shape={bad_ids.shape}"
                           f"({round(bad_ids.shape[-1]/self.max_pdf_id*100, 2)}%)")
            self.id2count[bad_ids] = 1
        for i in range(self.id2count.shape[0]):
            total_count = sum(self.id2seq_count[i].values())
            for seqlen in self.id2seq_count[i]:
                self.id2seq_count[i][seqlen] /= total_count
        self.id2count = np.zeros_like(self.id2count)

    def forward(self, ids):
        # ids - [0, 1, 0,...]
        dup_ids = np.concatenate([np.array([index] * self.sample_seq_len(index), dtype=np.int32) for index in ids])
        return dup_ids

    def sample_seq_len(self, index):
        distr = self.id2seq_count[index]
        if len(distr) == 0:
            return 1
        population, weights = np.asarray(list(distr.keys())), np.asarray(list(distr.values()))
        return np.random.choice(population, p=weights)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Id2LoglikeAMModel:
    @staticmethod
    def load_from_file(fname):
        with open(fname, 'rb') as f:
            key = kaldi_io.read_key(f)
            assert key == 'fam_model', RuntimeError(f"Bad fam model {fname}")
            id2sum = kaldi_io.read_mat(f)
        return id2sum

    def save_to_file(self, fname=None):
        if fname is None:
            fname = self.model_path
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, 'wb') as f:
            kaldi_io.write_mat(f, self.id2sum, key='fam_model')
            # kaldi_io.write_mat(f, self.id2priors.reshape((1, -1)), key='prior')

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument("--id2ll_model_path", required=True, type=str,
                            help='Model path for saving or loading id2loglike model')
        parser.add_argument('--apply_smoothing', action='store_true',
                            help='Add smoothing matrix to avg likelihoods.')
        parser.add_argument('--label_smoothing', default=1, type=float,
                            help='Weight for diagonal elements')
        parser.add_argument('--eps', default=5e-5, type=float,
                            help='Epsilon. all zeros are replaced by eps')
        parser.add_argument('--apply_priors', action='store_true',
                            help="div smoothing matrix by priors vector")

    @staticmethod
    def build_kwargs(args):
        kwargs = {'model_path': args.id2ll_model_path,
                  'max_pdf_id': args.max_pdf_id,
                  'label_smoothing': args.label_smoothing,
                  'eps': args.eps,
                  'apply_priors': args.apply_priors,
                  'apply_smoothing': args.apply_smoothing}
        return kwargs

    @classmethod
    def build_from_disk(cls, args):
        logger.info(f"Loading saved model from {args.id2ll_model_path}")
        id2sum = cls.load_from_file(args.id2ll_model_path)
        obj = cls(args.id2ll_model_path, id2sum=id2sum)
        obj.id2sum = id2sum

    @classmethod
    def build(cls, args, load_from_cashe=True):
        if os.path.exists(args.id2ll_model_path):
            if not load_from_cashe:
                raise RuntimeError(f"Model {args.id2ll_model_path} already exists")
            cls.build_from_disk(args)
        assert args.max_pdf_id is not None, RuntimeError("--max_pdf_id  required!")
        kwargs = cls.build_kwargs(args)
        return cls(**kwargs)

    def __init__(self, model_path, max_pdf_id=None, id2sum=None, label_smoothing=1, eps=2e-4,
                 apply_priors=True, apply_smoothing=True):
        self.model_path = model_path
        self.apply_smoothing=apply_smoothing
        self.label_smoothing = label_smoothing
        self.eps = eps
        self.apply_priors = apply_priors
        if id2sum is not None:
            self.id2sum = id2sum
            self.max_pdf_id = id2sum.shape[-1]
            self.id2count = np.ones((max_pdf_id,))
        else:
            self.max_pdf_id = max_pdf_id
            self.id2sum = np.zeros((max_pdf_id, max_pdf_id))
            self.id2count = np.zeros((max_pdf_id,))

        self.id2priors = np.zeros_like(self.id2count)

    def add_prob(self, index, prob):
        self.id2sum[index] += softmax(prob, axis=-1)
        self.id2count[index] += 1

    def add_utts(self, ids, probs):
        assert len(ids.shape) == 1 and len(probs.shape) == 2, RuntimeError(f"Wrong shape in add_probs")
        assert ids.shape[0] == probs.shape[0] , RuntimeError(f"Ali ({ids.shape[0]}) and features ({probs.shape[0]}) len not the same!")
        for t_id, prob in zip(ids, probs):
            self.add_prob(t_id, prob)

    def compute(self):
        logger.info("Starting Id2LoglikeAMModel Compute")
        # if np.any(self.id2count == 0):
        #     logger.warning(f"Not all pdf ids found in train data. bad pdf_ids = {np.where(self.id2count == 0)}")

        self.id2sum[self.id2sum == 0] = self.eps
        self.id2count[self.id2count == 0] = 1
        self.id2priors = self.id2count / self.id2count.sum()
        if self.apply_smoothing:
            self.id2count += 1
            self.id2sum += self.get_smooth()
        self.id2sum = self.id2sum / self.id2count.reshape(-1, 1)

    def get_smooth(self):
        id2smooth = np.zeros_like(self.id2sum)
        if self.label_smoothing != 1:
            other_p = (1 - self.label_smoothing) / (self.max_pdf_id - 1)
            id2smooth.fill(other_p)
        np.fill_diagonal(id2smooth, self.label_smoothing)
        if self.apply_priors:
            id2smooth /= self.id2priors.reshape(1, -1)
        return id2smooth

    def get_prob(self, index):
        return self.id2sum[index]

    def forward(self, ids):
        # ids - [0, 1, 0,...]
        return np.log(self.id2sum[ids])

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    AliStretchModel.add_args(parser)
    Id2LoglikeAMModel.add_args(parser)
    parser.add_argument('ali_dir')
    parser.add_argument('--max_pdf_id', type=int, default=None, help="Maximum pdf_id")
    args = parser.parse_args()

    ali_stretch_model = AliStretchModel.build(args, load_from_cashe=False)
    id2ll_model = Id2LoglikeAMModel.build(args, load_from_cashe=False)

    logger.info(f"Loading {args.ali_dir}/ali.*.gz")
    utt2ali = {key: ali for key, ali in tqdm(kaldi_io.read_vec_int_ark(f'ark: gunzip -c {args.ali_dir}/ali_pdf.1.gz|'))}
    i = 0
    for key, ali in tqdm(utt2ali.items()):
        i += 1
        ali_stretch_model.add_utts(ali)
    logger.info(f"AliStretchModel processed {i} utterances")
    ali_stretch_model.compute()
    ali_stretch_model.save_to_file()

    logger.info(f"Loaded {len(utt2ali)} alis")
    logger.info(f"Loading logprobs and train model")
    i = 0
    for k, m in tqdm(kaldi_io.read_mat_ark(f'ark: cat {args.ali_dir}/output.1.ark |'), total=len(utt2ali)):
        i += 1
        if k not in utt2ali.keys():
            logger.warning(f"Ali for {k} does not exist")
            continue
        ali = utt2ali[k]
        id2ll_model.add_utts(ali, m)
    logger.info(f"Id2LoglikeAMModel processed {i} utterances")
    id2ll_model.compute()
    id2ll_model.save_to_file()
    logger.info(f"Done.")
