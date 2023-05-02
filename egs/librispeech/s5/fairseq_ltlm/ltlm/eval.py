# Copyright 2021 STC-Innovation LTD (Author: Anton Mitrofanov)
import torch
import torch.utils.data
import logging
from tqdm import tqdm
import argparse
import numpy as np
from datetime import datetime
from fairseq import checkpoint_utils, data, options, tasks

from ltlm.pyutils.logging_utils import setup_logger
from ltlm.models import LTLM
from ltlm.datasets import LatsDataSet
from ltlm.pyutils.lattice_utils import collate_lats, best_path_nloglike
from ltlm.pyutils.kaldi_utils import compute_wer
from ltlm.Tokenizer import WordTokenizer
from ltlm.tasks import rescoring_task
logger = logging.getLogger(__name__)

RESCORE_STRATEGIES = frozenset(['base', 'bce'])


def compute_model_wer(model, dataset, ref_fname,
                      keep_tmp=False, acwt=1, lmwt=1, hyp_filter='cat', **kwargs):
    logger.info(f'Compute wer. acwt={acwt}. lmwt={lmwt}. hyp_filter={hyp_filter}. kwargs={kwargs}')
    is_model_training = False
    if model is not None:
        is_model_training = model.training
        model.eval()
    start = datetime.now()
    utt2hyp = get_rescoring_hyps(model, dataset, acwt=acwt, lmwt=lmwt, **kwargs)
    logger.info(f"Rescoring elapsed {datetime.now() - start}")
    removed_lats, removed_lats_ws, removed_lats_utts = dataset.get_removed_utts()
    tokenizer = dataset.tokenizer
    if len(removed_lats) > 0:
        final_word_id = tokenizer.get_eos_id()
        for lat, weights, utt in zip(removed_lats, removed_lats_ws, removed_lats_utts):
            p = weights[:, 0] * lmwt + weights[:, 1] * acwt
            _, hyp = best_path_nloglike(lat, p, final_word_id=final_word_id)
            hyp_line = tokenizer.decode([[arc[0] for arc in hyp]])[0]
            assert hyp_line[0] == '<s>' and hyp_line[-1] == '</s>', RuntimeError(f"{hyp_line}")
            utt2hyp[utt] = hyp_line[1:-1]
    wer_str = compute_wer(ref_fname, utt2hyp, keep_tmp=keep_tmp, hyp_filter=hyp_filter)[0]  # 0 - wer, 1 - ser
    if model is not None and is_model_training:
        model.train()
    return wer_str.strip()


def get_scores(model, dataset, device='cpu', btz=1, progress_bar=True, dataloader_nj=12):
    if model is not None:
        model = model.to(device)
    utt2score = {}
    iterator = torch.utils.data.DataLoader(dataset, batch_size=btz, shuffle=False,
                                           collate_fn=dataset.collater, num_workers=dataloader_nj)
    if progress_bar:
        iterator = tqdm(iterator)

    with torch.no_grad():
        for i, (batched_samples) in enumerate(iterator):
            x = batched_samples['net_input']['src_tokens']
            btz, sl = x.size(0), x.size(1)
            probs, _ = model(x.to(device), apply_sigmoid=True)
            probs_np = probs.view(btz, sl).cpu().numpy()
            for utt, utt_prob, words_ids in zip(batched_samples['utt_id'], probs_np, x[:,:,0]):
                utt2score[utt] = utt_prob[words_ids != 0]

    return utt2score


def apply_strategy(lat, lt_probs, strategy):
    assert strategy in RESCORE_STRATEGIES, RuntimeError(f"Bad rescoring strategy {strategy}. "
                                                        f"Strategy must be in '{RESCORE_STRATEGIES}'")
    lt_score = 0
    if lt_probs is None:
        pass
    elif strategy == 'base':
        lt_score = - np.log(lt_probs)
    elif strategy=='bce':
        lt_score = - np.log(lt_probs) + np.log(1 - lt_probs)
        #den_scores = - np.log(1 - lt_probs)
    else:
        RuntimeError(f"Unknown strategy {strategy}")

    return lt_score


def get_rescoring_hyps(model, dataset,  acwt=1, lmwt=1, model_weight=1.3, strategy='base', **kwargs):
    assert strategy in RESCORE_STRATEGIES, RuntimeError(f"Bad rescoring strategy {strategy}. "
                                                        f"Strategy must be in '{RESCORE_STRATEGIES}'")
    logger.info(f"Strategy in {strategy}")

    tokenizer = dataset.tokenizer
    final_word_id = tokenizer.get_eos_id()
    utt2hyp = {}

    if model_weight == 0:
        utt2score = {u: None for u in dataset.id2utt}
    else:
        utt2score = get_scores(model, dataset, **kwargs)

    for utt, lt_probs in utt2score.items():
        item = dataset[utt]
        # 'net_input': {'src_tokens': lat, },
        #                 'weights': weights, # L x 2
        #                 'utt_id': utt_id,
        #                 'ntokens': weights.shape[0]
        lat = item['net_input']['src_tokens']
        weights = item['weights'] # L x 2 
        axl_weight = weights[:, 0] * lmwt + weights[:, 1] * acwt
        lt_nll = apply_strategy(lat, lt_probs, strategy)
        nll = axl_weight + lt_nll * model_weight
        _, hyp = best_path_nloglike(lat, nll,
                                   final_word_id=final_word_id)
        hyp_line = tokenizer.decode([[arc[0] for arc in hyp]])[0]
        assert hyp_line[0] == '<s>' and hyp_line[-1] == '</s>', RuntimeError(f"{i} {hyp_line}")
        utt2hyp[utt] = hyp_line[1:-1]
    return utt2hyp


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate model")
    rescoring_task.RescoringTask.add_args(parser, add_def_opts=False, add_data_opts=False)
    LatsDataSet.add_args(parser)
    parser.add_argument('--no_progress_bar', action='store_true',
                        help="disable progress bar")
    parser.add_argument('--device', type=torch.device, default=torch.device('cpu'),
                        help="PyTorch device")
    parser.add_argument('--keep_tmp', action='store_true',
                        help="keep tmp hyp")
    parser.add_argument('--acwt', type=float, default=1, help="Acoustic weight")
    parser.add_argument("model",  type=str, help="Saved model")
    parser.add_argument("ref_fname", type=str, help="Reference text")

    args = parser.parse_args()
    setup_logger()
    tokenizer = WordTokenizer.build_from_args(args)
    dataset = LatsDataSet.build_from_args(args, tokenizer)
    models, model_args, task = checkpoint_utils.load_model_ensemble_and_task([args.model], arg_overrides=vars(args))
    model = models[-1].to(args.device)
    ref_fname = args.ref_fname
    wer = compute_model_wer(model=model,
                            dataset=dataset,
                            ref_fname=ref_fname,
                            device=args.device,
                            btz=args.infer_btz,
                            model_weight=args.model_weight,
                            progress_bar=(not args.no_progress_bar),
                            keep_tmp=args.keep_tmp,
                            dataloader_nj=args.dataloader_nj,
                            hyp_filter=args.hyp_filter,
                            acwt=args.acwt,
                            lmwt=args.lmwt,
                            strategy=args.strategy)

    print(f"Lattices {dataset.get_name()}, model {args.model}, model_weight {args.model_weight}.")
    print(wer)
