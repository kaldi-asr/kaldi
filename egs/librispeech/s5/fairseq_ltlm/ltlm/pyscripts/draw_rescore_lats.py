# Copyright 2021 STC-Innovation LTD (Author: Anton Mitrofanov)
import argparse
import graphviz
import logging
import torch
import torch.utils.data
import numpy as np
from tqdm import tqdm
import os

from fairseq import checkpoint_utils, data, options, tasks
import ltlm.eval
from ltlm.datasets import LatsOracleAlignDataSet, LatsDataSet

from ltlm.pyutils.logging_utils import setup_logger
from ltlm.models import LatticeTransformer
from ltlm.pyutils.lattice_utils import graphviz_lattice, norm_lt, ones_nochoices_arcs, arc_wer_map
from ltlm.pyutils.kaldi_utils import compute_wer
from ltlm.Tokenizer import WordTokenizer
from ltlm.tasks import rescoring_task
logger = logging.getLogger(__name__)


def draw_rescoring_lats(model, dataset, out_dir, utts,
                        norm=False, device='cpu', log=False, format='pdf', progress_bar=True):
    os.makedirs(out_dir, exist_ok=True)
    if model is not None:
        model = model.to(device)
        model.eval()
    iterator = utts
    if progress_bar:
        iterator = tqdm(iterator)

    with torch.no_grad():
        for i, (utt_id) in enumerate(iterator):
            sample = dataset[utt_id]
            batched_samples = dataset.collater([sample])
            x, weights = batched_samples['net_input']['src_tokens'], batched_samples['weights']
            bz, sl = x.size(0), x.size(1)
            logits, _ = model(x.to(device), apply_sigmoid=True)
            weight_t = logits.view(bz, sl).cpu()
            if norm:
                weight_t = norm_lt(x.squeeze(), weight_t.squeeze())
            if log:
                weight_t = weight_t.log()

            weights.append(weight_t)
            weights = [[round(w.item(), 4) for w in ws.squeeze()] for ws in weights]
            green = set(np.where((sample['target']) == 1)[0])
            # red_arcs = set(np.where((sample['target_mask'] == 1))[0]) # set(np.where((sample['targets_mask']) == 0)[0])
            # arcs, choice = arc_wer_map(sample['net_input']['src_tokens'], sample['ref'], dataset.tokenizer.get_eos_id())

            #weights.append(sample['wers'])
            # weights.append(sample['target_mask'])
            # blue = set(np.where(choice == 1)[0])

            dot = graphviz_lattice(x.squeeze(), dataset.tokenizer, *weights, utt_id=utt_id, green_arcs=green)
            dot.format = format
            f_name = os.path.join(out_dir, utt_id + '.gv')
            pdf_fname = dot.render(f_name)
            logger.info(f"Saved graph to {pdf_fname}")


def main():
    parser = argparse.ArgumentParser()
    # rescoring_task.RescoringTask.add_args(parser, add_def_opts=False, add_data_opts=False)
    WordTokenizer.add_args(parser)
    #LatsDataSet.add_args(parser)
    LatsOracleAlignDataSet.add_args(parser)
    parser.add_argument('--no_progress_bar', action='store_true',
                        help="disable progress bar")
    parser.add_argument('--device', type=torch.device, default=torch.device('cpu'),
                        help="PyTorch device")
    parser.add_argument('--format', type=str, default='pdf', help="Render file type. Default .pdf")
    parser.add_argument('--norm', action='store_true', default=False, help="Normalize lt probs")
    parser.add_argument("model", type=str, help="Saved model")
    parser.add_argument("out_dir", type=str, help='Out dir')
    parser.add_argument("utts", nargs="+", help="Utt_ids for draw")

    args = parser.parse_args()
    setup_logger()
    tokenizer = WordTokenizer.build_from_args(args)
    #dataset = LatsDataSet.build_from_args(args, tokenizer)
    dataset = LatsOracleAlignDataSet.build_from_args(args, tokenizer)

    models, model_args, task = checkpoint_utils.load_model_ensemble_and_task([args.model], arg_overrides=vars(args))
    model = models[-1].to(args.device)
    logger.info(f"Drawing {args.utts}")
    draw_rescoring_lats(model, dataset, args.out_dir, args.utts, norm=args.norm,
                        device=args.device, format=args.format, progress_bar=(not args.no_progress_bar))


if __name__ == "__main__":
    main()
