# Copyright 2021 STC-Innovation LTD (Author: Anton Mitrofanov)
import subprocess
import os
import time

import logging

logger = logging.getLogger(__name__)


def compute_wer(orig_fname, utt2hyp, tmp_dir='tmp', keep_tmp=False, hyp_filter='local/wer_hyp_filter', apply_hyp_filter=True):
    """ Compute wer using compute-wer kaldi util

    :param orig_fname: Original kaldi text file
    :param utt2hyp: dict[utt]= hypothesis
    :param tmp_dir: tmp dir for kaldi file with hypothesis
    :param keep_tmp: If True kaldi file with hypothesis
    :return: list WER report in kaldi format.
    Example ["%WER 11.33 [ 6162 / 54402, 895 ins, 478 del, 4789 sub ]",
             "%SER 72.70 [ 1965 / 2703 ]",
             "Scored 2703 sentences, 0 not present in hyp."]
    """
    os.makedirs(tmp_dir, exist_ok=True)
    out_lines = sorted([" ".join([utt_id, *hyp]) for utt_id, hyp in utt2hyp.items()])

    tmp_fname = os.path.join(tmp_dir, f"hyp_{time.time()}.txt")
    with open(tmp_fname, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_lines) + '\n')

    if apply_hyp_filter:
        hyp_pipe = f'ark:{hyp_filter} <{tmp_fname} |'
    else:
        hyp_pipe = f'ark:{tmp_fname}'

    s = subprocess.Popen(f'compute-wer ark:{orig_fname} "{hyp_pipe}"',
                         stdout=subprocess.PIPE, shell=True)
    s.wait()
    if not keep_tmp:
        os.remove(tmp_fname)
    out_s = s.stdout.read().decode('utf-8').strip()
    return out_s.split('\n')
