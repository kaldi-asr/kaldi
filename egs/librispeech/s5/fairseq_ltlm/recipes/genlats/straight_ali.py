# Copyright 2021 STC-Innovation LTD (Author: Anton Mitrofanov)
import numpy as np
import kaldi_io
import argparse
from collections import defaultdict
import logging
import os
import tqdm
import sys

from genlats.generate_fake_am_model import AliStretchModel
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers.clear()

# default output
c_handler = logging.StreamHandler(sys.stderr)
c_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    AliStretchModel.add_args(parser)
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument("ali_rspecifier",  help="align RSpecifier")
    parser.add_argument('ali_wspecifier', help='WSpecifier for straighted ali')

    args = parser.parse_args()
    logger.info(vars(args))
    if args.ali_wspecifier == 'ark:-':
        args.ali_wspecifier = sys.stdout.buffer
    if args.ali_rspecifier == 'ark:-':
        args.ali_rspecifier = sys.stdin.buffer
    # logger.info(f"Random seed is {args.seed}")
    np.random.seed(args.seed)

    model = AliStretchModel.build_from_disk(args)

    with kaldi_io.open_or_fd(args.ali_wspecifier, mode='wb') as f_out:
        i = 0
        for utt, ali in kaldi_io.read_vec_int_ark(args.ali_rspecifier):
            i += 1
            logger.debug(f'Process {utt}')
            s_ali = model(ali)
            logger.debug(f'Generated {s_ali.shape} ali from {ali.shape}')
            kaldi_io.write_vec_int(f_out, s_ali, key=utt)
        logger.info(f"Done. Processed {i} utterance")
