# Copyright 2021 STC-Innovation LTD (Author: Anton Mitrofanov)
import argparse
import json
import os
from glob import glob
import sys
import logging
import numpy as np

from ltlm.pyutils.logging_utils import setup_logger

logger = logging.getLogger(__name__)


def get_file2size(dirs):
    # return dict file -> size_MB
    file2size = {}
    for d in dirs:
        for f in glob(os.path.join(d, 'lat.*.dump')):
            size = os.path.getsize(f)
            file2size[f] = size/1024/1024
    return file2size


def balance_decode_gen_egs(decode_dirs, decode_suffs, gen_dirs, gen_suffs, epoch_max_MB=4000, reuse_data=True):
    # Generate list [epoch] -> [(egs, suf), (egs,suff), ...]
    d2suff = {d: s for d, s in (*zip(decode_dirs, decode_suffs), *zip(gen_dirs, gen_suffs))}
    decode_f2s = get_file2size(decode_dirs)
    gen_f2s = get_file2size(gen_dirs)
    if len(decode_f2s) > 0:
        logger.info(f"Found {len(decode_f2s)} egs with decoded lats. "
                    f"Average size is {round(sum(decode_f2s.values())/len(decode_f2s), 3)}MB.")
    else:
        logger.info(f"Decoded lats is empty")
    if len(gen_f2s) > 0:
        logger.info(f"Found {len(gen_f2s)} egs with generated from text lats. "
                    f"Average size is {round(sum(gen_f2s.values()) / len(gen_f2s), 3)}MB.")
    else:
        logger.info(f"Generated lats is empty")

    sorted_dec_f = [k for k, _ in sorted(decode_f2s.items(), key=lambda x: x[1], reverse=True)]
    sorted_gen_f = [k for k, _ in sorted(gen_f2s.items(), key=lambda x: x[1], reverse=True)]
    #sorted_dec_id = np.arange(len(sorted_dec_f), dtype=int)
    dec_not_used = np.ones(len(sorted_dec_f), dtype=bool)
    reused_dec = False
    #sorted_gen_id = np.arange(len(sorted_gen_f), dtype=int)
    gen_not_used = np.ones(len(sorted_gen_f), dtype=bool)
    reused_gen = False

    curr_dec_sz = 0
    curr_gen_sz = 0
    curr_files = []
    out = []
    while (not reused_dec and dec_not_used.any()) or \
          (not reused_gen and gen_not_used.any()):
        insert = False
        if curr_dec_sz <= curr_gen_sz or not gen_not_used.any():
            # get decode archive
            not_used_idx = np.where(dec_not_used)[0]
            #not_used = sorted_dec[not dec_used]
            for i in not_used_idx:
                f = sorted_dec_f[i]
                sz = decode_f2s[f]
                if curr_dec_sz + sz + curr_gen_sz <= epoch_max_MB:
                    d = os.path.dirname(f)
                    suff = d2suff[d]
                    curr_files.append((f, suff))
                    insert = True
                    dec_not_used[i] = False
                    curr_dec_sz += sz
                    break
        if not insert:
            # get gen archive:
            not_used_idx = np.where(gen_not_used)[0]
            #not_used = sorted_gen[not gen_used]
            for i in not_used_idx:
                f = sorted_gen_f[i]
                sz = gen_f2s[f]
                if curr_dec_sz + curr_gen_sz + sz <= epoch_max_MB:
                    d = os.path.dirname(f)
                    suff = d2suff[d]
                    curr_files.append((f, suff))
                    insert = True
                    gen_not_used[i] = False
                    curr_gen_sz += sz
                    break

        if not insert:
            # Cannot insert file. Size limit
            if len(curr_files) > 0:
                logger.info(f"For epoch {len(out)} balance decode/generate is {round(curr_dec_sz/curr_gen_sz,6)}.")
                out.append(curr_files)
                curr_files = []
                curr_dec_sz = 0
                curr_gen_sz = 0
                if reuse_data and (dec_not_used.any() or gen_not_used.any()):
                    if not dec_not_used.any():
                        reused_dec = True
                        dec_not_used.fill(True)
                    elif not gen_not_used.any():
                        reused_gen = True
                        gen_not_used.fill(True)
            elif dec_not_used.any() or gen_not_used.any():
                # Something gone wrong
                bad_dec = {f: decode_f2s[f] for f in map(sorted_dec_f.__getitem__, np.where(dec_not_used)[0])}
                bad_dec_str = '\n'.join((f'{f} {s}' for f, s in bad_dec.items()))
                bad_gen = {f: gen_f2s[f] for f in map(sorted_gen_f.__getitem__, np.where(gen_not_used)[0])}
                bad_gen_str = '\n'.join((f'{f} {s}' for f, s in bad_gen.items()))
                logger.warning(f"This files is too big.\n DECODED LATS:\n"
                               f"{bad_dec_str}.\n "
                               f"GENERATE:\n {bad_gen_str}.")
                logger.warning("Careful!!! These files are inserted into training as 1 dump per data-epoch")
                out.extend([[(f, d2suff[os.path.dirname(f)])] for f in (*bad_dec.keys(), *bad_gen.keys())])
                break

    if len(curr_files) > 0:
        if curr_gen_sz > 0 :
            logger.info(f"For epoch {len(out)} balance decode/generate is {round(curr_dec_sz/curr_gen_sz,6)}.")
        else:
            logger.info(f"For epoch {len(out)} only decode lattices")
        out.append(curr_files)
    return out



def main():
    setup_logger(stream=sys.stderr)
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch_max_MB', type=float, default=4000,
                        help="Maximum egs size per one epoch. in MB")
    parser.add_argument('--reuse_data', action='store_true', 
                        help="Reuse data. If decoded or generaten less than other split")
    parser.add_argument('--train_decoded', nargs="*", required=False, default=[],
                        help="Decoded train egs dirs.  format egs[,suff] ")
    parser.add_argument('--train_generated', nargs="*", required=False, default=[],
                        help="Generated from text egs dirs.  format egs[,suff] ")
    parser.add_argument('--valid', nargs="+", required=True, help="Valid. format 'egs,ref egs,ref2'")
    parser.add_argument('--test', nargs='*', default=None, help="Test. format like valid")
    parser.add_argument("--out", type=str, required=True, help="Path to result data_config.json file")
    args = parser.parse_args()
    if len(args.train_decoded) == len(args.train_generated) == 0 :
        parser.print_help()
        raise RuntimeError(f"--train_decoded or --train_generated required!")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    data = {"valid": []}
    for l_r in args.valid:
        l_r_sep = l_r.split(',')
        assert len(l_r_sep) == 2, RuntimeError("Wrong valid format. Must be lat,ref lat2,ref ...")
        l, r = l_r_sep
        data['valid'].append({'lats': l, 'ref': r})
    if args.test:
        data['test'] = []
        for l_r in args.test:
            l_r_sep = l_r.split(',')
            assert len(l_r_sep) == 2, RuntimeError("Wrong test format. Must be lat,ref lat2,ref ...")
            l, r = l_r_sep
            data['test'].append({'lats': l, 'ref': r})

    train_dec_dirs = []
    train_dec_suffs = []
    for l_s in args.train_decoded:
        l_s_sep = l_s.split(',')
        l = l_s_sep[0]
        s = l_s_sep[1] if len(l_s_sep) > 1 else ''
        train_dec_dirs.append(l)
        train_dec_suffs.append(s)
    train_gen_dirs = []
    train_gen_suffs = []
    for l_s in args.train_generated:
        l_s_sep = l_s.split(',')
        l = l_s_sep[0]
        s = l_s_sep[1] if len(l_s_sep) > 1 else ''
        train_gen_dirs.append(l)
        train_gen_suffs.append(s)

    epoch2data = balance_decode_gen_egs(train_dec_dirs, train_dec_suffs, train_gen_dirs, train_gen_suffs,
                                        epoch_max_MB=args.epoch_max_MB)
    data['train'] = [{"epoch": e, "lats": l, "utt_suff": s} for e, epoch_data in enumerate(epoch2data) for l, s in epoch_data]

    with open(args.out, 'w') as out:
        json.dump(data, out, indent=4)
    logger.info("Done")

if __name__ == "__main__":
    main()
