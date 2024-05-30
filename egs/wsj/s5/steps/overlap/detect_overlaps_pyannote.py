#!/usr/local/env python
# Copyright 2021  Johns Hopkins University (Author: Desh Raj)
#
# Overlap detection using pretrained Pyannote models
import torch
import argparse
import glob
import os
import pathlib


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_scp", help="Path to wav.scp file")
    parser.add_argument("out_dir", help="Path to output dir")
    parser.add_argument("--model-name", type=str, help="Name of model (ovl_ami/ovl_dihard)")
    args = parser.parse_args()
    return args

def main(wav_scp, out_dir, model="ovl_dihard"):
    ovl_pipeline = torch.hub.load(
        'pyannote/pyannote-audio',
        model,
        pipeline=True,
        step=0.25,
        batch_size=128,
        device='cpu'
    )
    with open(wav_scp, 'r') as f:
        for line in f:
            file_id, wav_path = line.strip().split()
            ovl_out = ovl_pipeline({'audio': wav_path})
            with open(f'{out_dir}/{file_id}.rttm', 'w') as f_out:
                ovl_out.write_rttm(f_out)

if __name__=="__main__":
    args = read_args()
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    main(args.wav_scp, args.out_dir, model=args.model_name)