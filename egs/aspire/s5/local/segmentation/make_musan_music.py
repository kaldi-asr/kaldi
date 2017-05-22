#! /usr/bin/env python

# Copyright 2017  Vimal Manohar
# Apache 2.0.

"""This script prepares MUSAN music corpus for perturbing data directory."""

from __future__ import print_function
import argparse
import os


def _get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--use-vocals", type=str, default="false",
                        choices=["true", "false"],
                        help="If true, also add music with vocals in the "
                        "output music-set-parameters")
    parser.add_argument("root_dir", type=str,
                        help="Root directory of MUSAN corpus")
    parser.add_argument("music_list", type=argparse.FileType('w'),
                        help="Convert music list into noise-set-paramters "
                        "for steps/data/reverberate_data_dir.py")

    args = parser.parse_args()

    args.use_vocals = True if args.use_vocals == "true" else False
    return args


def read_vocals(annotations):
    vocals = {}
    for line in open(annotations):
        parts = line.strip().split()
        if parts[2] == "Y":
            vocals[parts[0]] = True
    return vocals


def write_music(utt, file_path, music_list):
    """Write music file to list"""
    print ('{utt} {file_path}'.format(
        utt=utt, file_path=file_path), file=music_list)


def prepare_music_set(root_dir, use_vocals, music_list):
    """The main function that goes through the music part of the MUSAN corpus
    and writes out the files to a table indexed by the recording-id."""
    vocals = {}
    music_dir = os.path.join(root_dir, "music")
    num_done = 0
    for root, dirs, files in os.walk(music_dir):
        if os.path.exists(os.path.join(root, "ANNOTATIONS")):
            vocals = read_vocals(os.path.join(root, "ANNOTATIONS"))

        for f in files:
            file_path = os.path.join(root, f)
            if f.endswith(".wav"):
                utt = str(f).replace(".wav", "")
                if not use_vocals and utt in vocals:
                    continue
                num_done += 1
                write_music(utt, file_path, music_list)
    if num_done == 0:
        raise RuntimeError("Failed to get any music files")
    music_list.close()


def main():
    args = _get_args()

    try:
        prepare_music_set(args.root_dir, args.use_vocals,
                          args.music_list)
    finally:
        args.music_list.close()


if __name__ == '__main__':
    main()
