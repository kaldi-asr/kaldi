#!/usr/bin/env python

# Copyright 2017  Vimal Manohar
# Apache 2.0

"""
This script reads a wav.scp file from the input and perturbs the
volume of the recordings and writes to stdout the contents of
a new wav.scp file.
"""
from __future__ import print_function

import argparse
import re
import random
import sys

def get_args():
    parser = argparse.ArgumentParser(description="""
        This script reads a wav.scp file from the input and perturbs the
        volume of the recordings and writes to stdout the contents of
        a new wav.scp file.
        If --reco2vol is provided, then for each recording, the volume factor
        specified in that file is applied.
        Otherwise, a volume factor is chosen randomly from a uniform
        distribution between --scale-low and --scale-high.
        """)

    parser.add_argument("--scale-low", type=float, default=0.125,
                        help="Minimum volume scale to be applied.")
    parser.add_argument("--scale-high", type=float, default=2,
                        help="Maximum volume scale to tbe applid.")
    parser.add_argument("--reco2vol", type=str, default=None,
                        help="If supplied, it must be a file of the format "
                        "<reco-id> <volume-scale>, which specifies the "
                        "volume scale to be applied for each recording.")
    parser.add_argument("--write-reco2vol", type=str, default=None,
                        help="If provided, the volume scale used for each "
                        "recording will be written to this file")
    args = parser.parse_args()

    if args.reco2vol == "":
        args.reco2vol = None
    if args.write_reco2vol == "":
        args.write_reco2vol = None

    return args


def read_reco2vol(volumes_file):
    """Read volume scales for recordings.
    The format of volumes_file is <reco-id> <volume-scale>
    Returns a dictionary { reco-id : volume-scale }
    """
    volumes = {}
    with open(volumes_file) as volume_reader:
        for line in volume_reader.readlines():
            if len(line.strip()) == 0:
                continue

            parts = line.strip().split()
            if len(parts) != 2:
                raise RuntimeError("Unable to parse the line {0} in file {1}."
                                   "".format(line.strip(), volumes_file))
            volumes[parts[0]] = float(parts[1])
    return volumes


def run(args):
    random.seed(0)

    volumes = None
    if args.reco2vol is not None:
        volumes = read_reco2vol(args.reco2vol)

    if args.write_reco2vol is not None:
        volume_writer = open(args.write_reco2vol, 'w')

    for line in sys.stdin.readlines():
        if len(line.strip()) == 0:
            continue
        parts = line.strip().split()
        reco_id = parts[0]

        vol = random.uniform(args.scale_low, args.scale_high)
        if volumes is not None:
            if reco_id not in volumes:
                raise RuntimeError('Could not find volume for id {0} in '
                                   '{1}'.format(reco_id, args.reco2vol))
            vol = volumes[reco_id]

        # Handle three cases of rxfilenames appropriately;
        # 'input piped command', 'file offset' and 'filename'
        if line.strip()[-1] == '|':
            print ('{0} sox --vol {1} -t wav - -t wav - |'.format(
                line.strip(), vol))
        elif re.search(':[0-9]+$', line.strip()) is not None:
            print ('{id} wav-copy {wav} - | '
                   'sox --vol {vol} -t wav - -t wav - |'.format(
                       id=parts[0], wav=' '.join(parts[1:]), vol=vol))
        else:
            print ('{id} sox --vol {vol} -t wav {wav} -t wav - |'.format(
                id=parts[0], wav=' '.join(parts[1:]), vol=vol))

        if args.write_reco2vol is not None:
            volume_writer.write('{id} {vol}\n'.format(id=parts[0], vol=vol))


def main():
    args = get_args()
    run(args)


if __name__ == "__main__":
    main()
