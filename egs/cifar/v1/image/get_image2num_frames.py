#!/usr/bin/env python3

# Copyright      2018  Hossein Hadian


""" This script computes the image lengths (with padding) in an image data dir.
    The output is written to 'image2num_frames' in the given data dir. This
    file is later used by image/get_allowed_lengths.py to find a set of allowed lengths
    for the data dir. The output format is similar to utt2num_frames

"""

import argparse
import os
import sys
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description="""Computes the image lengths (i.e. width) in an image data dir
                                                and writes them (by default) to image2num_frames.""")
parser.add_argument('dir', type=str,
                    help='Source data directory (containing images.scp)')
parser.add_argument('--out-ark', type=str, default=None,
                    help='Where to write the output image-to-num_frames info. '
                    'Default: "dir"/image2num_frames')
parser.add_argument('--feat-dim', type=int, default=40,
                    help='Size to scale the height of all images')
parser.add_argument('--padding', type=int, default=5,
                    help='Number of white pixels to pad on the left'
                    'and right side of the image.')
args = parser.parse_args()


def get_scaled_image_length(im):
    scale_size = args.feat_dim
    sx, sy = im.size
    scale = (1.0 * scale_size) / sy
    nx = int(scale * sx)
    return nx

### main ###
data_list_path = os.path.join(args.dir,'images.scp')

if not args.out_ark:
    args.out_ark = os.path.join(args.dir,'image2num_frames')
if args.out_ark == '-':
    out_fh = sys.stdout
else:
    out_fh = open(args.out_ark, 'w', encoding='latin-1')

with open(data_list_path) as f:
    for line in f:
        line = line.strip()
        line_vect = line.split(' ')
        image_id = line_vect[0]
        image_path = line_vect[1]
        im = Image.open(image_path)
        im_len = get_scaled_image_length(im) + (args.padding * 2)
        print('{} {}'.format(image_id, im_len), file=out_fh)

out_fh.close()
