#!/usr/bin/env python3

# Copyright      2017  Chun Chieh Chang
#                2017  Ashish Arora

""" This script converts images to Kaldi-format feature matrices. The input to
    this script is the path to a data directory, e.g. "data/train". This script
    reads the images listed in images.scp and writes them to standard output
    (by default) as Kaldi-formatted matrices (in text form). It also scales the
    images so they have the same height (via --feat-dim). It can optionally pad
    the images (on left/right sides) with white pixels.

    eg. local/make_features.py data/train --feat-dim 40
"""

import argparse
import os
import sys
import numpy as np
from scipy import misc

parser = argparse.ArgumentParser(description="""Converts images (in 'dir'/images.scp) to features and
                                                writes them to standard output in text format.""")
parser.add_argument('dir', type=str,
                    help='Source data directory (containing images.scp)')
parser.add_argument('--out-ark', type=str, default=None,
                    help='Where to write the output image-to-num_frames info.')
parser.add_argument('--feat-dim', type=int, default=40,
                    help='Size to scale the height of all images')
parser.add_argument('--padding', type=int, default=5,
                    help='Number of white pixels to pad on the left'
                    'and right side of the image.')
args = parser.parse_args()


def get_scaled_image_length(im):
    scale_size = args.feat_dim
    sx = im.shape[1]
    sy = im.shape[0]
    scale = (1.0 * scale_size) / sy
    nx = int(scale * sx)
    return nx

### main ###
data_list_path = os.path.join(args.dir,'images.scp')

if not args.out_ark:
    args.out_ark = os.path.join(args.dir,'image2num_frames.txt')
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
        im = misc.imread(image_path)
        im_len = get_scaled_image_length(im) + (args.padding * 2)
        print('{} {}'.format(image_id, im_len), file=out_fh)

out_fh.close()
