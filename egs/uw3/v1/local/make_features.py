#!/usr/bin/env python3

# Copyright      2017  Chun Chieh Chang

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
from scipy import ndimage

from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL)

parser = argparse.ArgumentParser(description="""Converts images (in 'dir'/images.scp) to features and
                                                writes them to standard output in text format.""")
parser.add_argument('dir', help='data directory (should contain images.scp)')
parser.add_argument('--out-ark', default='-', help='where to write the output feature file.')
parser.add_argument('--feat-dim', type=int, default=40,
                    help='size to scale the height of all images (i.e. the dimension of the resulting features)')
parser.add_argument('--pad', type=bool, default=False, help='pad the left and right of the images with 10 white pixels.')

args = parser.parse_args()


def write_kaldi_matrix(file_handle, matrix, key):
    file_handle.write(key + " [ ")
    num_rows = len(matrix)
    if num_rows == 0:
        raise Exception("Matrix is empty")
    num_cols = len(matrix[0])
    for row_index in range(len(matrix)):
        if num_cols != len(matrix[row_index]):
            raise Exception("All the rows of a matrix are expected to "
                            "have the same length")
        file_handle.write(" ".join([str(x) for x in matrix[row_index]]))
        if row_index != num_rows - 1:
            file_handle.write("\n")
    file_handle.write(" ]\n")

def get_scaled_image(im):
    scale_size = args.feat_dim
    sx = im.shape[1]
    sy = im.shape[0]
    # Some Images are rotated
    if sy > sx:
        im = np.rot90(im, k = -1)
        sx = im.shape[1]
        sy = im.shape[0]

    scale = (1.0 * scale_size) / sy
    nx = int(scale_size)
    ny = int(scale * sx)
    im = misc.imresize(im, (nx, ny))

    noise = np.random.normal(2, 1,(nx, ny))
    im = im - noise

    return im

### main ###
data_list_path = os.path.join(args.dir,'images.scp')

if args.out_ark == '-':
    out_fh = sys.stdout
else:
    out_fh = open(args.out_ark,'wb')

with open(data_list_path) as f:
    for line in f:
        line = line.strip()
        line_vect = line.split(' ')
        image_id = line_vect[0]
        image_path = line_vect[1]

        im = misc.imread(image_path, flatten = True)
        im_scale = get_scaled_image(im)

        if args.pad:
            pad = np.ones((args.feat_dim, 10)) * 255
            im_data = np.hstack((pad, im_scale, pad))
        else:
            im_data = im_scale

        data = np.transpose(im_data, (1, 0))
        data = np.divide(data, 255.0)
        write_kaldi_matrix(out_fh, data, image_id)
