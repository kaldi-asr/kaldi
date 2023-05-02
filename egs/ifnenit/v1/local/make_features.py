#!/usr/bin/env python3


""" This script converts images to Kaldi-format feature matrices. The input to
    this script is the path to a data directory, e.g. "data/train". This script
    reads the images listed in images.scp and writes them to standard output
    (by default) as Kaldi-formatted matrices (in text form). It also scales the
    images so they have the same height (via --feat-dim). It can optionally pad
    the images (on left/right sides) with white pixels.
    
    eg. local/make_features.py data/train --feat-dim 40
"""
from __future__ import division

import argparse
import os
import sys
import scipy.io as sio
import numpy as np
from scipy import misc
import math

from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL)

parser = argparse.ArgumentParser(description="""Generates and saves the feature vectors""")
parser.add_argument('dir', help='directory of images.scp and is also output directory')
parser.add_argument('--out-ark', default='-', help='where to write the output feature file')
parser.add_argument('--feat-dim', type=int, default=40, help='size to scale the height of all images')
parser.add_argument('--padding', type=int, default=5, help='size to scale the height of all images')
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
    scale = (1.0 * scale_size)/ sy
    nx = int(scale_size)
    ny = int(scale * sx)
    im = misc.imresize(im, (nx, ny))
    padding_x = 0
    padding_y = 0
    for i in range(0,30):
        im_x = im.shape[1]
        im_y = im.shape[0]
        if im_x >= (28 + (20*i)) and im_x <= (28 + (20*(i+1))):
           padding_x = (30 + (20*(i+1))) - im_x
           padding_y = im_y
        else:
           continue
    im_pad = np.concatenate((255 * np.ones((padding_y, math.ceil(1.0 * padding_x / 2)) , dtype=int), im), axis=1)
    im_pad1 = np.concatenate((im_pad,255 * np.ones((padding_y, int(1.0 * padding_x / 2)), dtype=int)), axis=1)
    return im_pad1

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
        im = misc.imread(image_path)
        im_scale = get_scaled_image(im)
        im_scale_inversed = np.fliplr(im_scale)
        data = np.transpose(im_scale_inversed, (1, 0))
        data = np.divide(data, 255.0)
        write_kaldi_matrix(out_fh, data, image_id)
