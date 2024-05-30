#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (author: Hossein Hadian)
# Apache 2.0


""" This script prepares the training and test data for SVHN.
"""
from __future__ import division

import argparse
import os
import sys
import scipy.io as sio
import numpy as np

parser = argparse.ArgumentParser(description="""Converts train/test data of
                                                SVHN (Street View House Numbers)
                                                dataset to Kaldi feature format""")
parser.add_argument('matlab_file',
                    help='path to SVHN matlab data file (cropped version)')
parser.add_argument('dir',
                    help='output dir')
parser.add_argument('--out-ark',
                    default='-', help='where to write output feature data')

args = parser.parse_args()

# SVHN cropped images dimensions:
C = 3  # num_channels
H = 32  # num_rows
W = 32  # num_cols

def load_svhn_data(matlab_file):
    matlab_data = sio.loadmat(matlab_file)
    data = matlab_data['X'].astype(float) / 255.0  # H*W*C*NUM_IMAGES
    labels = matlab_data['y']  # NUM_IMAGES*1
    return data, labels

def write_kaldi_matrix(file_handle, matrix, key):
    # matrix is a list of lists
    file_handle.write(key + "  [ ")
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

def zeropad(x, length):
  s = str(x)
  while len(s) < length:
    s = '0' + s
  return s

### main ###
if args.out_ark == '-':
  out_fh = sys.stdout  # output file handle to write the feats to
else:
  out_fh = open(args.out_ark, 'wb')

labels_file = os.path.join(args.dir, 'labels.txt')
labels_fh = open(labels_file, 'wb')

data, labels = load_svhn_data(args.matlab_file)
num_images = np.shape(data)[-1]

# permute and reshape from H x W x C x NUM_IMAGES to NUM_IMAGES x W x (H*C)
data = np.reshape(np.transpose(data, (3, 1, 0, 2)), (num_images, W, H * C))

for i in range(num_images):
    img_id = i + 1
    key = zeropad(img_id, 6)
    lbl = labels[i, 0]
    if lbl == 10:
        lbl = 0
    labels_fh.write("{} {}\n".format(key, lbl))
    img = data[i]
    write_kaldi_matrix(out_fh, img, key)
    img_id += 1

labels_fh.close()
out_fh.close()
