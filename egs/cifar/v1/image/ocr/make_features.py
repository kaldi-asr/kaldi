#!/usr/bin/env python3

# Copyright      2017  Chun Chieh Chang
#                2017  Ashish Arora
#                2017  Yiwen Shao
#                2018  Hossein Hadian

""" This script converts images to Kaldi-format feature matrices. The input to
    this script is the path to a data directory, e.g. "data/train". This script
    reads the images listed in images.scp and writes them to standard output
    (by default) as Kaldi-formatted matrices (in text form). It also scales the
    images so they have the same height (via --feat-dim). It can optionally pad
    the images (on left/right sides) with white pixels. It by default performs 
    augmentation, (directly scaling down and scaling up). It will double the 
    data but we can turn augmentation off (via --no-augment).
    If an 'image2num_frames' file is found in the data dir, it will be used
    to enforce the images to have the specified length in that file by padding
    white pixels (the --padding option will be ignored in this case). This relates
    to end2end chain training.
    eg. local/make_features.py data/train --feat-dim 40
"""
import random
import argparse
import os
import sys
import numpy as np
from scipy import misc
import math
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE, SIG_DFL)

parser = argparse.ArgumentParser(description="""Converts images (in 'dir'/images.scp) to features and
                                                writes them to standard output in text format.""")
parser.add_argument('images_scp_path', type=str,
                    help='Path of images.scp file')
parser.add_argument('--allowed_len_file_path', type=str, default=None,
                    help='If supplied, each images will be padded to reach the '
                    'target length (this overrides --padding).')
parser.add_argument('--out-ark', type=str, default='-',
                    help='Where to write the output feature file')
parser.add_argument('--feat-dim', type=int, default=40,
                    help='Size to scale the height of all images')
parser.add_argument('--padding', type=int, default=5,
                    help='Number of white pixels to pad on the left'
                    'and right side of the image.')
parser.add_argument("--augment", action="store_true",
                   help="whether or not to do image augmentation")
parser.add_argument("--flip", action="store_true",
                   help="whether or not to flip the image")
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
        file_handle.write(" ".join(map(lambda x: str(x), matrix[row_index])))
        if row_index != num_rows - 1:
            file_handle.write("\n")
    file_handle.write(" ]\n")


def horizontal_pad(im, allowed_lengths = None):
    if allowed_lengths is None:
        left_padding = right_padding = args.padding
    else:  # Find an allowed length for the image
        imlen = im.shape[1] # width
        allowed_len = 0
        for l in allowed_lengths:
            if l > imlen:
                allowed_len = l
                break
        if allowed_len == 0:
            #  No allowed length was found for the image (the image is too long)
            return None
        padding = allowed_len - imlen
        left_padding = int(padding // 2)
        right_padding = padding - left_padding
    dim_y = im.shape[0] # height
    im_pad = np.concatenate((255 * np.ones((dim_y, left_padding),
                                           dtype=int), im), axis=1)
    im_pad1 = np.concatenate((im_pad, 255 * np.ones((dim_y, right_padding),
                                                    dtype=int)), axis=1)
    return im_pad1

def get_scaled_image_aug(im, mode='normal'):
    scale_size = args.feat_dim
    sx = im.shape[1]
    sy = im.shape[0]
    scale = (1.0 * scale_size) / sy
    nx = int(scale_size)
    ny = int(scale * sx) 
    scale_size = random.randint(10, 30)
    scale = (1.0 * scale_size) / sy
    down_nx = int(scale_size)
    down_ny = int(scale * sx)
    if mode == 'normal':
        im = misc.imresize(im, (nx, ny))
        return im
    else:
        im_scaled_down = misc.imresize(im, (down_nx, down_ny))
        im_scaled_up = misc.imresize(im_scaled_down, (nx, ny))
        return im_scaled_up
    return im


### main ###
random.seed(1)
data_list_path = args.images_scp_path
if args.out_ark == '-':
    out_fh = sys.stdout
else:
    out_fh = open(args.out_ark,'w')

allowed_lengths = None
allowed_len_handle = args.allowed_len_file_path
if os.path.isfile(allowed_len_handle):
    print("Found 'allowed_lengths.txt' file...", file=sys.stderr)
    allowed_lengths = []
    with open(allowed_len_handle) as f:
        for line in f:
            allowed_lengths.append(int(line.strip()))
    print("Read {} allowed lengths and will apply them to the "
          "features.".format(len(allowed_lengths)), file=sys.stderr)

num_fail = 0
num_ok = 0
aug_setting = ['normal', 'scaled']
with open(data_list_path) as f:
    for line in f:
        line = line.strip()
        line_vect = line.split(' ')
        image_id = line_vect[0]
        image_path = line_vect[1]
        im = misc.imread(image_path)
        if args.flip:
            im = np.fliplr(im)
        if args.augment:
            for i in range(2):
                image_aug_id = image_id + '_aug' + str(i + 1)
                im_aug = get_scaled_image_aug(im, aug_setting[i])
                im_horizontal_padded = horizontal_pad(im_aug, allowed_lengths)
                if im_horizontal_padded is None:
                    num_fail += 1
                    continue
                data = np.transpose(im_horizontal_padded, (1, 0))
                data = np.divide(data, 255.0)
                num_ok += 1
                write_kaldi_matrix(out_fh, data, image_aug_id)
        else:
            image_aug_id = image_id + '_aug'
            im_aug = get_scaled_image_aug(im, aug_setting[0])
            im_horizontal_padded = horizontal_pad(im_aug, allowed_lengths)
            if im_horizontal_padded is None:
                num_fail += 1
                continue
            data = np.transpose(im_horizontal_padded, (1, 0))
            data = np.divide(data, 255.0)
            num_ok += 1
            write_kaldi_matrix(out_fh, data, image_aug_id)

print('Generated features for {} images. Failed for {} (image too '
      'long).'.format(num_ok, num_fail), file=sys.stderr)
