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
    the images (on left/right sides) with white pixels.
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
import scipy.io as sio
import numpy as np
from scipy import misc
from scipy.ndimage.interpolation import affine_transform
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
parser.add_argument('--fliplr', type=lambda x: (str(x).lower()=='true'), default=False,
                   help="Flip the image left-right for right to left languages")
parser.add_argument("--augment", type=lambda x: (str(x).lower()=='true'), default=False,
                   help="performs image augmentation")
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

def contrast_normalization(im, low_pct, high_pct):
    element_number = im.size
    rows = im.shape[0]
    cols = im.shape[1]
    im_contrast = np.zeros(shape=im.shape)
    low_index = int(low_pct * element_number)
    high_index = int(high_pct * element_number)
    sorted_im = np.sort(im, axis=None)
    low_thred = sorted_im[low_index]
    high_thred = sorted_im[high_index]
    for i in range(rows):
        for j in range(cols):
            if im[i, j] > high_thred:
                im_contrast[i, j] = 255  # lightest to white
            elif im[i, j] < low_thred:
                im_contrast[i, j] = 0  # darkest to black
            else:
                # linear normalization
                im_contrast[i, j] = (im[i, j] - low_thred) * \
                    255 / (high_thred - low_thred)
    return im_contrast


def geometric_moment(frame, p, q):
    m = 0
    for i in range(frame.shape[1]):
        for j in range(frame.shape[0]):
            m += (i ** p) * (j ** q) * frame[i][i]
    return m


def central_moment(frame, p, q):
    u = 0
    x_bar = geometric_moment(frame, 1, 0) / \
        geometric_moment(frame, 0, 0)  # m10/m00
    y_bar = geometric_moment(frame, 0, 1) / \
        geometric_moment(frame, 0, 0)  # m01/m00
    for i in range(frame.shape[1]):
        for j in range(frame.shape[0]):
            u += ((i - x_bar)**p) * ((j - y_bar)**q) * frame[i][j]
    return u


def height_normalization(frame, w, h):
    frame_normalized = np.zeros(shape=(h, w))
    alpha = 4
    x_bar = geometric_moment(frame, 1, 0) / \
        geometric_moment(frame, 0, 0)  # m10/m00
    y_bar = geometric_moment(frame, 0, 1) / \
        geometric_moment(frame, 0, 0)  # m01/m00
    sigma_x = (alpha * ((central_moment(frame, 2, 0) /
                         geometric_moment(frame, 0, 0)) ** .5))  # alpha * sqrt(u20/m00)
    sigma_y = (alpha * ((central_moment(frame, 0, 2) /
                         geometric_moment(frame, 0, 0)) ** .5))  # alpha * sqrt(u02/m00)
    for x in range(w):
        for y in range(h):
            i = int((x / w - 0.5) * sigma_x + x_bar)
            j = int((y / h - 0.5) * sigma_y + y_bar)
            frame_normalized[x][y] = frame[i][j]
    return frame_normalized


def find_slant_project(im):
    rows = im.shape[0]
    cols = im.shape[1]
    std_max = 0
    alpha_max = 0
    col_disp = np.zeros(90, int)
    proj = np.zeros(shape=(90, cols + 2 * rows), dtype=int)
    for r in range(rows):
        for alpha in range(-45, 45, 1):
            col_disp[alpha] = int(r * math.tan(alpha / 180.0 * math.pi))
        for c in range(cols):
            if im[r, c] < 100:
                for alpha in range(-45, 45, 1):
                    proj[alpha + 45, c + col_disp[alpha] + rows] += 1
    for alpha in range(-45, 45, 1):
        proj_histogram, bin_array = np.histogram(proj[alpha + 45, :], bins=10)
        proj_std = np.std(proj_histogram)
        if proj_std > std_max:
            std_max = proj_std
            alpha_max = alpha
    proj_std = np.std(proj, axis=1)
    return -alpha_max


def horizontal_shear(im, degree):
    rad = degree / 180.0 * math.pi
    padding_x = int(abs(np.tan(rad)) * im.shape[0])
    padding_y = im.shape[0]
    if rad > 0:
        im_pad = np.concatenate(
            (255 * np.ones((padding_y, padding_x), dtype=int), im), axis=1)
    elif rad < 0:
        im_pad = np.concatenate(
            (im, 255 * np.ones((padding_y, padding_x), dtype=int)), axis=1)
    else:
        im_pad = im
    shear_matrix = np.array([[1, 0],
                             [np.tan(rad), 1]])
    sheared_im = affine_transform(im_pad, shear_matrix, cval=255.0)
    return sheared_im


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
        if args.fliplr:
            im = np.fliplr(im)
        if args.augment:
            im_aug = get_scaled_image_aug(im, aug_setting[0])
            im_contrast = contrast_normalization(im_aug, 0.05, 0.2)
            slant_degree = find_slant_project(im_contrast)
            im_sheared = horizontal_shear(im_contrast, slant_degree)
            im_aug = im_sheared
        else:
            im_aug = get_scaled_image_aug(im, aug_setting[0])
        im_horizontal_padded = horizontal_pad(im_aug, allowed_lengths)
        if im_horizontal_padded is None:
            num_fail += 1
            continue
        data = np.transpose(im_horizontal_padded, (1, 0))
        data = np.divide(data, 255.0)
        num_ok += 1
        write_kaldi_matrix(out_fh, data, image_id)

print('Generated features for {} images. Failed for {} (image too '
      'long).'.format(num_ok, num_fail), file=sys.stderr)
