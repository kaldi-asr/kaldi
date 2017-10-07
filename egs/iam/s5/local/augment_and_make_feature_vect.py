#!/usr/bin/env python
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

parser = argparse.ArgumentParser(
    description="""Generates and saves the feature vectors""")
parser.add_argument(
    'dir', type=str, help='directory of images.scp and is also output directory')
parser.add_argument('--seg', type=str, default='1',
                    help='JOB number of images.JOB.scp if run in parallel mode')
parser.add_argument('--out-ark', type=str, default='-',
                    help='where to write the output feature file')
parser.add_argument('--scale-size', type=int, default=40,
                    help='size to scale the height of all images')
parser.add_argument('--padding', type=int, default=5,
                    help='size to scale the height of all images')
parser.add_argument('--vertical-shift', type=int, default=10,
                    help='total number of padding pixel per column')
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


def get_scaled_image(im):
    scale_size = args.scale_size
    sx = im.shape[1]  # width
    sy = im.shape[0]  # height
    scale = (1.0 * scale_size) / sy
    nx = int(scale_size)
    ny = int(scale * sx)
    im = misc.imresize(im, (nx, ny))
    padding_x = max(5, int((args.padding / 100) * im.shape[1]))
    padding_y = im.shape[0]
    im_pad = np.concatenate(
        (255 * np.ones((padding_y, padding_x), dtype=int), im), axis=1)
    im_pad1 = np.concatenate(
        (im_pad, 255 * np.ones((padding_y, padding_x), dtype=int)), axis=1)
    return im_pad1


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


def find_slant(im):
    rows = im.shape[0]
    cols = im.shape[1]
    sum_max = 0
    slant_degree = 0
    for shear_degree in range(-45, 45, 5):
        sum = 0
        shear_rad = shear_degree / 360.0 * 2 * math.pi
        shear_matrix = np.array([[1, 0],
                                 [np.tan(shear_rad), 1]])
        sheared_im = affine_transform(im, shear_matrix, cval=255.0)
        for j in range(cols):
            foreground = (sheared_im[:, j] < 100)
            number = np.sum(foreground)
            # print(number)
            if number != 0:
                start_point = -1
                end_point = -1
                start_point = 0
                for i in range(rows):
                    if foreground[i] == 1:
                        start_point = i
                        break
                for i in range(rows - 1, -1, -1):
                    if foreground[i] == 1:
                        end_point = i
                        break
                length = end_point - start_point + 1
                #print(number, length)
                if length == number:
                    sum = sum + number * number
        #print(shear_degree, sum)
        if sum > sum_max:
            sum_max = sum
            slant_degree = shear_degree
    return slant_degree


def deslant(im, shear):
    padding_x = int(abs(np.tan(shear)) * im.shape[0])
    padding_y = im.shape[0]
    if shear > 0:
        im_pad = np.concatenate(
            (255 * np.ones((padding_y, padding_x), dtype=int), im), axis=1)
    else:
        im_pad = np.concatenate(
            (im, 255 * np.ones((padding_y, padding_x), dtype=int)), axis=1)

    shear_matrix = np.array([[1, 0],
                             [np.tan(shear), 1]])
    # sheared_im = affine_transform(image, shear_matrix, output_shape=(
    # im.shape[0], im.shape[1] + abs(int(im.shape[0] * np.tan(shear)))), cval=128.0)
    sheared_im = affine_transform(im_pad, shear_matrix, cval=255.0)
    return sheared_im


def vertical_shift(im, mode='mid'):
    total = args.vertical_shift
    if mode == 'mid':
        top = total / 2
        bottom = total - top
    elif mode == 'top':  # more padding on top
        top = random.randint(total / 2, total)
        bottom = total - top
    elif mode == 'bottom':  # more padding on bottom
        top = random.randint(0, total / 2)
        bottom = total - top
    width = im.shape[1]
    im_pad = np.concatenate(
        (255 * np.ones((top, width), dtype=int), im), axis=0)
    im_pad = np.concatenate(
        (im_pad, 255 * np.ones((bottom, width), dtype=int)), axis=0)
    return im_pad


def image_augment(im, out_fh, image_id):
    random.seed(1)
    shift_setting = ['mid', 'top', 'bottom']
    image_shift_id = []
    for i in range(3):
        image_shift_id.append(image_id + '_shift' + str(i + 1))
        im_shift = vertical_shift(im, shift_setting[i])
        im_scaled = get_scaled_image(im_shift)
        data = np.transpose(im_scaled, (1, 0))
        data = np.divide(data, 255.0)
        new_scp_list.append(image_id + '_shift' + str(i + 1))
        write_kaldi_matrix(out_fh, data, image_shift_id[i])


# main #
new_scp_list = list()
text_file = os.path.join(args.dir, 'backup', 'text.txt')
text_dict = dict()  # stores imageID and text

with open(text_file) as text_fh:
    for uttID_text in text_fh:
        uttID_text = uttID_text.strip()
        uttID_text_vect = uttID_text.split(" ")
        uttID = uttID_text_vect[0]
        imageID = uttID.split("_")[1]
        text_vect = uttID_text_vect[1:]
        text = " ".join(text_vect)
        text_dict[imageID] = text

utt2spk_file = os.path.join(args.dir, 'backup', 'utt2spk')
uttID_spk_dict = dict()  # stores imageID and speaker

with open(utt2spk_file) as utt2spk_fh:
    for uttID_spk in utt2spk_fh:
        uttID_spk = uttID_spk.strip()
        uttID_spk_vect = uttID_spk.split(" ")
        uttID = uttID_spk_vect[0]
        imageID = uttID.split("_")[1]
        spk = uttID_spk_vect[1]
        uttID_spk_dict[imageID] = spk

image_file = os.path.join(args.dir, 'backup', 'images.scp')
uttID_path_dict = dict()  # stores imageID and image path

with open(image_file) as image_fh:
    for uttID_path in image_fh:
        uttID_path = uttID_path.strip()
        uttID_path_vect = uttID_path.split(" ")
        uttID = uttID_path_vect[0]
        imageID = uttID.split("_")[1]
        path = uttID_path_vect[1]
        uttID_path_dict[imageID] = path

scp_name = 'images.scp'
data_list_path = os.path.join(args.dir, 'backup', scp_name)

if args.out_ark == '-':
    out_fh = sys.stdout
else:
    out_fh = open(args.out_ark, 'wb')

text_file = os.path.join(args.dir, 'text.txt')
text_fh = open(text_file, 'w+')

utt2spk_file = os.path.join(args.dir, 'utt2spk')
utt2spk_fh = open(utt2spk_file, 'w+')

image_file = os.path.join(args.dir, 'images.scp')
image_fh = open(image_file, 'w+')

with open(data_list_path) as f:
    for line in f:
        line = line.strip()
        line_vect = line.split(' ')
        image_id = line_vect[0]
        image_path = line_vect[1]
        im = misc.imread(image_path)
        #im_contrast = contrast_normalization(im, 0.05, 0.2)
        #shear = (find_slant(im_contrast) / 360.0) * 2 * math.pi
        im_scaled = get_scaled_image(im)
        image_augment(im_scaled, out_fh, image_id)

for uttID in new_scp_list:
    imageID = uttID.split("_")[1]
    text_fh.write(uttID + ' ' + text_dict[imageID] + '\n')
    utt2spk_fh.write(uttID + ' ' + uttID_spk_dict[imageID] + '\n')
    image_fh.write(uttID + ' ' + uttID_path_dict[imageID] + '\n')
