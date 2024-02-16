#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (author: Chun-Chieh "Jonathan" Chang)

""" This script prepares the training and testing data for Imagenet
"""

import argparse
import os
import sys
import scipy.io as sio
import numpy as np
from scipy import misc

from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL) 

parser = argparse.ArgumentParser(description="""Converts the imagenet data into Kaldi feature format""")
parser.add_argument('database_path', type=str, help='path to downloaded imagenet training data')
parser.add_argument('devkit_path', type=str, help='path to meta data')
parser.add_argument('tar_name', type=str, help='name of extracted tar file. Used to find the folder extracted from the tar file.')
parser.add_argument('dir', type=str, help='output dir')
parser.add_argument('--dataset', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--out-ark', type=str, default='-', help='where to write the output feature file')
parser.add_argument('--scale-size', type=int, default=256, help='size to rescale the test image')
parser.add_argument('--crop-size', type=int, default=224, help='crop size of test image')

args = parser.parse_args()

# Image Dimensions
# W, H of images are not the same for every image.
C = 3

def parse_mat_path():
    tar_folder_vect = args.tar_name.split('.')
    tar_folder = tar_folder_vect[0]
    dataset_year_vect = tar_folder.split('_')
    dataset_year = dataset_year_vect[0]
    mat_seq = (args.devkit_path,tar_folder,"data/meta")
    mat_path = "/".join(mat_seq)
    val_ground_truth_seq = (args.devkit_path,
                            tar_folder, "data",
                            dataset_year + "_validation_ground_truth.txt")
    val_ground_truth_path = "/".join(val_ground_truth_seq)
    return mat_path, val_ground_truth_path

def write_kaldi_matrix(file_handle, matrix, key):
    #file_handle.write("[ ")
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
    
def zeropad(x, length):
    s = str(x)
    while len(s) < length:
        s = '0' + s
    return s

def findIndex(list_of_lists,wnid):
    index = 0
    for sublist in list_of_lists:
        if sublist[0] == wnid:
            return index
        index = index + 1
    return None

def get_image_crops(im):
    crop_size = args.crop_size
    scale_size = float(args.scale_size)
    sx = im.shape[1]
    sy = im.shape[0]
    if sx > sy:
        scale = scale_size / sy
    else:
        scale = scale_size / sx
    im = misc.imresize(im, scale)

    W = im.shape[1]
    H = im.shape[0]

    crop_topleft = im[0:crop_size, 0:crop_size]
    crop_topleft_fliplr = np.fliplr(crop_topleft)
    crop_topright = im[0:crop_size, (W - crop_size):W]
    crop_topright_fliplr = np.fliplr(crop_topright)
    crop_bottomleft = im[(H - crop_size):H, 0:crop_size]
    crop_bottomleft_fliplr = np.fliplr(crop_bottomleft)
    crop_bottomright = im[(H - crop_size):H, (W - crop_size):W]
    crop_bottomright_fliplr = np.fliplr(crop_bottomright)

    center_left = float(W) / 2 - float(crop_size) / 2
    center_top = float(H) / 2 - float(crop_size) / 2
    crop_center = im[center_top:(center_top + crop_size),
                     center_left:(center_left + crop_size)]
    crop_center_fliplr = np.fliplr(crop_center)

    tup1 = (crop_topleft, crop_topright,
            crop_bottomleft, crop_bottomright,
            crop_center)
    tup2 = (crop_topleft_fliplr, crop_topright_fliplr,
            crop_bottomleft_fliplr, crop_bottomright_fliplr,
            crop_center_fliplr)
    tup = tup1 + tup2
    return tup

### main ###
mat_path, val_ground_truth_path = parse_mat_path()
mat_content = sio.loadmat(mat_path)
synsets_struct = mat_content['synsets']
wnid_vect = synsets_struct['WNID']

if args.out_ark == '-':
    out_fh = sys.stdout
else:
    out_fh = open(args.out_ark,'wb')

labels_file = os.path.join(args.dir + '/', 'labels.txt')
labels_fh = open(labels_file, 'wb')

database_contents = sorted(os.listdir(args.database_path))

if args.dataset == 'train':
    image_id = 1
    for dir_file in database_contents:
        potential_path = args.database_path + '/' + dir_file
        if os.path.isdir(potential_path):
            index = findIndex(wnid_vect, dir_file)
            class_ID = synsets_struct[index][0][0][0][0]
            class_contents = sorted(os.listdir(potential_path))
            for img_name in class_contents:
                key = zeropad(image_id,8)
                im = misc.imread(potential_path + '/' + img_name)
                W = im.shape[1]
                H = im.shape[0]
                if len(im.shape) == 3:
                    data = np.reshape(np.transpose(im, (1, 0, 2)), (W, H * C))
                else:
                    im_three = np.dstack((im, im, im))
                    data = np.reshape(np.transpose(im_three, (1, 0, 2)), (W, H * C))
                data = np.divide(data, 255.0)
                labels_fh.write(key + ' ' + str(int(class_ID) - 1) + '\n')
                write_kaldi_matrix(out_fh, data, key)
                image_id = image_id + 1
else:
    image_id = 1
    image_num = 1
    dataset_year_vect = args.tar_name.split('_')
    with open(val_ground_truth_path) as f:
        for line in f:
            if int(line) > 0:
                keyID = zeropad(image_id,8)
                file_name = (args.database_path + '/'
                            + dataset_year_vect[0] + '_val_' + keyID + '.JPEG')
                im_orig = misc.imread(file_name)
                im_crops = get_image_crops(im_orig)
                for im in im_crops:
                    keyNum = zeropad(image_num, 8)
                    W = im.shape[1]
                    H = im.shape[0]
                    if len(im.shape) == 3:
                        data = np.reshape(np.transpose(im, (1, 0, 2)), (W, H * C))
                    else:
                        im_three = np.dstack((im, im, im))
                        data = np.reshape(np.transpose(im_three, (1, 0, 2)), (W, H * C))
                    data = np.divide(data, 255.0)
                    labels_fh.write(keyNum + ' ' + str(int(line) - 1) + '\n')
                    write_kaldi_matrix(out_fh, data, keyNum)
                    image_num = image_num + 1
            image_id = image_id + 1

labels_fh.close()
out_fh.close()


