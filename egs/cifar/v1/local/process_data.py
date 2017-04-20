#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (author: Hossein Hadian)
# Apache 2.0


""" This script prepares the training and test data for CIFAR-10 or CIFAR-100.
"""

import argparse
import os
import sys
import re
import errno


sys.path.insert(0, 'steps')
import libs.common as common_lib

parser = argparse.ArgumentParser(description="""Converts train/test data of
                                                CIFAR-10 or CIFAR-100 to 
                                                Kaldi feature format""")
parser.add_argument('database', type=str,
                    default='data/dl/cifar-10-batches-bin',
                    description='path to downloaded cifar data (binary version)')
parser.add_argument('dir', type=str, description='output dir')
parser.add_argument('--dataset', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--out-ark', type=str, default='-', description='where to write output feature data')

args = parser.parse_args()

# CIFAR image dimensions:
C = 3  # num_channels
H = 32  # num_rows
W = 32  # num_cols

def load_cifar10_data_batch(datafile):
    num_images_in_batch = 10000
    data = []
    labels = []
    with open(datafile, 'rb') as fh:
        for i in range(num_images_in_batch):
            label = ord(fh.read(1))
            bin_img = fh.read(C * H * W)
            img = [[[ord(byte) for byte in bin_img[channel*H*W+row*W:channel*H*W+(row+1)*W]]
                  for row in range(H)] for channel in range(C)]
            labels += [label]
            data += [img]
    return data, labels

def load_cifar100_data_batch(datafile):
    num_images_in_batch = 10000
    data = []
    fine_labels = []
    coarse_labels = []
    with open(datafile, 'rb') as fh:
        for i in range(num_images_in_batch):
            coarse_label = ord(fh.read(1))
            fine_label = ord(fh.read(1))
            bin_img = fh.read(C * H * W)
            img = [[[ord(byte) for byte in bin_img[channel*H*W+row*W:channel*H*W+(row+1)*W]]
                  for row in range(H)] for channel in range(C)]
            fine_labels += [fine_label]
            coarse_labels += [coarse_label]
            data += [img]
    return data, fine_labels, coarse_labels

def image_to_feat_matrix(img):
  mat = [0]*H  # 32 * 96
  for row in range(H):
    mat[row] = [0]*C*W
    for ch in range(C):
      for col in range(W):
        mat[row][col*C+ch] = img[ch][row][col]
  return mat

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
        file_handle.write(" ".join(map(lambda x: str(x), matrix[row_index])))
        if row_index != num_rows - 1:
            file_handle.write("\n")
    file_handle.write(" ]\n")

def zeropad(x, length):
  s = str(x)
  while len(s) < length:
    s = '0' + s
  return s

### main ###
cifar10 = (args.database.find('cifar-100') == -1)
if args.out_ark == '-':
  out_fh = sys.stdout  # output file handle to write the feats to
else:
  out_fh = open(args.out_ark, 'wb')

if cifar10:
    img_id = 1  # similar to utt_id
    labels_file = os.path.join(args.dir, 'labels.txt')
    labels_fh = open(labels_file, 'wb')
    

    if args.dataset == 'train':
        for i in range(1, 6):
            fpath = os.path.join(args.database, 'data_batch_' + str(i) + '.bin')
            data, labels = load_cifar10_data_batch(fpath)
            for i in range(len(data)):
                key = zeropad(img_id, 5)
                labels_fh.write(key + ' ' + str(labels[i]) + '\n')
                feat_mat = image_to_feat_matrix(data[i])
                write_kaldi_matrix(out_fh, feat_mat, key)
                img_id += 1
    else:
        fpath = os.path.join(args.database, 'test_batch.bin')
        data, labels = load_cifar10_data_batch(fpath)
        for i in range(len(data)):
            key = zeropad(img_id, 5)
            labels_fh.write(key + ' ' + str(labels[i]) + '\n')
            feat_mat = image_to_feat_matrix(data[i])
            write_kaldi_matrix(out_fh, feat_mat, key)
            img_id += 1

    labels_fh.close()
else:
    img_id = 1  # similar to utt_id
    fine_labels_file = os.path.join(args.dir, 'fine_labels.txt')
    coarse_labels_file = os.path.join(args.dir, 'coarse_labels.txt')
    fine_labels_fh = open(fine_labels_file, 'wb')
    coarse_labels_fh = open(coarse_labels_file, 'wb')

    if args.dataset == 'train':
        fpath = os.path.join(args.database, 'train.bin')
        data, fine_labels, coarse_labels = load_cifar100_data_batch(fpath)
        for i in range(len(data)):
            key = zeropad(img_id, 5)
            fine_labels_fh.write(key + ' ' + str(fine_labels[i]) + '\n')
            coarse_labels_fh.write(key + ' ' + str(coarse_labels[i]) + '\n')
            feat_mat = image_to_feat_matrix(data[i])
            write_kaldi_matrix(out_fh, feat_mat, key)
            img_id += 1
    else:
        fpath = os.path.join(args.database, 'test.bin')
        data, fine_labels, coarse_labels = load_cifar100_data_batch(fpath)
        for i in range(len(data)):
            key = zeropad(img_id, 5)
            fine_labels_fh.write(key + ' ' + str(fine_labels[i]) + '\n')
            coarse_labels_fh.write(key + ' ' + str(coarse_labels[i]) + '\n')
            feat_mat = image_to_feat_matrix(data[i])
            write_kaldi_matrix(out_fh, feat_mat, key)
            img_id += 1

    fine_labels_fh.close()
    coarse_labels_fh.close()

out_fh.close()
