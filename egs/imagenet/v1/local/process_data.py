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

parser = argparse.ArgumentParser(description="""Converts the imagenet data into Kaldi feature format""")
parser.add_argument('databasePath', type=str, help='path to downloaded imagenet training data')
parser.add_argument('devkitPath', type=str, help='path to meta data')
parser.add_argument('tarName',type=str, help='name of extracted tar file')
parser.add_argument('dir', type=str, help='output dir')
parser.add_argument('--dataset', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--out-ark', type=str, default='-', help='where to write the output feature file')

args = parser.parse_args()

# Image Dimensions
# W, H of images are not the same for every image.
C = 3

def parse_mat_path():
    tarFolderVect = args.tarName.split('.')
    tarFolder = tarFolderVect[0]
    datasetYearVect = tarFolder.split('_')
    datasetYear = datasetYearVect[0]
    mat_seq = (args.devkitPath,tarFolder,"data/meta")
    mat_path = "/".join(mat_seq)
    val_ground_truth_seq = (args.devkitPath,tarFolder,"data",datasetYear + "_validation_ground_truth.txt")
    val_ground_truth_path = "/".join(val_ground_truth_seq)
    return mat_path, val_ground_truth_path

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
    
def zeropad(x, length):
    s = str(x)
    while len(s) < length:
        s = '0' + s
    return s

def findIndex(listOfLists,wnid):
    index = 0
    for sublist in listOfLists:
        if sublist[0] == wnid:
            return index
        index = index + 1
    return None


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
labels_fh = open(labels_file,'wb')

database_contents = sorted(os.listdir(args.databasePath))

if args.dataset == 'train':
    image_id = 1
    for dir_file in database_contents:
        potential_path = args.databasePath + '/' + dir_file
        if os.path.isdir(potential_path):
            index = findIndex(wnid_vect,dir_file)
            class_ID = synsets_struct[index][0][0][0][0]
            class_contents = sorted(os.listdir(potential_path))
            for img_Name in class_contents:
                key = zeropad(image_id,8)
                im = misc.imread(potential_path + '/' + img_Name)
		im = np.divide(im,255.0)
                im_shape = im.shape
                W = im_shape[1]
                H = im_shape[0]
                if len(im_shape) == 3:
                    data = np.reshape(np.transpose(im, (1, 0, 2)), (W, H * C))
                else:
                    data = np.reshape(np.transpose(np.dstack((im, im, im)), (1, 0, 2)), (W, H * C))
                labels_fh.write(key + ' ' + str(int(class_ID)-1) + '\n')
                write_kaldi_matrix(out_fh, data, key)
                image_id = image_id + 1
else:
    image_id = 1
    image_num = 1
    datasetYearVect = args.tarName.split('_')
    with open(val_ground_truth_path) as f:
        for line in f:
            if int(line) > 0:
                keyID = zeropad(image_id,8)
                keyNum = zeropad(image_num,8)
                file_name = args.databasePath + '/' + datasetYearVect[0] + '_val_' + keyID + '.JPEG'
                im = misc.imread(file_name)
		im = np.divide(im,255.0)
                im_shape = im.shape
                W = im_shape[1]
                H = im_shape[0]
                if len(im_shape) == 3:
                    data = np.reshape(np.transpose(im, (1, 0, 2)), (W, H * C))
                else:
                    data = np.reshape(np.transpose(np.dstack((im, im, im)), (1, 0, 2)), (W, H * C))
                labels_fh.write(keyNum + ' ' + str(int(line)-1) + '\n')
                write_kaldi_matrix(out_fh, data, keyNum)
                image_num = image_num + 1
            image_id = image_id + 1

labels_fh.close()
out_fh.close()


