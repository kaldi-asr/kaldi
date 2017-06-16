#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (author: Chun-Chieh "Jonathan" Chang)

# This script reads all the classes from the .mat file and puts them into text format

import argparse
import os
import sys
import scipy.io as sio
import string

parser = argparse.ArgumentParser(description="""This script reads the classes from the .mat file and puts them into a text file""")
parser.add_argument('devkit_path', type=str, help='Path to where devkit was extracted')
parser.add_argument('tar_name', type=str, help='Name of tar file. Used to find the folder extracted from the tar file.')
parser.add_argument('out_dir', type=str, help='Where the output should be saved')
parser.add_argument('--task', type=int, default=1, choices=[1, 2, 3],
                    help='Which Imagenet challenge task')

args = parser.parse_args()

def parse_mat_path():
    tar_folder_vect = args.tar_name.split('.')
    tar_folder = tar_folder_vect[0]
    seq = (args.devkit_path, tar_folder, "data/meta")
    path = "/".join(seq)
    return path


### main ###
mat_path = parse_mat_path()
mat_content = sio.loadmat(mat_path)
synsets_struct = mat_content['synsets']
wnid_vect = synsets_struct['WNID']

labels_file = os.path.join(args.out_dir + '/', 'classes.txt')
labels_fh = open(labels_file, 'wb')

if args.task < 3:
    for i in range(0, 1000):
        labels_fh.write(wnid_vect[i][0][0] + ' ' + str(i) + '\n')
else:
    for i in range(0, len(wnid_vect)):
        labels_fh.write(wnid_vect[i][0][0] + ' '
                        + str(int(synsets_struct[i][0][0][0][0]) - 1) + '\n')

labels_fh.close()
