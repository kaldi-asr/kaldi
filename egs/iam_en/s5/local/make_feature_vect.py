#!/usr/bin/env python

import argparse
import os
import sys
import scipy.io as sio
import numpy as np
from scipy import misc
from scipy import ndimage

from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL)

parser = argparse.ArgumentParser(description="""Generates and saves the feature vectors""")
parser.add_argument('dir', type=str, help='directory of images.scp and is also output directory')
parser.add_argument('--out-ark', type=str, default='-', help='where to write the output feature file')
parser.add_argument('--scale-size', type=int, default=40, help='size to scale the height of all images')
parser.add_argument('--color', type=int, default=3, help='number of color channels for image')
parser.add_argument('--pad', type=bool, default=False, help='pad the left and right of image with zeros')

args = parser.parse_args()


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

def get_scaled_image(im):
    scale_size = args.scale_size
    sx = im.shape[1]
    sy = im.shape[0]
    scale = (1.0 * scale_size) / sy
    nx = int(scale_size)
    ny = int(scale * sx)

    im = misc.imresize(im, (nx, ny))
    
    return im

### main ###
data_list_path = os.path.join(args.dir,'images.scp')

if args.out_ark == '-':
    out_fh = sys.stdout
else:
    out_fh = open(args.out_ark,'wb')

image_ID = 1
with open(data_list_path) as f:
    for line in f:
        line = line.strip()
        line_vect = line.split(' ')
        image_id = line_vect[0]
        image_path = line_vect[1]
        try:
            im = misc.imread(image_path)
        except:
            im = np.zeros((40,80))
        im_scale = get_scaled_image(im)
        
        if args.pad:
            pad = np.ones((args.scale_size, 10)) * 255
            im_data = np.hstack((pad, im_scale, pad))
        else:
            im_data = im_scale

        if args.color == 3:
            W = im_data.shape[1]
            H = im_data.shape[0]
            C = 3
            im_three = np.dstack((im_data, im_data, im_data))
            data = np.reshape(np.transpose(im_three, (1, 0, 2)), (W, H * C))
        elif args.color == 1:
            data = np.transpose(im_data, (1, 0))

        data = np.divide(data, 255.0)
        write_kaldi_matrix(out_fh, data, image_id)

