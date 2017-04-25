#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (author: Daniel Poveh)
#           2017 Yiwen Shao
# Apache 2.0


""" This script converts a Kaldi-format text matrix into a png image.
    It reads the matrix from its stdin and writes the png image to its
    stdout.
    For instance:
cat <<EOF | image/matrix_to_image.py --color 3 > foo.png
  [ 0.0  0.5  1.0
    0.0  0.0  0.0  ]
EOF
   The image format is that the number of rows equals the width of the image, and the
   number of columns equals the height of the image times the number of channels
   (1 for black and white, 3 for color (RGB)), with the channel varying the
   fastest.  The above example would produce a color image with width 2 and
   height 1. The first row corresponds to the left side of the image, and the 
   first column corresponds to the top of the image. 
"""

import argparse
import sys
import numpy as ny
from PIL import Image


parser = argparse.ArgumentParser(description="""Converts Kaldi-format text matrix
           representing an image on stdin into png image on stdout.  See
           comments at top of script for more details.""")

parser.add_argument('--color', type=int, default=3,
                    help='3 if the image is in RGB, 1 if the image is in grayscale ')


args = parser.parse_args()

matrix = []
num_rows = 0
num_cols = 0
while True:
    tmp = sys.stdin.readline().strip('\n').split()
    if tmp == []:
        break
    if tmp[0] == '[': # drop the "[" in the first row
        tmp = tmp[1:]
    if tmp[-1] == ']': # drop the "]" in the last row
        tmp = tmp[:-1]
    if num_rows == 0:
        num_cols = len(tmp) # initialize
    if len(tmp) != num_cols:
        raise Exception("All rows should be of same length")
    tmp = map(float, tmp) # string to float
    if max(tmp) > 1:
        raise Excetion("Elmement vaule in the matrix should be normalized and no larger than 1")
    tmp = [int(x * 255) for x in tmp] # float to integer ranging from 0 to 255
    matrix.append(tmp)
    num_rows+=1

if args.color == 3:
    if num_cols%3!=0:
        raise Exception("Number of columns should be 3*n in the colorful mode")
    width = num_rows
    height = num_cols/3

    image_array = ny.zeros((height, width, chan), dtype=ny.uint8)
    for i in range(height):
        for j in range(width):
            image_array[i,j] = [matrix[j][3*i], matrix[j][3*i+1], matrix[j][3*i+2]]
    im = Image.fromarray(image_array)
    im.save(sys.stdout,'png')
else:
    width = num_rows
    height = num_cols
    image_array = ny.zeros((height,width),dtype=ny.uint8)
    for i in range(height):
        for j in range(width):
            image_array[i,j] = matrix[j][i]
    im = Image.fromarray(image_array)
    im.save(sys.stdout,'png')

