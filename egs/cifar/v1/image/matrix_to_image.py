#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (author: Daniel Povey)
#           2017 Yiwen Shao
#           2017 Hossein Hadian
# Apache 2.0


""" This script converts a Kaldi-format text matrix into a bitmap image.
    It reads the matrix from its stdin and writes the .bmp image to its
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

   E.g. to see a (grayscale) line image from training feature files:
imgid=0001_010006;
copy-feats --binary=false $(grep $imgid data/train/feats.scp | cut -d' ' -f2) - | \
           image/matrix_to_image.py --color=1 > $imgid.bmp
"""

import argparse
import sys
from bmp_encoder import *


parser = argparse.ArgumentParser(description="""Converts Kaldi-format text matrix
           representing an image on stdin into bmp image on stdout.  See
           comments at top of script for more details.""")

parser.add_argument('--color', type=int, choices=(1, 3), default=3,
                    help='3 if the image is in RGB, 1 if the image is in grayscale.')


args = parser.parse_args()

matrix = []
num_rows = 0
num_cols = 0
while True:
    line = sys.stdin.readline().strip('\n').split()
    if line == []:
        break
    if line == ['[']:  # deal with the case that the first row only contains "["
        continue
    if line[0] == '[':  # drop the "[" in the first row
        line = line[1:]
    if line[-1] == ']':  # drop the "]" in the last row
        line = line[:-1]
    if num_cols == 0:
        num_cols = len(line)  # initialize
    if len(line) != num_cols:
        raise Exception("All rows should be of the same length")
    line = map(float, line)  # string to float
    if max(line) > 1:
        raise Excetion("Element value in the matrix should be normalized and no larger than 1")
    line = [int(x * 255) for x in line]  # float to integer ranging from 0 to 255
    matrix.append(line)
    num_rows += 1

if args.color == 3:
    if num_cols % 3 != 0:
        raise Exception("Number of columns should be a multiple of 3 in the color mode")
    width = num_rows
    height = num_cols / 3
    # reform the image matrix
    image_array = [[0 for i in range(width * 3)] for j in range(height)]
    for i in range(height):
        for j in range(width):
            image_array[i][3 * j] = matrix[j][3 * i]
            image_array[i][3 * j + 1] = matrix[j][3 * i + 1]
            image_array[i][3 * j + 2] = matrix[j][3 * i + 2]
    bmp_encoder(image_array, width, height)

elif args.color == 1:
    width = num_rows
    height = num_cols
    # reform the image matrix
    image_array = [[0 for i in range(width * 3)] for j in range(height)]
    for i in range(height):
        for j in range(width):
            image_array[i][3 * j] = matrix[j][i]
            image_array[i][3 * j + 1] = matrix[j][i]
            image_array[i][3 * j + 2] = matrix[j][i]
    bmp_encoder(image_array, width, height)
