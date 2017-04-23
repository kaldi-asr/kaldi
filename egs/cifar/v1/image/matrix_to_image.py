#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (author: Daniel Poveh)
#           2017 Yiwen Shao
# Apache 2.0


""" This script converts a Kaldi-format text matrix into a jpeg image.
    It reads the matrix from its stdin and writes the jpeg image to its
    stdout.
    For instance:
cat <<EOF | image/matrix_to_image.py --color=true > foo.jpeg
  [ 0.0  0.5  1.0
    0.0  0.0  0.0  ]
EOF
   The image format is that the number of rows equals the width of the image, and the
   number of columns equals the height of the image times the number of channels
   (1 for black and white, 3 for color (RGB)), with the channel varying the
   fastest.  The above example would produce a color image with width 2 and
   height 1.

"""

import argparse
import os
import sys


parser = argparse.ArgumentParser(description="""Converts Kaldi-format text matrix
           representing an image on stdin into jpeg image on stdout.  See
           comments at top of script for more details.""")

parser.add_argument('--color', type=bool, default=True,
                    help='True if the image is in color ')


args = parser.parse_args()


# TODO.
