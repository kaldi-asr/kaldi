#!/usr/bin/env python

# You can use this to select an image in an egs file:
# e.g.: 
# nnet3-copy-egs ark:exp/cifar10_egs/egs.1.ark ark,t:- | \
#     image/select_image_in_egs.py 00001 | image/matrix_to_image.py --color 3 > 00001.bmp
# e.g.:
# id=00002; nnet3-egs-augment-image --horizontal-flip-prob=0.2 --horizontal-shift=0.3 \
#     --vertical-shift=0.3 --srand=27 --num-channels=3 ark:exp/cifar10_egs/egs.1.ark ark,t:- | \
#     image/select_image_in_egs.py $id | image/matrix_to_image.py --color 3 > $id.bmp

from __future__ import print_function
import argparse
import sys

parser = argparse.ArgumentParser(description="""Select the image matrix for
         the eg specified by img_id. For purposes of viewing/debugging.""")
parser.add_argument('img_id', type=str, default=None,
                    help='image id')
args = parser.parse_args()

found = False
while True:
    line = sys.stdin.readline().strip()
    if not line:
        break
    if line.find(args.img_id + ' <Nnet3Eg>') != -1:
        found=True
        print('[')
        continue
    if found and line.find('</NnetIo>') != -1:
        break
    if found:
        print(line)

