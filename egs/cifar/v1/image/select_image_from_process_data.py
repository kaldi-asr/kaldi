#!/usr/bin/env python

# You can use this to select an image from the output of the process_data.py file:
# Mainly used to make sure that the cropping of the test images are correct if cropping is done.
# e.g.: 
# local/process_data.py /export/b18/imagenet_2012/val/ \
#                       /export/b18/imagenet_2012/devkit_t3 \
#                       ILSVRC2012_devkit_t3.tar data/test \
#                       --dataset test --scale-size 256 --crop-size 224 \
#   | image/select_image_from_process_data.py 00000001
#   | image/matrix_to_image.py --color 3 > test.bmp

import argparse
import sys

from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL) 

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
    if line.find(args.img_id) != -1:
        found = True
        line_len = len(args.img_id)
        line = line[(line_len + 1):]
        print(line)
        continue
    if found and line.find(']') != -1:
        line = line[:-1]
        print(line)
        break
    if found:
        print(line)

