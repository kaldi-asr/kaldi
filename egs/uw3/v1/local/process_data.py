#!/usr/bin/env python3

# Copyright 2017 (Author: Chun Chieh Chang)

# This script goes through the downloaded dataset and creates the text, utt2spk, image.scp
# text - matches the labels with the image name
# utt2spk - matches the image names with the speaker/writer names
# image.scp - matches the image names with the actual image file

import argparse
import os
import sys
import numpy as np
import random
from scipy import misc

parser = argparse.ArgumentParser(description="""Create text utt2spk image.scp""")
parser.add_argument('database_path', type=str, help='path to downloaded data')
parser.add_argument('out_dir', type=str, help='where to write output files')
parser.add_argument('--model_type', type=str, default='word',
                    choices=['word', 'character'],
                    help='word model or character model')
args = parser.parse_args()

### main ###
train_text_file = os.path.join(args.out_dir, 'train', 'text')
train_text_fh = open(train_text_file, 'w+')
train_utt2spk_file = os.path.join(args.out_dir, 'train', 'utt2spk')
train_utt2spk_fh = open(train_utt2spk_file, 'w+')
train_image_file = os.path.join(args.out_dir, 'train', 'images.scp')
train_image_fh = open(train_image_file, 'w+')

test_text_file = os.path.join(args.out_dir, 'test', 'text')
test_text_fh = open(test_text_file, 'w+')
test_utt2spk_file = os.path.join(args.out_dir, 'test', 'utt2spk')
test_utt2spk_fh = open(test_utt2spk_file, 'w+')
test_image_file = os.path.join(args.out_dir, 'test', 'images.scp')
test_image_fh = open(test_image_file, 'w+')

random.seed(0)
page_count = 0
for page in sorted(os.listdir(args.database_path)):
  page_path = os.path.join(args.database_path, page)
  page_count = page_count + 1
  for line in sorted(os.listdir(page_path)):
    if line.endswith('.txt'):
      text_path = os.path.join(args.database_path, page, line)
      image_name = line.split('.')[0]
      image_path = os.path.join(args.database_path, page, image_name + '.png')
      utt_id = page + '_' + image_name
      gt_fh = open(text_path, 'r')
      text = gt_fh.readlines()[0].strip()
      
      # The UW3 dataset doesn't have established training and testing splits
      # The dataset is randomly split train 95% and test 5%
      coin = random.randint(0, 20)
      if coin >= 1:
        train_text_fh.write(utt_id + ' ' + text + '\n')
        train_utt2spk_fh.write(utt_id + ' ' + str(page_count) + '\n')
        train_image_fh.write(utt_id + ' ' + image_path + '\n')
      elif coin < 1:
        test_text_fh.write(utt_id + ' ' + text + '\n')
        test_utt2spk_fh.write(utt_id + ' ' + str(page_count) + '\n')
        test_image_fh.write(utt_id + ' ' + image_path + '\n')
