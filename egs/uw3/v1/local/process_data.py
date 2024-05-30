#!/usr/bin/env python3

# Copyright       2017  Chun Chieh Chang

# This script goes through the downloaded UW3 dataset and creates data files "text",
# "utt2spk", and "images.scp" for the train and test subsets in data/train and data/test.

# text - matches the transcriptions with the image id
# utt2spk - matches the image id's with the speaker/writer names
# images.scp - matches the image is's with the actual image file

import argparse
import os
import random

parser = argparse.ArgumentParser(description="""Creates data/train and data/test.""")
parser.add_argument('database_path', help='path to downloaded (and extracted) UW3 corpus')
parser.add_argument('out_dir', default='data',
                    help='where to create the train and test data directories')
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
        train_text_fh.write("{} {}\n".format(utt_id, text))
        train_utt2spk_fh.write("{} {}\n".format(utt_id, page_count))
        train_image_fh.write("{} {}\n".format(utt_id, image_path))
      elif coin < 1:
        test_text_fh.write("{} {}\n".format(utt_id, text))
        test_utt2spk_fh.write("{} {}\n".format(utt_id, page_count))
        train_image_fh.write("{} {}\n".format(utt_id, image_path))
