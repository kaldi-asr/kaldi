#!/usr/bin/env python3

# Copyright      2018  Ashish Arora
#                2018  Chun Chieh Chang

""" This script reads the extracted Tamil OCR (yomdle and slam) database files 
    and creates the following files (for the data subset selected via --dataset):
    text, utt2spk, images.scp.
  Eg. local/process_data.py data/download/ data/local/splits/train.txt data/train

  Eg. text file: english_phone_books_0001_1 To sum up, then, it would appear that
      utt2spk file: english_phone_books_0001_0 english_phone_books_0001
      images.scp file: english_phone_books_0001_0 \
      data/download/truth_line_image/english_phone_books_0001_0.png
"""

import argparse
import os
import sys
import csv
import itertools
import unicodedata
import re
import string
parser = argparse.ArgumentParser(description="Creates text, utt2spk, and images.scp files")
parser.add_argument('database_path', type=str, help='Path to data')
parser.add_argument('data_split', type=str, help='Path to file that contain datasplits')
parser.add_argument('out_dir', type=str, help='directory to output files')
args = parser.parse_args()

### main ###
print("Processing '{}' data...".format(args.out_dir))

text_file = os.path.join(args.out_dir, 'text')
text_fh = open(text_file, 'w', encoding='utf-8')
utt2spk_file = os.path.join(args.out_dir, 'utt2spk')
utt2spk_fh = open(utt2spk_file, 'w', encoding='utf-8')
image_file = os.path.join(args.out_dir, 'images.scp')
image_fh = open(image_file, 'w', encoding='utf-8')

with open(args.data_split) as f:
    for line in f:
        line = line.strip()
        image_id = line
        image_filename = image_id + '.png'
        image_filepath = os.path.join(args.database_path, 'truth_line_image', image_filename)
        if not os.path.isfile (image_filepath):
            print("File does not exist {}".format(image_filepath))
            continue
        line_id = int(line.split('_')[-1])
        csv_filename = '_'.join(line.split('_')[:-1]) + '.csv'
        csv_filepath = os.path.join(args.database_path, 'truth_csv', csv_filename)
        csv_file = open(csv_filepath, 'r', encoding='utf-8')
        for row in csv.reader(csv_file):
            if row[1] == image_filename:
                text = row[11]
                if not text:
                    continue
                text_fh.write(image_id + ' ' + text + '\n')
                utt2spk_fh.write(image_id + ' ' + '_'.join(line.split('_')[:-1]) + '\n')
                image_fh.write(image_id + ' ' + image_filepath +  '\n')
