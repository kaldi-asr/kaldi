#!/usr/bin/env python3

# Copyright      2018  Ashish Arora
#                2018  Chun Chieh Chang

""" This script reads the extracted Farsi OCR (yomdle and slam) database files 
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

parser = argparse.ArgumentParser(description="Creates text, utt2spk, and images.scp files")
parser.add_argument('database_path', type=str, help='Path to data')
parser.add_argument('out_dir', type=str, help='directory to output files')
parser.add_argument('--head', type=int, default=-1, help='limit on number of synth data')
args = parser.parse_args()

### main ###
print("Processing '{}' data...".format(args.out_dir))

text_file = os.path.join(args.out_dir, 'text')
text_fh = open(text_file, 'w', encoding='utf-8')
utt2spk_file = os.path.join(args.out_dir, 'utt2spk')
utt2spk_fh = open(utt2spk_file, 'w', encoding='utf-8')
image_file = os.path.join(args.out_dir, 'images.scp')
image_fh = open(image_file, 'w', encoding='utf-8')

count = 0
for filename in sorted(os.listdir(os.path.join(args.database_path, 'truth_csv'))):
    if filename.endswith('.csv') and (count < args.head or args.head < 0):
        count = count + 1
        csv_filepath = os.path.join(args.database_path, 'truth_csv', filename)
        csv_file = open(csv_filepath, 'r', encoding='utf-8')
        row_count = 0
        for row in csv.reader(csv_file):
            if row_count == 0:
                row_count = 1
                continue
            image_id = os.path.splitext(row[1])[0]
            image_filepath = os.path.join(args.database_path, 'truth_line_image', row[1])
            text = unicodedata.normalize('NFC', row[11]).replace('\n', '')
            if os.path.isfile(image_filepath) and os.stat(image_filepath).st_size != 0 and text:
                text_fh.write(image_id + ' ' + text + '\n')
                utt2spk_fh.write(image_id + ' ' + '_'.join(image_id.split('_')[:-1]) + '\n')
                image_fh.write(image_id + ' ' + image_filepath + ' ' + row[13] +  '\n')
