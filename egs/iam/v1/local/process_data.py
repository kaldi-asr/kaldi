#!/usr/bin/env python3

# Copyright      2017  Chun Chieh Chang
#                2017  Ashish Arora

""" This script reads the extracted IAM database files and creates
    the following files (for the data subset selected via --dataset):
    text, utt2spk, images.scp.

  Eg. local/process_data.py data/local data/train data --dataset train
  Eg. text file: 000_a01-000u-00 A MOVE to stop Mr. Gaitskell from
      utt2spk file: 000_a01-000u-00 000
      images.scp file: 000_a01-000u-00 data/local/lines/a01/a01-000u/a01-000u-00.png
"""

import argparse
import os
import sys
import xml.dom.minidom as minidom

parser = argparse.ArgumentParser(description="""Creates text, utt2spk
                                                and images.scp files.""")
parser.add_argument('database_path', type=str,
                    help='Path to the downloaded (and extracted) IAM data')
parser.add_argument('out_dir', type=str,
                    help='Where to write output files.')
parser.add_argument('--dataset', type=str, default='train',
                    choices=['train', 'test','validation'],
                    help='Subset of data to process.')
args = parser.parse_args()

text_file = os.path.join(args.out_dir + '/', 'text')
text_fh = open(text_file, 'w')

utt2spk_file = os.path.join(args.out_dir + '/', 'utt2spk')
utt2spk_fh = open(utt2spk_file, 'w')

image_file = os.path.join(args.out_dir + '/', 'images.scp')
image_fh = open(image_file, 'w')

dataset_path = os.path.join(args.database_path,
                            args.dataset + '.uttlist')

text_file_path = os.path.join(args.database_path,
                              'ascii','lines.txt')
text_dict = {}
def process_text_file_for_word_model():
  with open (text_file_path, 'rt') as in_file:
    for line in in_file:
      if line[0]=='#':
        continue
      line = line.strip()
      utt_id = line.split(' ')[0]
      text_vect = line.split(' ')[8:]
      text = "".join(text_vect)
      text = text.replace("|", " ")
      text_dict[utt_id] = text

print("Processing '{}' data...".format(args.dataset))
process_text_file_for_word_model()

with open(dataset_path) as f:
  for line in f:
    line = line.strip()
    line_vect = line.split('-')
    xml_file = line_vect[0] + '-' + line_vect[1]
    xml_path = os.path.join(args.database_path, 'xml', xml_file + '.xml')
    img_num = line[-3:]
    doc = minidom.parse(xml_path)

    form_elements = doc.getElementsByTagName('form')[0]
    writer_id = form_elements.getAttribute('writer-id')
    outerfolder = form_elements.getAttribute('id')[0:3]
    innerfolder = form_elements.getAttribute('id')
    lines_path = os.path.join(args.database_path, 'lines',
                              outerfolder, innerfolder, innerfolder)
    image_file_path = lines_path + img_num + '.png'
    text =  text_dict[line]
    utt_id = writer_id + '_' + line
    text_fh.write(utt_id + ' ' + text + '\n')
    utt2spk_fh.write(utt_id + ' ' + writer_id + '\n')
    image_fh.write(utt_id + ' ' + image_file_path + '\n')
