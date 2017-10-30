#!/usr/bin/env python

import argparse
import os
import sys
import numpy as np
from scipy import misc
import xml.dom.minidom as minidom

parser = argparse.ArgumentParser(description="""Creates text utt2spk 
                                                and image file """)
parser.add_argument('database_path', type=str,
                    help='path to downloaded iam data')
parser.add_argument('out_dir', type=str,
                    help='where to write output files')
parser.add_argument('dataset_dir', type=str,
                    help='directory containing dataset')
parser.add_argument('--dataset', type=str, default='new_trainset',
                    choices=['new_trainset', 'new_testset','new_valset'],
                    help='choose new_trainset, testset')
parser.add_argument('--model_type', type=str,default='word',
                    choices=['word', 'character'],
                    help='word model or character model')
args = parser.parse_args()

### main ###
text_file = os.path.join(args.out_dir + '/', 'text')
text_fh = open(text_file, 'w')

utt2spk_file = os.path.join(args.out_dir + '/', 'utt2spk')
utt2spk_fh = open(utt2spk_file, 'w')

image_file = os.path.join(args.out_dir + '/', 'images.scp')
image_fh = open(image_file, 'w')

dataset_path = os.path.join(args.dataset_dir,
                            args.dataset + '.txt')

text_file_path = os.path.join(args.database_path,
                               'ascii','lines.txt')
text_dict = {}
def process_text_file_for_word_model():
  with open (text_file_path, 'rt') as in_file:
    for line in in_file:
      if line[0]=='#':
        continue
      line = line.strip()
      line_vect = line.split(' ')
      text_vect = line.split(' ')[8:]
      text = "".join(text_vect)
      text = text.replace("|", " ")
      text_dict[line_vect[0]] = text

def process_text_file_for_char_model():
  with open (text_file_path, 'rt') as in_file:
    for line in in_file:
      if line[0]=='#':
        continue
      line = line.strip()
      line_vect = line.split(' ')
      text_vect = line.split(' ')[8:]
      text = "".join(text_vect)
      characters = list(text)
      spaced_characters = " ".join(characters)
      spaced_characters = spaced_characters.replace("|", "SIL")
      spaced_characters = "SIL " + spaced_characters
      spaced_characters = spaced_characters + " SIL"
      text_dict[line_vect[0]] = spaced_characters


if args.model_type=='word':
  print 'processing word model'
  process_text_file_for_word_model()
else:
  print 'processing char model'
  process_text_file_for_char_model()

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
    lines_path = os.path.join(args.database_path, 'lines', outerfolder, innerfolder, innerfolder)
    image_file_path = lines_path + img_num + '.png'
    text =  text_dict[line]
    utt_id = writer_id + '_' + line
    text_fh.write(utt_id + ' ' + text + '\n')
    utt2spk_fh.write(utt_id + ' ' + writer_id + '\n')
    image_fh.write(utt_id + ' ' + image_file_path + '\n')
