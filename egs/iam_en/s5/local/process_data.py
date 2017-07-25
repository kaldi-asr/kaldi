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
                    default='data/download',
                    help='path to downloaded iam data')
parser.add_argument('split_file_path', type=str, default='data/download/dataSplit/trainset.txt', help='split file, contains file names belonging to that split')
parser.add_argument('out_dir', type=str, default='-', help='where to write output files')
args = parser.parse_args()

### main ###
text_file = os.path.join(args.out_dir + '/', 'text')
text_fh = open(text_file, 'w+')

utt2spk_file = os.path.join(args.out_dir + '/', 'utt2spk')
utt2spk_fh = open(utt2spk_file, 'w+')

image_file = os.path.join(args.out_dir + '/', 'images.scp')
image_fh = open(image_file, 'w+')
xml_path = os.path.join(args.database_path, 'xml')
split_files_fh = open(args.split_file_path)

for split_files in split_files_fh.readlines():
  sfile = split_files[:-1]
  xml_file = sfile[:-3] + '.xml'
  img_num = sfile[-3:]
  path = os.path.join(xml_path, xml_file)
  doc = minidom.parse(path)
  form_elements = doc.getElementsByTagName('form')[0]
  writer_id = form_elements.getAttribute('writer-id')

  outerfolder = form_elements.getAttribute('id')[0:3]
  innerfolder = form_elements.getAttribute('id')
  lines_path = os.path.join(args.database_path, 'lines')
  lines_path = os.path.join(lines_path, outerfolder)
  lines_path = os.path.join(lines_path, innerfolder)
  lines_path = os.path.join(lines_path, innerfolder)
  image_file_path = lines_path + img_num + '.png'
  utt_id = writer_id + '_' + sfile
  line_elements = doc.getElementsByTagName('line')
  for element in line_elements:
    text = element.getAttribute('text')
    image_id = element.getAttribute('id')
    if image_id==sfile:
      text_fh.write(utt_id + ' ' + text + '\n')
      utt2spk_fh.write(utt_id + ' ' + writer_id + '\n')
      image_fh.write(utt_id + ' ' + image_file_path + '\n') 

