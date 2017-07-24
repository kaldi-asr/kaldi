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
xml_files = os.listdir(xml_path)
for first_file in xml_files:
  first_path = os.path.join(xml_path, first_file)
  doc = minidom.parse(first_path)
  form_elements = doc.getElementsByTagName('form')[0]
  writer_id = form_elements.getAttribute('writer-id')

  outerfolder = form_elements.getAttribute('id')[0:3]
  innerfolder = form_elements.getAttribute('id')
  lines_path = os.path.join(args.database_path, 'lines')
  lines_path = os.path.join(lines_path, outerfolder)
  lines_path = os.path.join(lines_path, innerfolder)
  lines_path = os.path.join(lines_path, innerfolder)

  line_elements = doc.getElementsByTagName('line')
  #add a loop
  utt_id_list = []
  text_list = []
  writer_id_list = []
  image_file_path_list = []
  for ele in line_elements:
    text = ele.getAttribute('text')
    image_id = ele.getAttribute('id')
    img_num = ele.getAttribute('id')[-3:]
    image_file_path = lines_path + img_num + '.png'
    utt_id = writer_id + '_' + image_id
    utt_id_list += [utt_id]
    text_list += [text]
    writer_id_list += [writer_id]
    image_file_path_list += [image_file_path]
    text_fh.write(utt_id + ' ' + text + '\n')
    utt2spk_fh.write(utt_id + ' ' + writer_id + '\n')
    image_fh.write(utt_id + ' ' + image_file_path + '\n') 

#text_fh.write(utt_id + ' ' + text + '\n')
#utt2spk_fh.write(utt_id + ' ' + writer_id + '\n')
#image_fh.write(utt_id + ' ' + image_file_path + '\n')
