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
parser.add_argument('--dataset', type=str, default='train',
                    choices=['train','test'],
                    help='choose trainset, testset, validationset1, or validationset2')
parser.add_argument('--model_type', type=str,default='word',
                    choices=['word', 'character'],
                    help='word model or character model')
args = parser.parse_args()

### main ###

image_file = os.path.join(args.out_dir + '/', 'images.scp')
image_fh = open(image_file, 'w+')

utt2spk_file = os.path.join(args.out_dir + '/', 'utt2spk')
utt2spk_fh = open(utt2spk_file, 'w+')

text_dict = {}
utt_dict = {}
img_dict = {}
def process_text_file_for_word_model():
  imgs_y = []
  imgs_x = []
  sets = {}
  flag = 0
  if args.dataset == 'train':
    sets = {'set_a','set_b','set_c'}
    flag = 0
  else:
    sets = {'set_d'}
    flag = 0
  for dir_name in sorted(sets):
      if( dir_name == "set_e" or dir_name == "set_f" or dir_name == "set_s"):
        png_path = args.database_path + '/' + dir_name + '/png'
        tru_path = args.database_path + '/' + dir_name + '/tru'
        for i in range(0,len(os.listdir(png_path))-flag):
          png_file_name = sorted(os.listdir(png_path))[i][:-4]
          writer_id = png_file_name[0:5]
          utt_id = png_file_name
          image_fh.write(utt_id + ' ' + png_path + '/' + sorted(os.listdir(png_path))[i] + '\n' )
          utt2spk_fh.write(utt_id + ' ' + writer_id + '\n')
      else: 
        png_path = args.database_path + '/' + dir_name + '/png'
        tru_path = args.database_path + '/' + dir_name + '/tru'
        for i in range(0,len(os.listdir(png_path))-flag):
          png_file_name = sorted(os.listdir(png_path))[i][:-4]
          writer_id = png_file_name[0:4]
          utt_id = png_file_name
          image_fh.write(utt_id + ' ' + png_path + '/' + sorted(os.listdir(png_path))[i] + '\n' )
          utt2spk_fh.write(utt_id + ' ' + writer_id + '\n')


if args.model_type=='word':
  print 'processing word model'
  process_text_file_for_word_model()


