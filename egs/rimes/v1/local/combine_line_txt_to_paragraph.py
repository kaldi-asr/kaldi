#!/usr/bin/env python3

""" This script creates paragraph level text file. It reads 
    the line level text file and combines them to get
    paragraph level file.
  Eg. local/combine_line_txt_to_paragraph.py
  Eg. Input:  writer000000_eval2011-0_000001  Comme indiqué dans
              writer000000_eval2011-0_000002  habitation n° DVT 36
              writer000000_eval2011-0_000003  de mon domicile
      Output: writer000000_eval2011-0 Comme indiqué dans habitation n° DVT 36 de mon domicile
"""

import argparse
import os
import io
import sys
### main ###
infile = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
output = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

paragraph_txt_dict = dict()
for line in infile:
  line_vect = line.strip().split(' ')
  line_id = int(line_vect[0].split('_')[-1])
  paragraph_id = line_vect[0].split('-')[-1]
  paragraph_id = int(paragraph_id.split('_')[0])
  line_text = " ".join(line_vect[1:])
  if paragraph_id not in paragraph_txt_dict.keys():
      paragraph_txt_dict[paragraph_id] = dict()
  paragraph_txt_dict[paragraph_id][line_id] = line_text


para_txt_dict = dict()
for para_id in sorted(paragraph_txt_dict.keys()):
    para_txt = ""
    for line_id in sorted(paragraph_txt_dict[para_id]):
        text = paragraph_txt_dict[para_id][line_id]
        para_txt = para_txt + " " + text
    para_txt_dict[para_id] = para_txt
    utt_id = 'writer' + str(para_id).zfill(6) + '_' + 'eval2011-' + str(para_id)
    output.write(utt_id + ' ' + para_txt + '\n')
