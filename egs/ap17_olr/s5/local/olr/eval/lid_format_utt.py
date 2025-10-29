#!/usr/bin/env python

import sys, os, os.path

data_base = sys.argv[1]

lang_dict = {'ct-cn':'lang0', 'id-id':'lang1', 'ja-jp':'lang2', 'ko-kr':'lang3', 'ru-ru':'lang4', 'vi-vn':'lang5', 'zh-cn':'lang6', 'Kazak':'lang7', 'Tibet':'lang8', 'Uyghu':'lang9'}


if not os.path.isdir(r'lid_score'):
        os.makedirs(r'lid_score')

dir = 'lid_net_output'
for parent,dirnames,filenames in os.walk(dir):
        for file in filenames:
                if file.endswith('ark.utt_average') and data_base in file:
			file_path = 'lid_net_output/' + file
			new_path = 'lid_score/' + file
			new_score = open(new_path, 'w')
			new_score.write('      lang0    lang1    lang2    lang3    lang4    lang5    lang6    lang7    lang8    lang9 \n')
			
			with open(file_path, 'r') as lines:
				for col in [line.strip().split() for line in lines]:
					line_id = col[0]
					lang_id = lang_dict[line_id[0:5]]
					new_score.write(lang_id + ' ' + ' '.join(col[2:12]) + '\n')
			
			new_score.close()
                        print "Utter level format for " + data_base + " done."
