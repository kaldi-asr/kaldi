#!/usr/bin/env python

# Copyright 2017 Tsinghua University (Author: Zhiyuan Tang)
# Apache 2.0

import sys, collections, os

dir = sys.argv[1]

if not os.path.isdir(r'exp/olr_ali'):
	os.makedirs(r'exp/olr_ali')  # store the alignment of language id

len_dict = collections.OrderedDict()
with open(dir + '/feats.len', 'r') as utt_lens:
        for utt_len in [line.strip().split(' ') for line in utt_lens]:
                len_dict[utt_len[0]] = utt_len[1]

counts = []
for n in range(0, 10):  # 10 languages
	counts.append(0)

lang_dict = {'ct-cn':'0', 'id-id':'1', 'ja-jp':'2', 'ko-kr':'3', 'ru-ru':'4', 'vi-vn':'5', 'zh-cn':'6', 'Kazak':'7', 'Tibet':'8', 'Uyghu':'9'} # each language with an ID

lang_ali = open('exp/olr_ali/ali.ark', 'w')
for i in len_dict.keys():
        line_id = i
	lang_id = lang_dict[line_id[0:5]]
        num = int(len_dict[i])
	counts[int(lang_id)] += num
	line_id += (' ' + lang_id) * num
        lang_ali.write(line_id + '\n')
lang_ali.close()

lang_counts = open('exp/olr_ali/frame_counts', 'w') # total frames for each language
lang_counts.write('[')
for j in counts:
        lang_counts.write(' ' + str(j))
lang_counts.write(' ]')
lang_counts.close()

lang_num = open('exp/olr_ali/lang_num', 'w')
lang_num.write('10') # 10 languages
lang_num.close()

print 'Language alignment stored in exp/olr_ali.'
