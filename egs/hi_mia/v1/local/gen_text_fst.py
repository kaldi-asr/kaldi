#!/usr/bin/env python3
# Copyright 2020 Audio, Speech and Language Processing Group (ASLP@NPU), Northwestern Polytechnical University (Authors: Zhuoyuan Yao, Xiong Wang, Jingyong Hou, Lei Xie)
#           2020 AIShell-Foundation (Author: Bengu WU) 
#           2020 Beijing Shell Shell Tech. Co. Ltd. (Author: Hui BU) 
# Apache 2.0
#
# Generate the fst file from keyword phone file

import sys
import codecs
import argparse

FLAGS = None

def read_keyword_phone_file(file_name):
    keywords = {} 
    fid = codecs.open(file_name, 'r', 'utf-8')
    for line in fid.readlines():
        arr = line.strip().split()
        assert(len(arr) >= 2)
        keywords[arr[0]] = arr[1:]
    fid.close()
    return keywords

# Format <source state> <dest state> <ilabel> <olabel> <<weight|optional>
# <final state> <final weight>
def build_text_graph(keyword_phone_file, text_fst_file):
    keyword_phones = read_keyword_phone_file(keyword_phone_file)
    fout = codecs.open(text_fst_file, 'w', 'utf-8')
    # start to silence/filler
    fout.write('0 1 sil <eps>\n')
    fout.write('0 2 <GBG> <gbg>\n')
    # silence to silence/filler
    fout.write('1 1 sil <eps>\n')
    fout.write('1 2 <GBG> <gbg>\n')
    # filler to silence/filler
    fout.write('2 1 sil <eps>\n')
    fout.write('2 2 <GBG> <gbg>\n')
    #final_state = 3

    cur_state = 3
    for keyword, phones in keyword_phones.items():
        # start/silence/filler to keyword start
        fout.write('0 %d %s <eps>\n' % (cur_state, phones[0]))
        fout.write('1 %d %s <eps>\n' % (cur_state, phones[0]))
        fout.write('2 %d %s <eps>\n' % (cur_state, phones[0]))
        for i in range(0, len(phones)-1):
            # fout.write('%d %d sil <eps>\n' % (cur_state, cur_state))
            # fout.write('%d %d <GBG> <eps>\n' % (cur_state, cur_state))
            fout.write('%d %d %s <eps>\n' % (cur_state, cur_state, phones[i]))
            if not i == len(phones) - 2:
                fout.write('%d %d %s <eps>\n' % (cur_state, cur_state+1, phones[i+1]))
            else:
                fout.write('%d %d %s %s\n' % (cur_state, cur_state+1, phones[i+1], keyword))
            cur_state += 1
        fout.write('%d %d %s <eps>\n' % (cur_state, cur_state, phones[-1]))
        # make it last state of keyword to final state
        fout.write('%d 1.0\n' % cur_state)
        cur_state += 1
    fout.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate the text fst ')
    parser.add_argument('keyword_phone_file', help='keyword phone file')
    parser.add_argument('text_fst_file', help='text fst file')

    FLAGS = parser.parse_args()

    build_text_graph(FLAGS.keyword_phone_file, FLAGS.text_fst_file)
