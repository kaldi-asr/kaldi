#!/usr/bin/env python3

# Copyright  2018  Ashish Arora
#                  Chun-Chieh Chang

import argparse
import os

parser = argparse.ArgumentParser(description="""Creates the list of characters and words in lexicon""")
parser.add_argument('dir', type=str, help='output path')
parser.add_argument('--data-dir', type=str, default='data', help='Path to text file')
args = parser.parse_args()

### main ###
radical = ['日', '月', '金', '木', '水', '火', '土', '竹', '戈', '十', '大', '中', '一', '弓', '人', '心', '手','口','尸','廿','山','女','田','卜']
lex = {}
text_path = os.path.join(args.data_dir, 'train', 'text')
text_fh = open(text_path, 'r', encoding='utf-8')

# Used specially for Chinese.
# Uses the ChangJie keyboard input method to create subword units for Chinese.
cj5_table = {}
with open(os.path.join(args.dir, 'cj5-cc.txt'), 'r', encoding='utf-8') as f:
    for line in f:
        line_vect = line.strip().split()
        if not line_vect[0].startswith('yyy') and not line_vect[0].startswith('z'):
            cj5_table[line_vect[-1]] = "cj5_" + " cj5_".join(line_vect[:-1])
#            lex[line_vect[1]] = "cj5_" + " cj5_".join(list(line_vect[0]))

with open(text_path, 'r', encoding='utf-8') as f:
    for line in f:
        line_vect = line.strip().split()
        for i in range(1, len(line_vect)):
            characters = list(line_vect[i])
	    # Put SIL instead of "|". Because every "|" in the beginning of the words is for initial-space of that word
            characters = " ".join([ 'SIL' if char == '|' else char if char in radical else cj5_table[char] if char in cj5_table else char for char in characters])
            characters = characters.replace('#','<HASH>')
            lex[line_vect[i]] = characters

with open(os.path.join(args.dir, 'lexicon.txt'), 'w', encoding='utf-8') as fp:
    for key in sorted(lex):
        fp.write(key + " " + lex[key] + "\n")
