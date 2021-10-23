#!/usr/bin/env python3

# Copyright 2021 ASLP, NWPU (Author: Hang Lyu)
#
# This script splits the Chinese words into character and keep the English
# words for the MER evalutaion.

import sys
import codecs

def main():
    with codecs.open(sys.argv[1], 'r', 'utf-8') as fin:
        with codecs.open(sys.argv[2], 'w', 'utf-8') as fout:
            for line in fin:
                words = line.strip().split()
                line_new = words[0] + '\t'
                for word in words[1:]:
                    for char in word:
                        if 'A' <= char <= 'Z':
                            line_new += char
                        else:
                            line_new += char + ' '
                    line_new += ' '
                line_new = line_new.replace('  ', ' ')
                fout.write('%s\n' % line_new)

if __name__ == '__main__':
    main()
