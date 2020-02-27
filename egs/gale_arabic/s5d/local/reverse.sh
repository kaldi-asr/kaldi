#!/usr/bin/env python

import sys

text = sys.argv[1]
output = sys.argv[2]

with open(text, 'r') as text:
    with open(output, 'w') as output:
        for line in text.readlines():
            sentence = list()
            line_list = line.split()
            line_list.reverse()
            for word in line_list:
                sentence.append("".join(reversed(word)))
            output.write(" ".join(sentence)+"\n")
