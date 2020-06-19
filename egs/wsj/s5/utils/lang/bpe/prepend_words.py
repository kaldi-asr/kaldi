#!/usr/bin/env python3

# This script, prepend '|' to every words in the transcript to mark
# the beginning of the words for finding the initial-space of every word
# after decoding.

import sys
import io
import re

whitespace = re.compile("[ \t]+")
infile = io.TextIOWrapper(sys.stdin.buffer, encoding='latin-1')
output = io.TextIOWrapper(sys.stdout.buffer, encoding='latin-1')
for line in infile:
    words = whitespace.split(line.strip(" \t\r\n"))
    output.write(' '.join([ "|"+word for word in words]) + '\n')
