#!/usr/bin/env python3

# This script, prepend '|' to every words in the transcript to mark
# the beginning of the words for finding the initial-space of every word
# after decoding.

import argparse
import sys, io

parser = argparse.ArgumentParser(description="Prepends '|' to the beginning of every word")
parser.add_argument('--encoding', type=str, default='latin-1',
                    help='Type of encoding')
args = parser.parse_args()

infile = io.TextIOWrapper(sys.stdin.buffer, encoding=args.encoding)
output = io.TextIOWrapper(sys.stdout.buffer, encoding=args.encoding)
for line in infile:
    output.write(' '.join([ "|"+word for word in line.split()]) + '\n')


