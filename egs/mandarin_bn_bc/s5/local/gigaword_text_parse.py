#!/usr/bin/env/python

# Apache 2.0

from __future__ import print_function
import io
import sys

if __name__ == '__main__':
  input_stream = io.TextIOWrapper(sys.stdin.buffer,encoding='utf-8')
  anker = False
  for line in input_stream.readlines():
    line = line.strip()
    if line == "<P>":
      anker = True
      continue
    elif line == "</P>":
      anker = False
    elif anker:
      print(line)
