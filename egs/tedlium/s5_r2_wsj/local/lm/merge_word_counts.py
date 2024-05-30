#! /usr/bin/env python

# Copyright 2016    Vimal Manohar
# Apache 2.0.

"""This script merges pocolm word_counts and write a new word_counts file.
A min-count argument is required to only write counts that are above the
specified minimum count.
"""
from __future__ import print_function

import sys


def main():
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: {0} <min-count>\n".format(sys.argv[0]))
        raise SystemExit(1)

    words = {}
    for line in sys.stdin.readlines():
        parts = line.strip().split()
        words[parts[1]] = words.get(parts[1], 0) + int(parts[0])

    for word, count in words.items():
        if count >= int(sys.argv[1]):
            print ("{0} {1}".format(count, word))


if __name__ == '__main__':
    main()
