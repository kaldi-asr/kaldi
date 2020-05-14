#!/usr/bin/env python3

# Copyright 2018-2020  Daniel Povey
#           2018-2020  Yiming Wang
# Apache 2.0

""" This script adds prefix to utt-id for entries in scp files.
"""


import argparse
import re
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="""Adds augmentation prefix to
        original utterance ids to obtain <new_id> <value> entry""")
    parser.add_argument('--prefix', nargs='+',
                        help='prefix to add to each utterance id')
    args = parser.parse_args()

    for line in sys.stdin:
        line = line.strip()
        id, val = line.split(None, 1)
        if re.match(r'^sp\d\.\d-', id):
            pass
        else:
            for prefix in args.prefix:
                print(prefix + '-' + id + ' ' + val)


if __name__ == "__main__":
    main()


