#!/usr/bin/env python

# Copyright 2018 Johns Hopkins University (author: Yiming Wang)
# Apache 2.0

""" This script prepares the speech commands data into kaldi format.
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
            #head, remainder = id.split('-', 1)
            #for prefix in args.prefix:
            #    print(head + '-' + prefix + '_' + remainder + ' ' + val)
        else:
            for prefix in args.prefix:
                if prefix.startswith('rev'):
                    print(prefix + '-' + id + ' ' + val)
                else:
                    print(prefix + '_' + id + ' ' + val)


if __name__ == "__main__":
    main()


