#!/usr/bin/env python3

import argparse
import json
import os
import numpy as np
import sys
from glob import glob
def get_args():
    parser = argparse.ArgumentParser(
        """Extract noises from the corpus based on the non-speech regions.""".format(sys.argv[0]))
    parser.add_argument("data_dir", help="""Location of the anscriptions. e.g. /export/corpora4/ranscriptions/train/""")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    segment_handle = open(os.path.join(args.data_dir,'vad.txt'), 'r')
    segment_data = segment_handle.read().strip().split("\n")
    overlap_dict={}
    for line in segment_data:
        count_0=0
        count_1=0
        parts = line.strip().split()
        uttid=parts[0].strip()
        overlap_info=parts[2:-1]
        for val in overlap_info:
            if int(val) == 1:
                count_1=count_1+1
            else:
                count_0=count_0+1
        overlap_dict[uttid]=overlap_info
        print(uttid, int(count_1*100/2000))

if __name__ == '__main__':
    main()
