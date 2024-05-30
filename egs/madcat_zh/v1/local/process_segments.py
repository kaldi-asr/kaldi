#!/usr/bin/env python3

# Copyright     2017 Chun Chieh Chang

""" This script reads the provided word segmentations of chinese
    and ensures that all of them are normalized to the same
    unicode form.
"""

import argparse
import os
import unicodedata

parser = argparse.ArgumentParser(description="""Takes in word segmentations and normalizes character form.""")
parser.add_argument('segmentation_path', type=str,
                    help='Path to chinese word segmentation')
parser.add_argument('out_dir', type=str,
                    help='Where to write output file')
args = parser.parse_args()

segment_file = os.path.join(args.out_dir, 'segmented_words')
segment_fh = open(segment_file, 'w', encoding='utf-8')

with open(args.segmentation_path, encoding='utf-8') as f:
    for line in f:
        line_normalize = unicodedata.normalize('NFKC', line)
        segment_fh.write(line_normalize + '\n')
