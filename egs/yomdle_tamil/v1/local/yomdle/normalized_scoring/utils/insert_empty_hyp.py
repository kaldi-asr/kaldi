#!/usr/bin/env python3

""" This script adds ids with empty utterance. It is used during scoring
    in cases where some of the reference ids are missing in the hypothesis.
    Eg. insert_empty_hyp.py <ids-to-insert> <in-hyp-file> <out-hyp-file>
"""

import sys
from snor import SnorIter

if len(sys.argv) != 4:
    print("Usage: insert_empty_hyp.py <ids-to-insert> <in-hyp-file> <out-hyp-file>")
    sys.exit(1)

ids_file = sys.argv[1]
hyp_in_file = sys.argv[2]
hyp_out_file = sys.argv[3]

def main():

    with open(hyp_in_file, 'r', encoding='utf-8') as hyp_in_fh, open(hyp_out_file, 'w', encoding='utf-8') as hyp_out_fh, open(ids_file, 'r') as ids_fh:
        # First just copy input hyp file over
        for line in hyp_in_fh:
            hyp_out_fh.write(line)

        # Now add missing ids

        for line in ids_fh:
            uttid = line.strip()
            hyp_out_fh.write("(%s)\n" % uttid)

if __name__ == "__main__":
    main()
