#!/usr/bin/env python3

""" This script finds and prints the hypothesis utterance ids which
    are not present in the reference utterance ids.
    Eg. find_missing_hyp_ids.py <ref-file> <hyp-file>
"""

import sys
from snor import SnorIter

if len(sys.argv) != 3:
    print("Usage: find_missing_hyp_ids.py <ref-file> <hyp-file>")
    sys.exit(1)

hyp_file = sys.argv[1]
ref_file = sys.argv[2]

def main():

    with open(hyp_file, 'r', encoding='utf-8') as hyp_fh, open(ref_file, 'r', encoding='utf-8') as ref_fh:
        ref_ids = set()
        for utt, uttid in SnorIter(ref_fh):
            ref_ids.add(uttid)

        for utt, uttid in SnorIter(hyp_fh):
            if uttid not in ref_ids:
                print(uttid)

if __name__ == "__main__":
    main()
