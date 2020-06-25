#!/usr/bin/env python3

""" This script is used for partial scoring, it will remove the given
    utterance ids and will score with the remailing utterance ids.
    Eg. filter_ids.py <ids-to-filter> <input-trans> <output-trans>
"""

import unicodedata
import sys
from snor import SnorIter

if len(sys.argv) != 4:
    print("Usage: filter_ids.py <ids-to-filter> <input-trans> <output-trans>")
    sys.exit(1)

input_ids_file = sys.argv[1]
input_trans = sys.argv[2]
output_trans = sys.argv[3]

def main():

    # First load ids to filter out of transcript
    ids_to_filter = set()
    with open(input_ids_file, 'r') as fh:
        for line in fh:
            ids_to_filter.add(line.strip())

    # Now load input transcript and filter out the ids
    with open(input_trans, 'r', encoding='utf-8') as fh, open(output_trans, 'w', encoding='utf-8') as fh_out:
        for utt, uttid in SnorIter(fh):
            if uttid in ids_to_filter:
                continue

            fh_out.write("%s (%s)\n" % (utt, uttid))



if __name__ == "__main__":
    main()
