#!/usr/bin/env python3

""" This script normalizes a text file. It performs following normalizations:
    multiple continuous spaces to single space, removes spaces at the begining
    and end of the word.
    Eg. normalize_spaces.py <input-file> <output-file>
"""
import sys
from snor import SnorIter

if len(sys.argv) != 3:
    print("Usage: normalize_spaces.py <input-file> <output-file>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

def main():

    with open(input_file, 'r', encoding='utf-8') as fh, open(output_file, 'w', encoding='utf-8') as fh_out:
        for utt, uttid in SnorIter(fh):
            # Only output one space at a time
            space_chars = set([" ", "\t", "\u00a0"])

            last_char_was_space = False

            # Strip spaces at beginning and end of utterance
            utt = utt.strip(' ')
            for char in utt:
                if char in space_chars:
                    if not last_char_was_space:
                        fh_out.write(" ")
                    last_char_was_space = True
                else:
                    fh_out.write(char)
                    last_char_was_space = False

            # Finally, print out uttid and newline
            fh_out.write(" (%s)\n" % uttid)

if __name__ == "__main__":
    main()
