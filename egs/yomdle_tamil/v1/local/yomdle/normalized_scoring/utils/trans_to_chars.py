#!/usr/bin/env python3

""" This script converts the words into sequence of space separated characters.
    It also converts space between words into "<sp> "
    Eg. trans_to_chars.py <input-file> <output-file>
"""

import unicodedata
import sys
from snor import SnorIter

if len(sys.argv) != 3:
    print("Usage: trans_to_chars.py <input-file> <output-file>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

def main():
    with open(input_file, 'r', encoding='utf-8') as fh, open(output_file, 'w', encoding='utf-8') as fh_out:
        for utt, uttid in SnorIter(fh):
            for char in utt:
                if char == " ":
                    fh_out.write("<sp> ")
                else:
                    fh_out.write(char)
                    fh_out.write(" ")
            # Finally write out uttid and newline
            fh_out.write("(%s)\n" % uttid)


if __name__ == "__main__":
    main()
