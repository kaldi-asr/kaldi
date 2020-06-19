#!/usr/bin/env python3

""" This script normalizes a text file. It performs following normalizations:
    dots/filled-circles to periods, # variuos dashes to regular hyphen, full
    width left/right-paren to regular left/right paren.
    Eg. normalize_common.py <input-file> <output-file>
"""
import sys
from snor import SnorIter

if len(sys.argv) != 3:
    print("Usage: normalize_common.py <input-file> <output-file>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

def main():

    with open(input_file, 'r', encoding='utf-8') as fh, open(output_file, 'w', encoding='utf-8') as fh_out:
        for utt, uttid in SnorIter(fh):
            for char in utt:
                if char == "\u25cf" or char == "\u2022" or char == "\u2219":
                    # Convert "dots"/"filled-circles" to periods
                    fh_out.write("\u002e")
                elif char == "\u2010" or char == "\u2011" or char == "\u2012" or char == "\u2013" or char == "\u2014" or char == "\u2015":
                    # Change variuos Unicode dashes to Reular hyphen
                    fh_out.write("\u002d")
                elif char == "\uff09":
                    # Change Full width right-paren to regular paren
                    fh_out.write("\u0029")
                elif char == "\uff08":
                    # Change Full width left-paren to regular paren
                    fh_out.write("\u0028")
                else:
                    # Otherwise just apapend char w/o modification
                    fh_out.write(char)

            # Finally, print out uttid and newline
            fh_out.write(" (%s)\n" % uttid)


if __name__ == "__main__":
    main()
