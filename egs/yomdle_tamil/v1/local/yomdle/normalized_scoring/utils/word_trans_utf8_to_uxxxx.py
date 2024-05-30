#!/usr/bin/env python3

""" This script converts characters from utf-8 format to hexadecimal format.
    Eg. word_trans_utf8_to_uxxxx.py <input-file> <output-file>
"""

import sys
from snor import SnorIter

if len(sys.argv) != 3:
    print("Usage: word_trans_utf8_to_uxxxx.py <input-file> <output-file>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

def main():
    with open(input_file, 'r', encoding='utf-8') as fh, open(output_file, 'w', encoding='utf-8') as fh_out:
        for utt, uttid in SnorIter(fh):
            for word in utt.split():
                fh_out.write(utf8_char_to_uxxxx(word[0]))
                for char in word[1:]:
                    fh_out.write("_")
                    fh_out.write(utf8_char_to_uxxxx(char))
                fh_out.write(" ")
            # Finally write out uttid and newline
            fh_out.write("(%s)\n" % uttid)


def utf8_char_to_uxxxx(char):
    raw_hex = hex(ord(char))[2:].zfill(4).lower()
    uxxxx_char = "u%s" % raw_hex
    return uxxxx_char

if __name__ == "__main__":
    main()
