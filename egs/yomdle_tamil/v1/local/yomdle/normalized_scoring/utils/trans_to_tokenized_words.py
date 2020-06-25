#!/usr/bin/env python3

""" This script splits punctuations, digits and currency symbols
    from the word. 
    Eg. "They have come!" he said reverently, gripping his
        " They have come ! " he said reverently , gripping his
    Eg. trans_to_tokenized_words.py <input-file> <output-file>
"""

import unicodedata
import sys
from snor import SnorIter

if len(sys.argv) != 3:
    print("Usage: trans_to_tokenized_words.py <input-file> <output-file>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]


punc =  set(chr(i) for i in range(sys.maxunicode)
            if unicodedata.category(chr(i)).startswith('P'))
currency_symbols =  set(chr(i) for i in range(sys.maxunicode)
                        if unicodedata.category(chr(i)) == "Sc")
digits =  set(chr(i) for i in range(sys.maxunicode)
                        if unicodedata.category(chr(i)) == "Nd")

split_punc = True
split_digits = True
def main():

    with open(input_file, 'r', encoding='utf-8') as fh, open(output_file, 'w', encoding='utf-8') as fh_out:
        for utt, uttid in SnorIter(fh):
            for char in utt:
                if (split_punc and char in punc) or (split_punc and char in currency_symbols) or (split_digits and char in digits):
                    fh_out.write(" ")
                    fh_out.write(char)
                    fh_out.write(" ")
                else:
                    fh_out.write(char)

            # Finally write out uttid and newline
            fh_out.write(" (%s)\n" % uttid)



if __name__ == "__main__":
    main()
