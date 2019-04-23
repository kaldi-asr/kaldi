#!/usr/bin/env python3

# 2019 Dongji Gao

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

from make_lexicon_fst import read_lexiconp
import argparse
import math

def get_args():
    parser = argparse.ArgumentParser(description="""This script creates a
        position-dependent subword lexicon from a position-independent subword lexicon
        by adding suffixes ("_B", "_I", "_E", "_S") to the related phones.
        It assumes that the input lexicon does not contain disambiguation symbols.""")
    parser.add_argument("--separator", type=str, default="@@", help="""Separator
        indicates the position of a subword in a word. 
        Subword ends with separator can only appear at the beginning or middle of a word. 
        Subword without separator can only appear at the end of a word or is a word itself.
        E.g. "international -> inter@@ nation@@ al";
             "nation        -> nation"
        The separator should match the separator used in the input lexicon.""")
    parser.add_argument("lexiconp", type=str, help="""Filename of subword position-independent 
        lexicon with pronunciation probabilities, with lines of the form 'subword prob p1 p2 ...'""")
    args = parser.parse_args()
    return args

def is_end(subword, separator):
    """Return true if the subword can appear at the end of a word (i.e., the subword 
    does not end with separator). Return false otherwise."""
    return not subword.endswith(separator)

def write_position_dependent_lexicon(lexiconp, separator):
    """Print a position-dependent lexicon for each subword from the input lexiconp by adding
    appropriate suffixes ("_B", "_I", "_E", "_S") to the phone sequence related to the subword.
    There are 4 types of position-dependent subword:
    1) Beginning subword. It can only appear at the beginning of a word.
       The first phone suffix should be "_B" and other suffixes should be "_I"s:
        nation@@ 1.0 n_B ey_I sh_I ih_I n_I
        n@@      1.0 n_B
    2) Middle subword. It can only appear at the middle of a word.
       All phone suffixes should be "_I"s:
        nation@@ 1.0 n_I ey_I sh_I ih_I n_I
    3) End subword. It can only appear at the end of a word.
       The last phone suffix should be "_E" and other suffixes should be "_I"s:
        nation   1.0 n_I ey_I sh_I ih_I n_E
        n        1.0 n_E
    4) Singleton subword (i.e., the subword is word it self). 
       The first phone suffix should be "_B" and the last suffix should be "_E".
       All other suffixes should be "_I"s. If there is only one phone, its suffix should be "_S":
        nation   1.0 n_B ey_I sh_I ih_I n_E
        n        1.0 n_S
    In most cases (i.e., subwords have more than 1 phones), the suffixes of phones in the middle are "_I"s.
    So the suffix_list is initialized with all _I and we only replace the first and last phone suffix when
    dealing with different cases when necessary.
    """
    for (word, prob, phones) in lexiconp:
        phones_length = len(phones)

        # suffix_list is initialized by all "_I"s.
        suffix_list = ["_I" for i in range(phones_length)]

        if is_end(word, separator):
            # print end subword lexicon by replacing the last phone suffix by "_E"
            suffix_list[-1] = "_E"
            phones_list = [phone + suffix for (phone, suffix) in zip(phones, suffix_list)]
            print("{} {} {}".format(word, prob, ' '.join(phones_list)))

            # print singleton subword lexicon
            # the phone suffix is "_S" if the there is only 1 phone.
            if phones_length == 1:
                suffix_list[0] = "_S"
                phones_list = [phone + suffix for (phone, suffix) in zip(phones, suffix_list)]
                print("{} {} {}".format(word, prob, ' '.join(phones_list)))
            # the first phone suffix is "_B" is there is more than 1 phones.
            else:
                suffix_list[0] = "_B"
                phones_list = [phone + suffix for (phone, suffix) in zip(phones, suffix_list)]
                print("{} {} {}".format(word, prob, ' '.join(phones_list)))
        else:
            # print middle subword lexicon
            phones_list = [phone + suffix for (phone, suffix) in zip(phones, suffix_list)]
            print("{} {} {}".format(word, prob, ' '.join(phones_list)))

            # print beginning subword lexicon by replacing the first phone suffix by "_B"
            suffix_list[0] = "_B"
            phones_list = [phone + suffix for (phone, suffix) in zip(phones, suffix_list)]
            print("{} {} {}".format(word, prob, ' '.join(phones_list)))

def main():
    args = get_args()
    lexiconp = read_lexiconp(args.lexiconp)
    write_position_dependent_lexicon(lexiconp, args.separator)

if __name__ == "__main__":
    main()
