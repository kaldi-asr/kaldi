#!/usr/bin/env python3

# Copyright  2017  Jian Wang
# License: Apache 2.0.

import io
import os
import argparse
import sys

import re
tab_or_space = re.compile('[ \t]+')

parser = argparse.ArgumentParser(description="This script checks whether the special symbols "
                                 "appear in words.txt with expected values, if not, it will "
                                 "print out the options with correct value to stdout, which may look like "
                                 "'--bos-symbol=14312 --eos-symbol=14313 --brk-symbol=14320'.",
                                 epilog="E.g. " + sys.argv[0] + " < exp/rnnlm/config/words.txt > exp/rnnlm/special_symbol_opts.txt",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

args = parser.parse_args()

# this dict stores the special_symbols and their corresponding (expected_id, option_name)
special_symbols = {'<s>':   (1, '--bos-symbol'),
                   '</s>':  (2, '--eos-symbol'),
                   '<brk>': (3, '--brk-symbol')}
upper_special_symbols = [key.upper() for key in special_symbols]

lower_ids = {}
upper_ids = {}
input_stream = io.TextIOWrapper(sys.stdin.buffer, encoding='latin-1')
for line in input_stream:
    fields = re.split(tab_or_space, line)
    assert(len(fields) == 2)
    sym = fields[0]
    if sym in special_symbols:
        assert sym not in lower_ids
        lower_ids[sym] = int(fields[1])
    elif sym in upper_special_symbols:
        assert sym.lower() not in upper_ids
        upper_ids[sym.lower()] = int(fields[1])

printed = False
for sym in special_symbols:
    if sym in lower_ids:
        if special_symbols[sym][0] != lower_ids[sym]:
            print('{0}={1} '.format(special_symbols[sym][1], lower_ids[sym]), end='')
            printed = True
        if sym in upper_ids:
            print(sys.argv[0] + ": both uppercase and lowercase are present for " + sym,
                  file=sys.stderr)
    elif sym in upper_ids:
        if special_symbols[sym][0] != upper_ids[sym]:
            print('{0}={1} '.format(special_symbols[sym][1], upper_ids[sym]), end='')
            printed = True
    else:
        raise ValueError("Special symbol is not appeared: " + sym)
        sys.exit(1)
if printed:
    print('')
