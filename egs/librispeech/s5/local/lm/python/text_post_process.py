#!/usr/bin/env python

# Copyright 2014 Vassil Panayotov
# Apache 2.0

# [This script was taken verbatim from the LibriVox alignment setup]

# Post processes the .opl file produced by 'nsw_expand':
# - removes the non-word tokens
# - corrects likely wrong normalizations (e.g. Sun -> Sunday)
# - splits the sentences into separate lines

import sys, argparse
import re

def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-process an .opl file into plain text")
    parser.add_argument('--max-sent-len', type=int, default=600,
                        help="The maximum allowed # of words per sentence")
    parser.add_argument('--abort-long-sent', type=bool, default=False,
                        help='If True and a sentence longer than "max-sent-len" detected' +\
                             'exit with error code 1. If False, just split the long sentences.')
    parser.add_argument('--sent-end-marker', default="DOTDOTDOT")
    parser.add_argument("in_text", help="Input text")
    parser.add_argument("out_text", help="Output text")
    parser.add_argument("sent_bounds",
                        help="A file that will contain a comma separated list of numbers, s.t. if" +
                             "i is in this list, then there is a sententence break after token i")
    return parser.parse_args()


if __name__ == '__main__':
    opts = parse_args()
    with open(opts.in_text) as src, \
         open(opts.out_text, 'w') as dst, \
         open(opts.sent_bounds, 'w') as bounds:
        corrections = 0
        lines = list()
        current_line = list()
        sent_bounds = list()
        n_tokens = 0
        for opl_line in src:
            start_scan = 3
            opl_line = opl_line.upper()
            opl_tokens = opl_line.split()
            if opl_tokens[0] == opts.sent_end_marker.upper():
                sent_bounds.append(n_tokens - 1)
                if len(current_line) > opts.max_sent_len:
                    if opts.abort_long_sent:
                        sys.stderr.write('ERROR: Too long sentence - aborting!\n')
                        sys.exit(1)
                    else:
                        sys.stderr.write('WARNING: Too long sentence - splitting ...\n')
                        sent_start = 0
                        while sent_start < len(current_line):
                            lines.append(' '.join(current_line[sent_start:\
                                                  sent_start + opts.max_sent_len]))
                            sent_start += opts.max_sent_len
                else:
                    lines.append(' '.join(current_line))
                current_line = list()
                continue
            if len(opl_tokens) >= 4 and opl_tokens[3] == 'SUNDAY' and opl_tokens[1] == 'EXPN':
                corrections += 1
                n_tokens += 1
                start_scan = 4
                current_line.append('SUN')
            for i in range(start_scan, len(opl_tokens)):
                m = re.match("^[A-Z]+\'?[A-Z\']*$", opl_tokens[i])
                if m is not None:
                    n_tokens += 1
                    current_line.append(opl_tokens[i])
                #else:
                #    sys.stderr.write('rejected: %s\n' % opl_tokens[i])
        sys.stderr.write('Corrected tokens: %d\n' % corrections)
        dst.write('\n'.join(lines) + '\n')
        bounds.write(','.join([str(t) for t in sent_bounds]))
