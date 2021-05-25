#!/usr/bin/env python3

import argparse
import re


def parse_opts():
    parser = argparse.ArgumentParser(
        description='This',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Debug info - e.g. showing the lines that were stripped')

    parser.add_argument('in_text', type=str, help='Input text file')
    parser.add_argument('out_text', type=str, help='Filtered output text file')
    opts = parser.parse_args()
    global debug
    debug = opts.debug
    return opts


if __name__ == '__main__':
    opts = parse_opts()

    with open(opts.in_text, 'r', encoding="utf-8") as in_text:
        in_lines = [l.strip() for l in in_text.readlines()]

    out_text = open(opts.out_text, 'w', encoding="utf-8")

    cnt = 0
    for i, l in enumerate(in_lines):
        if len(l) == 0:
            continue

        if len(l) > 20:
            print("len(l) > 20: %s" % l)

        m1 = re.match("^[a-zA-Z]+[\'\-]?[a-zA-Z\']*$", l)
        m2 = re.match("^[a-zA-Z]+(\-[a-zA-Z\']+)*$", l)  # "brother-in-law" "brother-in-law's"
        if m1 is None and m2 is None:
            print(l)
            cnt += 1
        else:
            print(l, file=out_text)

    out_text.close()
    print("total: %d" % cnt)
