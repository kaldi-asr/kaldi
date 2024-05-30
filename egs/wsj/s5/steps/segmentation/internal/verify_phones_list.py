#!/usr/bin/env python

# Copyright 2017  Vimal Manohar
# Apache 2.0

"""This script verifies the list of phones read from stdin are valid
phones present in lang/phones.txt."""

import argparse
import sys

def get_args():
    parser = argparse.ArgumentParser(description="""
    This script verifies the list of phones read from stdin are valid
    phones present in lang/phones.txt.""")

    parser.add_argument("phones", type=str,
                        help="File containing the list of all phones as the "
                        "first column")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    phones = set()
    for line in open(args.phones):
        phones.add(line.strip().split()[0])

    for line in sys.stdin.readlines():
        p = line.strip()

        if p not in phones:
            sys.stderr.write("Could not find phone {p} in {f}"
                             "\n".format(p=p, f=args.phones))
            raise SystemExit(1)


if __name__ == "__main__":
    main()
