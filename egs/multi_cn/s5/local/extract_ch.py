#!/usr/bin/env python

from __future__ import print_function

import sys
from builtins import str


def extract():
    #     print (sys.argv)
    cn_lines = []
    with open(sys.argv[1], "r") as txt:
        data = txt.read()
        index = 1
        for char in data:
            if char == "\n":
                index += 1
            elif not char.islower() and not char.isupper():
                if index not in cn_lines:
                    cn_lines.append(index)
    return cn_lines


if __name__ == "__main__":
    result = extract()
    print(str(result).strip("[").strip("]"))
