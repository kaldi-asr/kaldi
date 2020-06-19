#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import sys
import re
import writenumbers
from string import maketrans

## Global vars

normdict = {".": "",
            ",": "",
            ":": "",
            ";": "",
            "?": "",
            "\\": " ",
            "\t": " "
            }

from_chars = ''.join(list(normdict.keys()))
to_chars = ''.join(list(normdict.values()))

#t_table = maketrans(from_chars, to_chars)


## Main

numtable = writenumbers.loadNumTable(sys.argv[1])
transcript = codecs.open(sys.argv[2], "r", "utf8")
outtext = codecs.open(sys.argv[3], "w", "utf8")


for line in transcript:
    normtext1 = re.sub(r'[\.,:;\?]', '', line)
    normtext2 = re.sub(r'[\t\\]', ' ', normtext1)
    normtext3 = re.sub(r'  +', ' ', normtext2.strip())
    normtext4 = writenumbers.normNumber(normtext3, numtable)
    outtext.write(normtext4)

transcript.close()
outtext.close()
