#!/usr/bin/env python
import codecs
import sys
import re
#import writenumbers


## Global vars

normdict = {".": "",
            ",": "",
            ":": "",
            ";": "",
            "?": "",
            "!": "",
            "\\": " ",
            "\t": " "
            }
#removes all the above signs

from_chars = ''.join(list(normdict.keys()))
to_chars = ''.join(list(normdict.values()))

t_table = str.maketrans(normdict)

## Main

transcript = codecs.open(sys.argv[1], "r", "utf8")
outtext = codecs.open(sys.argv[2], "w", "utf8")

#TODO: Add number normalisation and remove uppercasing

for line in transcript:
    line = line.replace(".\Punkt", ".")
    line = line.replace(",\Komma", ",")
    normtext1 = re.sub(r'[\.,:;\?]', '', line)
    normtext2 = re.sub(r'[\t\\]', ' ', normtext1)
    normtext3 = re.sub(r'  +', ' ', normtext2.strip())
    outtext.write(normtext3.upper() + "\n")

transcript.close()
outtext.close()
