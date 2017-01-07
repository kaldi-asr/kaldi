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

t_table = str.maketrans(normdict)

## Main

transcript = codecs.open(sys.argv[1], "r", "utf8")
outtext = codecs.open(sys.argv[2], "w", "utf8")

for line in transcript:
	line = line.replace(".\Punkt", ".")
	line = line.replace(",\Komma", ",")
	normtext1 = line.translate(t_table)
	normtext2 = re.sub(r'  +', ' ', normtext1.strip())
	outtext.write(normtext2.upper() + "\n")


transcript.close()
outtext.close()
