import codecs
import sys
import re
import writenumbers


## Global vars

normdict = {".": "",
            ",": "",
            ":": "",
            ";": "",
            "?": "",
            "\\": " ",
            "\t": " "
            }

t_table = str.maketrans(normdict)


## Main

numtable = writenumbers.loadNumTable(sys.argv[1])
transcript = codecs.open(sys.argv[2], "r", "utf8")
outtext = codecs.open(sys.argv[3], "w", "utf8")


for line in transcript:
    normtext1 = line.translate(t_table)
    normtext2 = re.sub(r'  +', ' ', normtext1.strip())
    normtext3 = writenumbers.normNumber(normtext2, numtable)
    outtext.write(normtext3.upper())

transcript.close()
outtext.close()
