import codecs
import sys
import re

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

transcript = codecs.open(sys.argv[1], "r")
outtext = codecs.open(sys.argv[2], "w", "utf8")


for line in transcript:
    normtext1 = line.translate(t_table)
    
    normtext2 = re.sub(r'  +', ' ', normtext1.strip())

    outtext.write(normtext2.upper()+ "\n")

transcript.close()
outtext.close()    
