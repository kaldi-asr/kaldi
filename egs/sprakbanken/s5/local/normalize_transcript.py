import codecs
import sys
import re

## Global vars

normdict = {#".": "",
            ",": " ",
            ":": " ",
            ";": " ",
            "?": " ",
            "\\": " ",
            "\t": " "
    }

t_table = str.maketrans(normdict)

## Utility function

def getuttid_text(line):
    return line.split(" ", 1)


## Main

textin = codecs.open(sys.argv[1], "r", "utf8")
outtext = codecs.open(sys.argv[2], "w", "utf8")

if len(sys.argv) > 3:
    make_wlist = True
    wlist = codecs.open(sys.argv[3], "w", "utf8")
else:
    make_wlist = False

for line in textin:
    utt_id, text = getuttid_text(line)
    normtext1 = text.translate(t_table)
    
    normtext2 = re.sub(r'  +', ' ', normtext1.strip())

    outtext.write(utt_id+ " " +normtext2+ "\n")
    if make_wlist:
        wlist.write(normtext2+ "\n")

if make_wlist:
    wlist.close()

textin.close()
outtext.close()    
