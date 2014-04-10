import codecs
import sys

transcript = codecs.open(sys.argv[1], "r", "utf8")
outtext = codecs.open(sys.argv[2], "w", "utf8")


for line in transcript:
    outline = "<s> " +line.strip()+ " </s>\n"
    outtext.write(outline)


transcript.close()
outtext.close()    
