#!/usr/bin/env python

lexicon = open("data/local/tmp/lexicon.1")
lexicon_ext = open("data/local/tmp/lexicon.2", "w")

for line in lexicon:
    line = line.strip()
    if "//" not in line:
        lexicon_ext.write(line + "\n")
        continue
    lineComp = line.split("\t")
    prons = lineComp[1].split("//")
    for item in prons:
        lexicon_ext.write(lineComp[0] + "\t" + item + "\n")

lexicon.close()
lexicon_ext.close()
    

