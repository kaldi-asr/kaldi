#!/usr/bin/env py

# Converts a romanized ECA word list (symbol table) to
# a version in the arabic script

import sys
import codecs

if len(sys.argv) < 3:
    print "USAGE: local/convert_symtable_to_utf.py [SYMTABLE] [ECA-LEXICON]"
    print "E.g., local/convert_symtable_to_utf.py data/lang/words.txt \
                /export/corpora/LDC/LDC99L22"
    sys.exit(1)

# Note that the ECA lexicon's default encoding is ISO-8859-6, not UTF8
symtable = codecs.open(sys.argv[1], encoding="utf8")
lexicon = codecs.open(sys.argv[2] + "/callhome_arabic_lexicon_991012/ar_lex.v07", encoding="iso-8859-6")

dict_cache = {}
# First read off the dictionary and store stuff in a cache
for line in lexicon:
    line = line.strip().split()
    roman = line[0].strip()
    script = line[1].strip()
    assert roman not in dict_cache
    dict_cache[roman] = script

# Now read the symbol table and write off the ut8 versions
for line in symtable:
    line = line.strip().split()
    if line[0] in dict_cache:
        output = dict_cache[line[0]] + " " + line[1]
    else:
        output = line[0] + " " + line[1]
    sys.stdout.write(output.encode("utf-8") + "\n")

lexicon.close()
symtable.close()
