#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2014  Gaurav Kumar.   Apache 2.0
#    2018  Nagendra Kumar Goel, Saikiran Valluri, GoVivace inc., Avaaya
# Merges unique words from Spanish Fisher, Gigaword and the LDC spanish lexicon
from __future__ import print_function
import sys, re
import json
import codecs
import operator

wordlimit = 64000
tmpdir = sys.argv[1]
ldc_lexicon = sys.argv[2]
uw_fisher = tmpdir + "/uniquewords"
uw_gigaword = tmpdir + "/es_wordlist.json"
uw_LDC = ldc_lexicon + "/callhome_spanish_lexicon_970908/preferences"

filtered_letters = re.compile(u'[¡¥ª°º¿àçèëìîôö0123456789]')
merged_lexicon = []
# All three lexicons are in different formats
# First add the data from lexicon_fisher (A) into the dictionary
fisher = codecs.open(uw_fisher, encoding='utf-8')
for line in fisher:
    merged_lexicon.append(line.strip())
fisher.close()

print("After adding the fisher data, the lexicon contains {} entries".format(len(merged_lexicon)))

# Now add data from the LDC lexicon
ldc = codecs.open(uw_LDC, encoding='iso-8859-1')
for line in ldc:
    entries = line.strip().split('\t')
    if entries[0].lower() not in merged_lexicon:
        merged_lexicon.append(entries[0].lower())

print("After adding the LDC data, the lexicon contains {} entries".format(len(merged_lexicon)))

# Finally add the gigaword data
gigaword = json.load(open(uw_gigaword))
gigaword = reversed(sorted(gigaword.items(), key=operator.itemgetter(1)))

for item in gigaword:
    # We need a maximum of wordlimit words in the lexicon
    if len(merged_lexicon) == wordlimit:
        break

    if item[0].lower() not in merged_lexicon:
        merged_lexicon.append(item[0].lower())

print("After adding the Gigaword data, the lexicon contains {} entries".format(len(merged_lexicon)))

# Now write the uniquewords to a file
lf = codecs.open(tmpdir + '/uniquewords64k', encoding='utf-8', mode='w+')
ltuples = sorted(merged_lexicon)

for item in ltuples:
    if not item==u'ñ' and not re.search(filtered_letters, item):
        lf.write(item + "\n")

lf.close()

print("Finshed writing unique words")
