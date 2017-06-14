#!/usr/bin/env python
# Copyright 2014  Gaurav Kumar.   Apache 2.0
# -*- coding: utf-8 -*-
#
# Merges unique words from Spanish Fisher, Gigaword and the LDC spanish lexicon

import sys
import json
import codecs
import operator

wordlimit = 64000
tmpdir = sys.argv[1]
ldc_lexicon = sys.argv[2]
uw_fisher = tmpdir + "/uniquewords"
uw_gigaword = tmpdir + "/es_wordlist.json"
uw_LDC = ldc_lexicon + "/callhome_spanish_lexicon_970908/preferences"

merged_lexicon = []
# All three lexicons are in different formats
# First add the data from lexicon_fisher (A) into the dictionary
fisher = codecs.open(uw_fisher, encoding='utf-8')
for line in fisher:
    merged_lexicon.append(line.strip())
fisher.close()

print "After adding the fisher data, the lexicon contains " \
      + str(len(merged_lexicon)) + " entries."

# Now add data from the LDC lexicon
ldc = codecs.open(uw_LDC, encoding='iso-8859-1')
for line in ldc:
    entries = line.strip().split('\t')
    if entries[0].lower() not in merged_lexicon:
        merged_lexicon.append(entries[0].lower())

print "After adding the LDC data, the lexicon contains " \
      + str(len(merged_lexicon)) + " entries."

# Finally add the gigaword data
gigaword = json.load(open(uw_gigaword))
gigaword = reversed(sorted(gigaword.iteritems(), key=operator.itemgetter(1)))

for item in gigaword:
    # We need a maximum of wordlimit words in the lexicon
    if len(merged_lexicon) == wordlimit:
        break

    if item[0].lower() not in merged_lexicon:
        merged_lexicon.append(item[0].lower())

print "After adding the Gigaword data, the lexicon contains " \
      + str(len(merged_lexicon)) + " entries."

# Now write the uniquewords to a file
lf = codecs.open(tmpdir + '/uniquewords64k', encoding='utf-8', mode='w+')
ltuples = sorted(merged_lexicon)

for item in ltuples:
    lf.write(item + "\n")

lf.close()

print "Finshed writing unique words"
