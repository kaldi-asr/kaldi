#!/usr/bin/env python

# Copyright 2015  Minhua Wu
# Apache 2.0

# convert acronyms in swbd transcript to fisher convention
# accoring to first two columns in the input acronyms mapping

import argparse,re
__author__ = 'Minhua Wu'
 
parser = argparse.ArgumentParser(description='format acronyms to a._b._c.')
parser.add_argument('-i','--input', help='Input transcripts',required=True)
parser.add_argument('-o','--output',help='Output transcripts', required=True)
parser.add_argument('-M','--Map', help='Input acronyms mapping',required=True)
args = parser.parse_args()

fin_map = open(args.Map, "r")
dict_acronym = {}
dict_acronym_noi = {} # Mapping of acronyms without I, i
for pair in fin_map:
    items = pair.split('\t')
    dict_acronym[items[0]] = items[1].strip()
    dict_acronym_noi[items[0]] = items[1].strip()
fin_map.close()
del dict_acronym_noi['i']
del dict_acronym_noi['I']

fin_trans = open(args.input, "r")
fout_trans = open(args.output, "w")
for line in fin_trans:
    line = line.strip()
    items = line.split()
    L = len(items)
    # First pass mapping to map I as part of acronym
    for i in range(L):
        if items[i] == 'i':
            x = 0
            while(i-1-x >= 0 and re.match(r'^[A-Z]$',items[i-1-x])):
                x += 1
            
            y = 0
            while(i+1+y < L and re.match(r'^[A-Z]$',items[i+1+y])):
                y += 1

            if x+y > 0:
                for bias in range(-x, y+1):
                    items[i+bias] = dict_acronym[items[i+bias]]
                  
    # Second pass mapping (not mapping 'i' and 'I')
    for i in range(len(items)):
        if items[i] in dict_acronym_noi.keys():
            items[i] = dict_acronym_noi[items[i]]
    sentence = ' '.join(items[1:])
    fout_trans.write(items[0]+ ' ' +sentence.lower() +'\n')

fin_trans.close()
fout_trans.close()
