#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#    2018  Saikiran Valluri, GoVivace inc.

import os, sys

if len(sys.argv) < 3:
    print("Usage : python . <pocolmmodelpat> <unigram weights outfile>")
    print("      Used for generating the unigram weights for second pass vocabulary from the first pass pocolm training metaparameters.")
    sys.exit()
 
pocolmdir=sys.argv[1]
unigramwts=open(sys.argv[2], 'w')

names = open(pocolmdir+"/names", 'r')
metaparams = open(pocolmdir+"/metaparameters", 'r')

name_mapper={}
for line in names:
    fields=line.split()
    name_mapper[fields[0]] = fields[1]
    
lns = metaparams.readlines()
for lineno in range(len(name_mapper.keys())):
    line = lns[lineno]
    fileid = line.split()[0].split("_")[-1]
    weight = line.split()[1]
    unigramwts.write(name_mapper[fileid] + "  " + weight + "\n")

names.close()
unigramwts.close()
metaparams.close()
