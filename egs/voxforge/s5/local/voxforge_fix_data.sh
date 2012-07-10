#!/bin/bash

# Copyright 2012 Vassil Panayotov
# Apache 2.0

# This script is meant to implement fixes for various irregularities in the
# structure of VoxForge's data.

echo "=== Starting VoxForge data pre-formatting ..."

if [ $# -ne 1 ]; then
  echo "Usage: $0 <data-dir>"
  exit 1
fi

data=$1
meta=data/local

if [ ! -d $data ]; then
  echo "\"$DATA\" is not a directory!"
  exit 1
fi

mkdir -p $meta

echo "--- Mapping the \"anonymous\" speakers to unique IDs ..."
ls -d $data/anonymous-*-* |\
 gawk '
 BEGIN {FS="-"; i=0} 
 { if (!dates[$2]) {dates[$2]=$0} 
   else { dates[$2] = dates[$2] " " $0}} 
 END { for (d in dates) {printf("anon%04d %s\n", i, dates[d]); i+=1 }}' |\
 sort -k1 > $meta/anon.map

while read l; do
  user=`echo $l | cut -f1 -d' '`
  echo "$l" | cut -f2- -d' ' | tr ' ' '\n' | while read -r ad; do
    newdir=`echo "$ad" | sed -e 's:anonymous\(-.*-.*\):'$user'\1:'`
    mv $ad $newdir
  done
done < "${meta}/anon.map" 

echo "*** Finished VoxForge data pre-formatting!"
