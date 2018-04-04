#!/bin/bash

# Copyright 2012 Vassil Panayotov
# Apache 2.0

# Map anonymous users to unique IDs

echo "=== Starting to map anonymous users to unique IDs ..."

if [ $# -ne 1 ]; then
  echo "Usage: $0 <data-dir>"
  exit 1
fi

data=$1
data_local=data/local

if [ ! -d $data ]; then
  echo "\"$DATA\" is not a directory!"
  exit 1
fi

mkdir -p $data_local

echo "--- Mapping the \"anonymous\" speakers to unique IDs ..."
ls -d $data/anonymous-*-* |\
 awk '
 BEGIN {i=0}
 { anon_users[++i] = $0; }
 END { for (j in anon_users) {printf("anonymous%04d %s\n", j, anon_users[j]);}}' |\
 sort -k1 > $data_local/anon.map

while read l; do
  user=$(echo $l | cut -f1 -d' ')
  echo "$l" | cut -f2- -d' ' | while read -r ad; do
    newdir=`echo "$ad" | sed -e 's:anonymous\(-.*-.*\):'$user'\1:'`
    mv $ad $newdir
  done
done < "${data_local}/anon.map"

echo "*** Finished mapping anonymous users!"
