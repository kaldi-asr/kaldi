#!/bin/bash

# Copyright 2012 Vassil Panayotov
# Apache 2.0

# Selects parts of VoxForge corpus based on certain criteria
# and makes symbolic links to the respective recordings 

# regex to select speakers based on pronunciation dialect
dialect='(American)|(British)'

# e.g. accept a "dialect" parameter
. utils/parse_options.sh

echo "=== Starting VoxForge subset selection(accent: $dialect) ..."

if [ $# -ne 2 ]; then
  echo "Usage: $0 [--dialect <regex>] <src-dir> <dst-dir>";
  exit 1;
fi

SRCDIR=$1
DSTDIR=$2

rm -rf ${DSTDIR} 1>/dev/null 2>&1
mkdir -p ${DSTDIR}

find $SRCDIR -iwholename '*etc/readme*' \
 -exec egrep -iHl 'pronunciation dialect.*'${dialect} {} \; |\
while read f; do 
  d=`dirname $f`; 
  d=`dirname $d`; 
  ln -s $d ${DSTDIR}; 
done

echo "*** VoxForge subset selection finished!"
