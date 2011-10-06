#!/bin/bash


# Usage: is_sorted.sh [script-file]
# This script returns 0 (success) if the script file argument [or standard input]
# is sorted and 1 otherwise.

export LC_ALL=C

if [ $# == 0 ]; then
  scp=-
fi
if [ $# == 1 ]; then
  scp=$1
fi
if [ $# -gt 1 -o "$1" == "--help" -o "$1" == "-h" ]; then
  echo "Usage: is_sorted.sh [script-file]"
  exit 1
fi

cat $scp > /tmp/tmp1.$$
sort /tmp/tmp1.$$ > /tmp/tmp2.$$
cmp /tmp/tmp1.$$ /tmp/tmp2.$$ >/dev/null
ret=$?
rm /tmp/tmp1.$$  /tmp/tmp2.$$
if [ $ret == 0 ]; then
   exit 0;
else
  echo "is_sorted.sh: script file $scp is not sorted";
  exit 1;
fi
