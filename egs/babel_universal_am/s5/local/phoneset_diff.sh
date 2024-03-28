#!/bin/bash
# Copyright 2016  Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0

if [ $# -eq 0 ]; then
  echo "Usage: ./local/phoneset_diff.sh <phones1.txt> <phones2.txt>"
  echo "Find the phones in 1 that are not in 2."
  exit 1
fi

phones1=$1
phones2=$2

# Find the phones in 1 that are not in 2.
# Strip the position depending phone ending off of them, remove duplicates and
# ignore disambiguation phones.
comm -23 <(awk '{print $1}' $phones1 | sort ) \
         <(awk '{print $1}' $phones2 | sort ) |\
         sed 's/_[BISE]//g' | sort -u | grep -v '#.*'
