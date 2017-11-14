#!/bin/bash


if [ $# -eq 0 ]; then
  echo "Usage: ./local/phoneset_diff.sh <phones1.txt> <phones2.txt>"
  exit 1
fi

phones1=$1
phones2=$2

# Find the phones in 1 that are not in 2
comm -23 <(awk '{print $1}' $phones1 | sort ) \
         <(awk '{print $1}' $phones2 | sort )
