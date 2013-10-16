#!/bin/bash

# Apache 2.0.  Copyright 2013, Hong Kong University of Science and Technology (author: Ricky Chan Ho Yin)

# This script calculates the average decoding real-time factor of a decoding directory by using the run time information inside the logs 

if [ $# -ne 1 ] && [ $# -ne 2 ]; then
  echo "Usage: $0 decode_directory [framePerSecond]"
  echo ""
  echo "## The default framerate framePerSecond = 100  i.e. 10ms sliding for input features during decoding"
  exit
fi

decodeDIR=$1

if [ ! -d $decodeDIR/log ]; then
  echo "decoding directory $decodeDIR/log not exist" 
  exit
fi

if [ $# -eq 2 ]; then
  framePerSecond=$2
else
  framePerSecond=100.0
fi

printf "$decodeDIR/log\t"

tail $decodeDIR/log/decode*.log | egrep -e 'Time taken .* real-time|Overall log-likelihood per frame' | awk -v fps=$framePerSecond 'BEGIN{sumTime=0; sumFrame=0;} {if($0 ~ / Time taken /) {pos=match($0, " [0-9.]+s:"); pos2=match($0, "s: real-time factor"); sumTime+=substr($0, pos+1, pos2-pos-1); } else {sumFrame+=$(NF-1)}; }; END{print sumTime/(sumFrame/fps)}'

