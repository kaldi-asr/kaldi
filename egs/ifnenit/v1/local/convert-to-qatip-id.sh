#!/bin/bash

# This script reads utterance ids from stdin and convert them
# to QATIP standard Kaldi ids. A standard Kaldi ID has the format
#   <corpus-name>-<10-char-spk-id>_<10-char-utt-id>
# IDs in stdin need to have the format
#   <spk-id>_<utt-id>
# The ids are shortened or filled up with leading xs if they do not
# consists of exactly 10 characters
#
# Usage: echo '<spk-id>_<utt-id>' | convert-to-qatip-id.sh <corpus-name>

if [ $# != 1 ]; then
  echo "Usage: echo '<spk-id>_<utt-id>' | convert-to-qatip-id.sh <corpus-name>"
  exit 1
fi

corpus=$1
if [[ ${#corpus} -ne 5 ]]
then
  echo "Corpus name must have 5 characters"
  exit 1
fi

function trimTo10
{
  if [[ ${#1} -ge 10 ]]
  then
    echo ${1:$(echo ${#1}-10 | bc)}
  else
    zeros="xxxxxxxxxx"
    nZeros=$(echo "10-${#1}" | bc)
    echo ${zeros:0:$nZeros}$1
  fi
}

while read line
do
  uttId=$line
  spkId=$uttId
  if [[ $uttId == *"_"* ]]
  then
    uttId=$(echo $line | cut -d'_' -f1)
    spkId=$(echo $line | cut -d'_' -f2-)
  fi 
  if [[ "$uttId" == *" "* || "$spkId" == *" "* ]]
  then
    echo "Space in utt id or spk id!"
    exit 1
  fi
  echo "$corpus-"$(trimTo10 "$uttId")_$(trimTo10 "$spkId")
done < /dev/stdin
