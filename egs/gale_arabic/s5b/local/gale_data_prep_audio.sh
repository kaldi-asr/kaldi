#!/bin/bash 

# Copyright 2014 QCRI (author: Ahmed Ali)
# Apache 2.0


galeData=$(readlink -f "${@: -1}" );  # last argumnet; the local folder
audio_dvds=${@:1:${#}-1} # all the audio dvds for GALE corpus; ; check audio=( in ../run.sh

mkdir -p $galeData 

# check that sox is installed 
which sox  &>/dev/null
if [[ $? != 0 ]]; then 
 echo "sox is not installed"; exit 1 
fi

for dvd in $audio_dvds; do
  dvd_full_path=$(readlink -f $dvd)
  if [[ ! -e $dvd_full_path ]]; then 
    echo missing $dvd_full_path; exit 1;
  fi
  find $dvd_full_path \( -name "*.wav" -o -name "*.flac" \)  | while read file; do
    id=$(basename $file | awk '{gsub(".wav","");gsub(".flac","");print}')
    echo "$id sox $file -r 16000 -t wav - |"
  done 
done | sort -u > $galeData/wav.scp

echo data prep audio succeded

exit 0

