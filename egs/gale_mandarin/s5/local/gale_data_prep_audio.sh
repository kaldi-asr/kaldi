#!/bin/bash 

# Copyright 2014 QCRI (author: Ahmed Ali)
# Apache 2.0


if [ $# -ne 2 ]; then
   echo "Arguments should be the <output folder> <data folder> "; exit 1
fi

# check that sox is installed 

which sox  &>/dev/null
if [[ $? != 0 ]]; then 
 echo "sox is not installed"
 exit 1 
fi

galeData=$1
wavedir=$galeData/wav
mkdir -p $wavedir

audio_path=$2

mkdir -p $wavedir/
  
#copy and convert the flac to wav
find $audio_path -type f -name *.flac  | while read file; do
  f_name=$(basename $file)
  if [[ ! -e $wavedir/"${f_name%.flac}.wav" ]]; then
   echo "soxing $file to $wavedir/$CD/"${f_name%.flac}.wav" "
   sox $file $wavedir/"${f_name%.flac}.wav"
  fi
  
done

find $wavedir -name *.wav > $galeData/wav$$ 
awk -F "/" '{print $NF}' $galeData/wav$$  | sed 's:\.wav::' > $galeData/id$$ 
paste -d ' ' $galeData/id$$ $galeData/wav$$  | sort -u > $galeData/wav.scp  

#clean 
rm -fr $galeData/id$$ $galeData/wav$$
echo data prep audio succeded

exit 0

