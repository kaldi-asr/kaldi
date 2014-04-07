#!/bin/bash 

# Copyright 2014 QCRI (author: Ahmed Ali)
# Apache 2.0


if [ $# -ne 7 ]; then
   echo "Arguments should be the <gale folder> <gale CD1> ... <gale CD6>"; exit 1
fi

# check that sox is installed 

which sox  &>/dev/null
if [[ $? != 0 ]]; then 
 echo "sox is not installed"
 exit 1 
fi

galeData=$(readlink -f $1); shift
wavedir=$galeData/wav
mkdir -p $wavedir


for var in "$@"; do
  CD=$(basename $var)
  mkdir -p $wavedir/$CD
  find $var -type f -name *.wav | while read file; do
    f=$(basename $file)
    if [[ ! -L "$wavedir/$CD/$f" ]]; then
      ln -s $file $wavedir/$CD/$f
    fi
done
  
  #copy and convert the flac to wav
  find $var -type f -name *.flac  | while read file; do
    f_name=$(basename $file)
	if [[ ! -e $wavedir/$CD/"${f_name%.flac}.wav" ]]; then
	 echo "soxing $file to $wavedir/$CD/"${f_name%.flac}.wav" "
     sox $file $wavedir/$CD/"${f_name%.flac}.wav"
	fi
    
  done
done

find $wavedir -name *.wav > $galeData/wav$$ 
awk -F "/" '{print $NF}' $galeData/wav$$  | sed 's:\.wav::' > $galeData/id$$ 
paste -d ' ' $galeData/id$$ $galeData/wav$$  | sort -u > $galeData/wav.scp  

#clean 
rm -fr $galeData/id$$ $galeData/wav$$
echo data prep audio succeded

exit 0

