#!/bin/bash

# Copyright 2014 QCRI (author: Ahmed Ali)
# Copyright 2016 Johns Hopkins Univeersity (author: Jan "Yenda" Trmal)
# Apache 2.0


echo $0 "$@"

galeData=$(readlink -f "${@: -1}" );
wavedir=$galeData/wav
mkdir -p $wavedir


length=$(($#-1))
args=${@:1:$length}

# check that sox is installed
which sox  &>/dev/null
if [[ $? != 0 ]]; then
 echo "$0: sox is not installed"
 exit 1
fi

set -e -o pipefail

for var in $args; do
  CD=$(basename $var)
  [ -d $wavedir/$CD ] && rm -rf $wavedir/$CD
  mkdir -p $wavedir/$CD
  find $var -type f -name *.wav | while read file; do
    f=$(basename $file)
    if [[ ! -L "$wavedir/$CD/$f" ]]; then
      ln -sf $file $wavedir/$CD/$f
    fi
  done

  #make an flac symmlink as well
  find $var -type f -name *.flac  | while read file; do
    f=$(basename $file)

    if [[ ! -L "$wavedir/$CD/$f" ]]; then
      ln -sf $file $wavedir/$CD/$f
    fi
  done
done

#figure out the proper sox command line
#the flac will be converted on the fly
(
  for w in `find $wavedir -name *.wav` ; do
    base=`basename $w .wav`
    fullpath=`readlink -f $w`
    echo "$base sox $fullpath -r 16000 -t wav - |"
  done

  for w in `find $wavedir -name *.flac` ; do
    base=`basename $w .flac`
    fullpath=`readlink -f $w`
    echo "$base sox $fullpath -r 16000 -t wav - |"
  done
)  | sort -u > $galeData/wav.scp

#clean
rm -fr $galeData/id$$ $galeData/wav$$
echo "$0: data prep audio succeded"

exit 0

