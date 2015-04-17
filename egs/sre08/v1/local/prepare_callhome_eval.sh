#!/bin/bash

set -e

. path.sh 

echo $*

if [ $# -ne 2 ]; then
  echo "Usage: local/prepare_callhome_eval.sh <DATA> <data-dir>"
  echo " e.g.: local/prepare_callhome_eval.sh /home/dpovey/diarization/data data/callhome_eval"
  exit 1
fi

DATA=$1
dir=$2

mkdir -p $dir

sph2pipe=`which sph2pipe`

if [ -z "$sph2pipe" ]; then 
  echo "$0: Cannot find sph2pipe"
  exit 1
fi

for x in `find $DATA/ -name "*.sph"`; do
  y=${x##*/}
  z=${y%.sph}
  echo "$z $sph2pipe -f wav -p $x |"
done | sort -k 1,1 > $dir/wav.scp

awk '{print $1" "$1}' $dir/wav.scp > $dir/utt2spk
cp $dir/utt2spk $dir/spk2utt

utils/fix_data_dir.sh $dir
