#!/bin/bash


if [ $# -eq 0 ]; then
  echo "Usage: ./local_/prepend_lang_id.sh <prefix> <data_dir> <odata>" 
  exit 1
fi

prefix=$1
data=$2
odata=$3


# In the data dir there should be at least: wav.scp, utt2spk, text.
# Optionally, there may be: segments, reco2file_and_channel.
# There may also be feats.scp and cmvn.scp files. 

mkdir -p $odata
# Check that all necessary files exist
for f in wav.scp utt2spk text; do
  if [ ! -f ${data}/${f} ]; then
    echo "Expected file $f to exist" && exit 1
  fi
done

# We do things differently depending on whether or not a segments file exists
if [ -f ${data}/segments ]; then
  # Prepend
  for f in text segments; do
    awk -v var=$prefix '{print var"_"$0}' ${data}/${f} > ${odata}/${f}
  done
  awk -v var=$prefix '{print var"_"$1, var"_"$2}' ${data}/utt2spk > ${odata}/utt2spk
  if [ -f ${data}/feats.scp ]; then
    awk -v var=$prefix '{print var"_"$0}' ${data}/feats.scp > ${odata}/feats.scp
    awk -v var=$prefix '{print var"_"$0}' ${data}/cmvn.scp > ${odata}/cmvn.scp
  fi
fi

./utils/utt2spk_to_spk2utt.pl ${odata}/utt2spk > ${odata}/spk2utt
cp ${data}/wav.scp ${odata}/wav.scp

exit 0

