#!/bin/bash

# Copyright 2014 Vassil Panayotov
# Apache 2.0

if [ $# -ne "2" ]; then
  echo "Usage: $0 <base-url> <download_dir>"
  echo "e.g.: $0 http://www.openslr.org/resources/11 data/local/lm"
  exit 1
fi

base_url=$1
dst_dir=$2

# associative array to hold the expected file sizes
# the array assignments below can be auto-generated using e.g.
# for f in $data_dir/*; do bf=$(basename $f); fs=$(du -L -b $f | awk '{print $1}'); echo "sizes[\"$bf\"]=\"$fs\""; done
declare -A sizes
sizes["3-gram.arpa.gz"]="759636181"
sizes["3-gram.pruned.1e-7.arpa.gz"]="34094057"
sizes["3-gram.pruned.3e-7.arpa.gz"]="13654242"
sizes["4-gram.arpa.gz"]="1355172078"
sizes["4-gram.pruned.1e-7.arpa.gz"]="32174144"
sizes["4-gram.pruned.3e-7.arpa.gz"]="11970993"
sizes["g2p-model-5"]="20098243"
sizes["librispeech-lm-corpus.tgz"]="1803499244"
sizes["librispeech-vocab.txt"]="1737588"
sizes["librispeech-lexicon.txt"]="5627653"

function check_and_download () {
  [[ $# -eq 1 ]] || { echo "check_and_download() expects exactly one argument!"; return 1; }
  fname=$1
  echo "Downloading file '$fname' into '$dst_dir'..."
  expect_size="${sizes["$fname"]}"
  if [[ -s $dst_dir/$fname ]]; then
    fsize=$(du -b $dst_dir/$fname | awk '{print $1}')
    if [[ "$fsize" -eq "$expect_size" ]]; then
      echo "'$fname' already exists and appears to be complete"
      return 0
    else
      echo "'$fname' exists, but the size is wrong - re-downloading ..."
    fi
  fi
  wget --no-check-certificate -O $dst_dir/$fname $base_url/$fname || {
    echo "Error while trying to download $fname!"
    return 1
  }
  fsize=$(du -b $dst_dir/$fname | awk '{print $1}')
  [[ "$fsize" -eq "$expect_size" ]] || { echo "$fname: file size mismatch!"; return 1; }
  return 0
}

mkdir -p $dst_dir

for f in 3-gram.arpa.gz 3-gram.pruned.1e-7.arpa.gz 3-gram.pruned.3e-7.arpa.gz 4-gram.arpa.gz \
         g2p-model-5 librispeech-lm-corpus.tgz librispeech-vocab.txt librispeech-lexicon.txt; do
  check_and_download $f || exit 1
done

cd $dst_dir
ln -sf 3-gram.pruned.1e-7.arpa.gz lm_tgmed.arpa.gz
ln -sf 3-gram.pruned.3e-7.arpa.gz lm_tgsmall.arpa.gz
ln -sf 3-gram.arpa.gz lm_tglarge.arpa.gz
ln -sf 4-gram.arpa.gz lm_fglarge.arpa.gz

exit 0
