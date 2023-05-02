#!/usr/bin/env bash

# Copyright   2014  Johns Hopkins University (author: Daniel Povey) 
# Copyright   2016  Tsinghua University (author: Dong Wang)
# Apache 2.0

# Adapted from librispeech recipe local/download_and_untar.sh

remove_archive=false

if [ "$1" == --remove-archive ]; then
  remove_archive=true
  shift
fi

if [ $# -ne 3 ]; then
  echo "Usage: $0 [--remove-archive] <data-base> <url-base> <corpus-part>"
  echo "e.g.: $0 /nfs/public/materials/data/thchs30-openslr www.openslr.org/resources/18 data_thchs30"
  echo "With --remove-archive it will remove the archive after successfully un-tarring it."
  echo "<corpus-part> can be one of: data_thchs30, test-noise, resource"
fi

data=$1
url=$2
part=$3

if [ ! -d "$data" ]; then
  echo "$0: no such directory $data, make it"
  mkdir -p $data
fi

part_ok=false
list="data_thchs30 test-noise resource"
for x in $list; do 
  if [ "$part" == $x ]; then part_ok=true; fi
done
if ! $part_ok; then
  echo "$0: expected <corpus-part> to be one of $list, but got '$part'"
  exit 1;
fi

if [ -z "$url" ]; then
  echo "$0: empty URL base."
  exit 1;
fi

if [ -f $data/$part/.complete ]; then
  echo "$0: data part $part was already successfully extracted, nothing to do."
  exit 0;
fi


sizes="6453425169 1971460210 24813708"

if [ -f $data/$part.tgz ]; then
  size=$(/bin/ls -l $data/$part.tgz | awk '{print $5}')
  size_ok=false
  for s in $sizes; do if [ $s == $size ]; then size_ok=true; fi; done
  if ! $size_ok; then
    echo "$0: removing existing file $data/$part.tgz because its size in bytes $size"
    echo "does not equal the size of one of the archives."
    rm $data/$part.tgz
  else
    echo "$data/$part.tgz exists and appears to be complete."
  fi
fi

if [ ! -f $data/$part.tgz ]; then
  if ! which wget >/dev/null; then
    echo "$0: wget is not installed."
    exit 1;
  fi
  full_url=$url/$part.tgz
  echo "$0: downloading data from $full_url.  This may take some time, please be patient."

  cd $data
  pwd
  echo " wget --no-check-certificate $full_url"
  if ! wget --no-check-certificate $full_url; then
    echo "$0: error executing wget $full_url"
    exit 1;
  fi
  cd -
fi

cd $data

if ! tar -xvzf $part.tgz; then
  echo "$0: error un-tarring archive $data/$part.tgz"
  exit 1;
fi

cd -

touch $data/$part/.complete

echo "$0: Successfully downloaded and un-tarred $data/$part.tgz"

if $remove_archive; then
  echo "$0: removing $data/$part.tgz file since --remove-archive option was supplied."
  rm $data/$part.tgz
fi
