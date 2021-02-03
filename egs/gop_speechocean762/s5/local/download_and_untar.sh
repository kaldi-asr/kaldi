#!/usr/bin/env bash

# Copyright      2014  Johns Hopkins University (author: Daniel Povey)
#           2020-2021  Xiaomi Corporation (Author: Junbo Zhang, Yongqing Wang)
# Apache 2.0

set -e

remove_archive=false
if [ "$1" == --remove-archive ]; then
  remove_archive=true
  shift
fi

if [ $# -ne 2 ]; then
  echo "Usage: $0 [--remove-archive] <url-base> <data-base>"
  echo "e.g.: $0 www.openslr.org/resources/101 /home/storage07/zhangjunbo/data"
  echo "With --remove-archive it will remove the archive after successfully un-tarring it."
  exit 1
fi

url=$1
data=$2
[ -d $data ] || mkdir -p $data

corpus_name=speechocean762

if [ -z "$url" ]; then
  echo "$0: empty URL base."
  exit 1;
fi

if [ -f $data/$corpus_name/.complete ]; then
  echo "$0: data part $corpus_name was already successfully extracted, nothing to do."
  exit 0;
fi

# Check the archive file in bytes
ref_size=520810923
if [ -f $data/$corpus_name.tar.gz ]; then
  size=$(/bin/ls -l $data/$corpus_name.tar.gz | awk '{print $5}')
  if [ $ref_size != $size ]; then
    echo "$0: removing existing file $data/$corpus_name.tar.gz because its size in bytes $size"
    echo "does not equal the size of one of the archives."
    rm $data/$corpus_name.tar.gz
  else
    echo "$data/$corpus_name.tar.gz exists and appears to be complete."
  fi
fi

# If you have permission to access Xiaomi's server, you would not need to
# download it from OpenSLR
path_on_mi_server=/home/storage06/wangyongqing/share/data/$corpus_name.tar.gz
if [ -f $path_on_mi_server ]; then
  cp $path_on_mi_server $data/$corpus_name.tar.gz
fi

if [ ! -f $data/$corpus_name.tar.gz ]; then
  if ! which wget >/dev/null; then
    echo "$0: wget is not installed."
    exit 1;
  fi
  full_url=$url/$corpus_name.tar.gz

  echo "$0: downloading data from $full_url.  This may take some time, please be patient."
  if ! wget -c --no-check-certificate $full_url -O $data/$corpus_name.tar.gz; then
    echo "$0: error executing wget $full_url"
    exit 1;
  fi
fi

cd $data
if ! tar -xvzf $corpus_name.tar.gz; then
  echo "$0: error un-tarring archive $data/$corpus_name.tar.gz"
  exit 1;
fi

touch $corpus_name/.complete
cd -

echo "$0: Successfully downloaded and un-tarred $data/$corpus_name.tar.gz"

if $remove_archive; then
  echo "$0: removing $data/$corpus_name.tar.gz file since --remove-archive option was supplied."
  rm $data/$corpus_name.tar.gz
fi
