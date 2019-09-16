#!/bin/bash

# Copyright   2014  Johns Hopkins University (author: Daniel Povey)
#             2017  Xingyu Na
# Apache 2.0

remove_archive=false

if [ "$1" == --remove-archive ]; then
  remove_archive=true
  shift
fi

if [ $# -ne 2 ]; then
  echo "Usage: $0 [--remove-archive] <data-base> <url-base>"
  echo "e.g.: $0 /export/a05/xna/data www.openslr.org/resources/38"
  echo "With --remove-archive it will remove the archive after successfully un-tarring it."
fi

data=$1
url=$2
part=ST-CMDS-20170001_1-OS

if [ ! -d "$data" ]; then
  echo "$0: no such directory $data, make it"
  mkdir -p $data
fi

if [ -z "$url" ]; then
  echo "$0: empty URL base."
  exit 1;
fi

if [ -f $data/.complete ]; then
  echo "$0: data part $part was already successfully extracted, nothing to do."
  exit 0;
fi

# sizes of the archive files in bytes.
sizes="8231662593"

if [ -f $data/$part.tar.gz ]; then
  size=$(/bin/ls -l $data/$part.tar.gz | awk '{print $5}')
  size_ok=false
  for s in $sizes; do if [ $s == $size ]; then size_ok=true; fi; done
  if ! $size_ok; then
    echo "$0: removing existing file $data/$part.tar.gz because its size in bytes $size"
    echo "does not equal the size of one of the archives."
    rm $data/$part.tar.gz
  else
    echo "$data/$part.tar.gz exists and appears to be complete."
  fi
fi

if [ ! -f $data/$part.tar.gz ]; then
  if ! which wget >/dev/null; then
    echo "$0: wget is not installed."
    exit 1;
  fi
  full_url=$url/$part.tar.gz
  echo "$0: downloading data from $full_url.  This may take some time, please be patient."

  cd $data
  if ! wget --no-check-certificate $full_url; then
    echo "$0: error executing wget $full_url"
    exit 1;
  fi
fi

cd $data

if ! tar -xvzf $part.tar.gz; then
  echo "$0: error un-tarring archive $data/$part.tar.gz"
  exit 1;
fi

touch $data/.complete

echo "$0: Successfully downloaded and un-tarred $data/$part.tgz"

if $remove_archive; then
  echo "$0: removing $data/$part.tar.gz file since --remove-archive option was supplied."
  rm $data/$part.tar.gz
fi

exit 0;
