#!/usr/bin/env bash

# Copyright   2014  Johns Hopkins University (author: Daniel Povey)
#             2017  Luminar Technologies, Inc. (author: Daniel Galvez)
# Apache 2.0

remove_archive=false

if [ "$1" == --remove-archive ]; then
  remove_archive=true
  shift
fi

if [ $# -ne 3 ]; then
  echo "Usage: $0 [--remove-archive] <data-base> <url-base> <corpus-part>"
  echo "e.g.: $0 /export/a05/dgalvez/ www.openslr.org/resources/31 dev-clean-2"
  echo "With --remove-archive it will remove the archive after successfully un-tarring it."
  echo "<corpus-part> can be one of: dev-clean-2, test-clean-5, dev-other, test-other,"
  echo "          train-clean-100, train-clean-360, train-other-500."
fi

data=$1
url=$2
part=$3

if [ ! -d "$data" ]; then
  echo "$0: no such directory $data"
  exit 1;
fi

data=$(readlink -f $data)

part_ok=false
list="dev-clean-2 train-clean-5"
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

if [ -f $data/LibriSpeech/$part/.complete ]; then
  echo "$0: data part $part was already successfully extracted, nothing to do."
  exit 0;
fi


#sizes="126046265 332747356"
sizes="126046265 332954390"

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
  cd -
fi

cd $data

if ! tar -xvzf $part.tar.gz; then
  echo "$0: error un-tarring archive $data/$part.tar.gz"
  exit 1;
fi

touch $data/LibriSpeech/$part/.complete

echo "$0: Successfully downloaded and un-tarred $data/$part.tar.gz"

if $remove_archive; then
  echo "$0: removing $data/$part.tar.gz file since --remove-archive option was supplied."
  rm $data/$part.tar.gz
fi
