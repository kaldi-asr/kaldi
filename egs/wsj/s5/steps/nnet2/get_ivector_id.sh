#!/bin/bash
# Copyright (c) 2016, Johns Hopkins University (Yenda Trmal <jtrmal@gmail.com>)
# License: Apache 2.0

# Begin configuration section.
# End configuration section
set -e -o pipefail
set -o nounset                              # Treat unset variables as an error

# End configuration section.

#echo >&2 "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 1 ]; then
  echo >&2 "Usage: $0 <directory>"
  echo >&2 " e.g.: $0 exp/nnet3/extractor"
  exit 1
fi

ivecdir=$1

if [ -f $ivecdir/final.ie.id ] ; then
  cat $ivecdir/final.ie.id
elif [ -f $ivecdir/final.ie ] ; then
  # note the creation can fail in case the extractor directory
  # is not read-only media or the user des not have access rights
  # in that case we will just behave as if the id is not available
  id=$(md5sum $ivecdir/final.ie | awk '{print $1}')
  echo "$id" > $ivecdir/final.ie.id || true
  echo "$id"
else
  exit 0
fi

exit 0



