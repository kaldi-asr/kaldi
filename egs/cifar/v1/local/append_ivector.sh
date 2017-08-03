#!/usr/bin/env bash

# Begin configuration section.
stage=0
# end configuration section

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 2 ]; then
  echo "Usage: $0 [opts] <image-data-dir> <ivector-data-dir>"
  echo " e.g.:  data/cifar10_train exp/ivectors_cifar10_train " 
  exit 1;
fi

set -eu
image_data=$1
ivector_data=$2
dir=$2

for f in $image_data/images.scp $ivector_data/ivector.scp; do
   if [ ! -f $f ]; then
     echo "$0: expected file $f to exist"
     exit 1
   fi
done

if [ $stage -le 0 ]; then
  nnet3-append-ivector-to-image scp:$image_data/images.scp scp:$ivector_data/ivector.scp \
  ark,scp:$dir/ivector_appended.ark,$dir/ivector_appended.scp
fi
