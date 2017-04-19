#!/usr/bin/env bash

# This script is like steps/nnet3/get_egs.sh, except it is specialized for
# classification of fixed-size images; and you have to provide the
# dev or test data in a separate directory.


# Begin configuration section.
cmd=run.pl
egs_per_archive=25000
test_mode=false
# end configuration section

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 2 ]; then
  echo "Usage: $0 [opts] <train-data-dir> <test-or-dev-data-dir> <egs-dir>"
  echo " e.g.: $0 --egs-per-iter 25000 data/cifar10_train exp/cifar10_train_egs"
  echo " or: $0 --test-mode true data/cifar10_test exp/cifar10_test_egs"
  echo "Options (with defaults):"
  echo "  --cmd 'run.pl'     How to run jobs (e.g. queue.pl)"
  echo "  --test-mode false  Set this to true if you just want a single archive"
  echo "                     egs.ark to be created (useful for test data)"
  echo "  --egs-per-archive 25000  Number of images to put in each training archive"
  echo "                     (this is a target; the actual number will be chosen"
  echo "                    as some fraction of the total."
  exit 1;
fi
