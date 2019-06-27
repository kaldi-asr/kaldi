#!/bin/bash

# Copyright   2019  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.
#             2019  Idiap Research Institute (Author: Srikanth Madikeri)
#
# This script computes the total left and right context needed for example (eg)
# creation from a set of 'chain' models.
# See the usage message for more information about input and output formats.

# Begin configuration section.
frame_subsampling_factor=1   # The total frame subsampling factor of the bottom
                             # + top model, i.e. the relative difference in
                             # frame rate between the input of the bottom model
                             # and the output of the top model.  Would normally
                             # be 3.

langs=default                # the list of languages.  This script checks that
                             # in the dir (first arg to the script), each
                             # language exists as $lang.mdl, and it warns if
                             # any model files appear (which might indicate a
                             # script bug).
# End configuration section

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 2 ]; then
  cat 1>&2 <<EOF
Usage: $0 [opts] <model-dir> <output-info-file>
This script works out some acoustic-context-related information,
and writes it, long with  the options provided to the script,
to the <output-info-file> provided.  An example of what
output-info-file> might contain after this script is called, is:
langs default
frame_subsampling_factor 3
bottom_subsampling_factor 3
model_left_context 22
model_right_context 22
  e.g.: $0 --frame-subsampling-factor 3 
          --langs 'default' exp/chaina/tdnn1a_sp/0 exp/chaina/tdnn1a_sp/0/info.txt
 Options:
     --frame-subsampling-factor    # (default: 1)  Total frame subsampling factor of
                                   # both models combined, i.e. ratio of
                                   # frame rate of input features vs.
                                   # alignments and decoding (e.g. 3).
     --bottom-subsampling-factor   # (default: 1) Controls the frequency at which
                                   # the output of the bottom model is
                                   # evaluated, and the interpretation of frame
                                   # offsets in the top config file.  Must be a
                                   # divisor of --frame-subsampling-factor
     --langs                       # The list of languages (must be in quotes,
                                   # to be parsed as a single arg).  May be
                                   # 'default' or e.g. 'english french'
EOF
  exit 1;
fi


dir=$1
info_file=$2

# die on error or undefined variable.
set -e -u

if [ ! -d $dir ]; then
  echo 1>&2 "$0: expected directory $dir to exist"
  exit 1
fi

if [ -z $langs ]; then
  echo 1>&2 "$0: list of languages (--langs option) is empty"
  exit 1
fi

if  ! [ $frame_subsampling_factor -ge 1 ]; then
  echo 1>&2 "$0: there was a problem with the options --frame-subsampling-factor=$frame_subsampling_factor"
  exit 1
fi

mkdir -p $dir/temp

for lang in $langs; do
  if [ ! -s $dir/$lang.mdl ]; then
    echo 1>&2 "$0: expected file $dir/$lang.mdl to exist and be nonempty (check --langs option)"
    exit 1
  fi
  nnet3-am-info $dir/$lang.mdl > $dir/temp/$lang.info
  this_left_context=$(grep '^left-context:' $dir/temp/$lang.info | awk '{print $2}')
  this_right_context=$(grep '^right-context:' $dir/temp/$lang.info | awk '{print $2}')
done

left_context=$this_left_context
right_context=$this_right_context


cat >$info_file <<EOF
frame_subsampling_factor $frame_subsampling_factor
langs $langs
model_left_context $left_context
model_right_context $right_context
EOF


echo "$0: Finished getting model context"
