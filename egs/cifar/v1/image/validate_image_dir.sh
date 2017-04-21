#!/usr/bin/env bash

# This script validates a directory containing training or test images
# for image-classification tasks with fixed-size images.


if [ $# != 1 ]; then
  echo "Usage: $0 <image-dir-to-validate>"
  echo "e.g.: $0 data/cifar10_train"
fi

dir=$1

[ -e ./path.sh ] && . ./path.sh

if [ ! -d $dir ]; then
  echo "$0: directory $dir does not exist."
fi

for f in images.scp labels.txt classes.txt num_colors; do
  if [ ! -s "$dir/$f" ]; then
    echo "$0: expected file $dir/$f to exist and be nonempty"
  fi
done


num_colors=$(cat $dir/num_colors)

if ! [[ $num_colors -gt 0 ]]; then
  echo "$0: expected the file $dir/num_colors to contain a number >0"
  exit 1
fi

paf="--print-args=false"

num_cols=$(head -n 1 $dir/images.scp | feat-to-dim $paf scp:- -)
if ! [[ $[$num_cols%$num_colors] == 0 ]]; then
  echo "$0: expected the number of columns in the image matrices ($num_cols) to "
  echo "    be a multiple of the number of colors ($num_colors)"
  exit 1
fi

num_rows=$(head -n 1 $dir/images.scp | feat-to-len $paf scp:- -)

height=$[$num_cols/$num_colors]

echo "$0: images are width=$num_rows by height=$height, with $num_colors colors."

if ! cmp <(awk '{print $1}' $dir/images.scp) <(awk '{print $1}' $dir/labels.txt); then
  echo "$0: expected the first fields of $dir/images.scp and $dir/labels.txt to match up."
  exit 1;
fi

if ! [[ $num_cols -eq $(tail -n 1 $dir/images.scp | feat-to-dim $paf scp:- -) ]]; then
  echo "$0: the number of columns in the image matrices is not consistent."
  exit 1
fi

if ! [[ $num_rows -eq $(tail -n 1 $dir/images.scp | feat-to-len scp:- -) ]]; then
  echo "$0: the number of rows in the image matrices is not consistent."
  exit 1
fi

# Note: we don't require images.scp and labels.txt to be sorted, but they
# may not contain repeated keys.
if ! awk '{if($1 in a) { print "validate_image_dir.sh: key " $1 " is repeated in labels.txt"; exit 1; } a[$1]=1; }'; then
  exit 1
fi


if ! utils/int2sym.pl -f 2 $dir/classes.txt <$dir/labels.txt >/dev/null; then
  echo "$0: classes.txt may have the wrong format or may not cover all labels in $dir/labels.txt"
  exit 1;
fi


echo "$0: validated image-data directory $dir"
exit 0
