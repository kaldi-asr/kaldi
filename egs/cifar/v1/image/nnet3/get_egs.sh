#!/usr/bin/env bash

# This script is like steps/nnet3/get_egs.sh (it dumps examples for nnet3-based
# neural net training), except it is specialized for classification of
# fixed-size images (setups like MNIST, CIFAR and ImageNet); and you have to
# provide the dev or test data in a separate directory.


# Begin configuration section.
cmd=run.pl
egs_per_archive=25000
train_subset_egs=5000
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
  echo "                     as an integer fraction of the total."
  echo "  --train-subset-egs 5000  Number of images to put in the subset of"
  echo "                     training examples that's used for diagnostics on"
  echo "                     each iteration and for combination at the end"
  echo "                     (note: there is no data held-out from training"
  echo "                     data; we use the test or dev set for that.)"
  exit 1;
fi


train=$1
test=$2
egs=$3

for f in train/images.scp train/labels.txt test/images.scp test/labels.txt; do
   if [ ! -f $f ]; then
     echo "$0: expected file $f to exist"
     exit 1
   fi
done



if ! mkdir -p $egs; then
  echo "$0: could not make directory $egs"
  exit 1
fi

mkdir -p $dir/info $dir/log


paf="--print-args=false"
num_channels=$(cat $train/num_channels)
num_cols=$(head -n 1 $train/images.scp | feat-to-dim $paf scp:- -)
num_rows=$(head -n 1 $train/images.scp | feat-to-len $paf scp:- ark,t:- | awk '{print $2}')
width=$num_rows
height=$[$num_cols/$num_channels]
# the width of the image equals $num_rows.


# We put the label on t=0, and on the input, the t values
# go from 0 to $width-1, so in a sense the left-context
# of the model is 0 and the right-context is $width-1.
# This way of looking at it is more natural for speech
# or handwriting-recognition/OCR tasks than it is for
# images, but it's the way we do it.
echo 0 > $dir/info/left_context
echo $[num_rows-1] > $dir/info/right_context
echo $num_cols >$dir/info/feat_dim

num_train_images=$(wc -l < $train/labels.txt)
num_test_images=$(wc -l < $test/labels.txt)

awk '{print $1}' $train/labels.txt | utils/shuffle_list.pl | \
   head -n $train_subset_egs > $dir/train_subset_ids.txt


num_classes=$(wc -l <$dir/classes.txt)

$cmd $dir/log/get_train_diagnostic_egs.log \
  ali-to-post 'scp:filter_scp.pl $dir/train_subset_ids.txt $train/labels.scp|' ark:- \| \
  post-to-smat --dim=$num_classes ark:- ark:- \| \
  nnet3-get-egs-simple input='scp:filter_scp.pl $dir/train_subset_ids.txt $train/images.scp|' \
    output=ark:- ark:$dir/train_diagnostic.egs

$cmd $dir/log/get_test_or_dev_egs.log \
  ali-to-post scp:$test/labels.txt ark:- \| \
  post-to-smat --dim=$num_classes ark:- ark:- \| \
  nnet3-get-egs-simple input='scp:$test/images.scp|' \
    output=ark:$dir/train_diagnostic.egs \


# Now work out the split of the training data.

num_train_images=$(wc -l <$train/labels.txt)

# the + 1 is to round up, not down... we assume it doesn't divide exactly.
num_archives=$[num_train_images/samples_per_iter+1]

echo "$0: creating $num_archive archives of egs"

awk '{print $1}' <$train/labels.txt >$dir/train_ids.txt

split_ids=$(for n in $(seq $num_archives); do echo $dir/train_ids.$n.txt; done)

utils/split_scp.pl $dir/train_ids.txt $split_ids || exit 1



