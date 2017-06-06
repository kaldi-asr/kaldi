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
stage=0
# end configuration section

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 3 ]; then
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


set -eu

train=$1
test=$2
dir=$3

for f in $train/images.scp $train/labels.txt $test/images.scp $test/labels.txt; do
   if [ ! -f $f ]; then
     echo "$0: expected file $f to exist"
     exit 1
   fi
done



if ! mkdir -p $dir; then
  echo "$0: could not make directory $dir"
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


num_classes=$(wc -l <$train/classes.txt)
num_classes_test=$(wc -l <$test/classes.txt)

if ! [ "$num_classes" -eq "$num_classes_test" ]; then
  echo "$0: training and test dirs $train and $test are not compatible"
  exit 1
fi

if [ $stage -le 0 ]; then
  $cmd $dir/log/get_train_diagnostic_egs.log \
       ali-to-post "ark:filter_scp.pl $dir/train_subset_ids.txt $train/labels.txt|" ark:- \| \
       post-to-smat --dim=$num_classes ark:- ark:- \| \
       nnet3-get-egs-simple input="scp:filter_scp.pl $dir/train_subset_ids.txt $train/images.scp|" \
       output=ark:- ark:$dir/train_diagnostic.egs
fi



if [ $stage -le 1 ]; then
  # we use the same filenames as the regular training script, but
  # the 'valid_diagnostic' egs are actually used as the test or dev
  # set.
  $cmd $dir/log/get_test_or_dev_egs.log \
       ali-to-post ark:$test/labels.txt ark:- \| \
       post-to-smat --dim=$num_classes ark:- ark:- \| \
       nnet3-get-egs-simple input=scp:$test/images.scp \
       output=ark:- ark:$dir/valid_diagnostic.egs
fi

# Now work out the split of the training data.

num_train_images=$(wc -l <$train/labels.txt)

# the + 1 is to round up, not down... we assume it doesn't divide exactly.
num_archives=$[num_train_images/egs_per_archive+1]


if [ $stage -le 2 ]; then
  echo "$0: creating $num_archives archives of egs"

  image/split_image_dir.sh $train $num_archives

  sdata=$train/split$num_archives

  $cmd JOB=1:$num_archives $dir/log/get_egs.JOB.log \
       ali-to-post ark:$sdata/JOB/labels.txt ark:- \| \
       post-to-smat --dim=$num_classes ark:- ark:- \| \
       nnet3-get-egs-simple input=scp:$sdata/JOB/images.scp \
        output=ark:- ark:$dir/egs.JOB.ark
fi

rm $dir/train_subset_ids.txt 2>/dev/null || true

ln -sf train_diagnostic.egs $dir/combine.egs

echo $num_archives >$dir/info/num_archives

# 'frames_per_eg' is actually the number of supervised frames per example, and
# in this case we only have supervision on t=0 at the output.
echo 1 >$dir/info/frames_per_eg

echo $num_train_images >$dir/info/num_frames

# actually 'output_dim' is not something that would be present in the
# 'info' directory for speech tasks; we add it here for the convenience of
# the training script, to make it easier to get the number of classes.
echo $num_classes >$dir/info/output_dim

echo "$0: finished generating egs"
exit 0
