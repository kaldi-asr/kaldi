#!/usr/bin/env bash

#  Copyright 2014  Johns Hopkins University (Author: Daniel Povey)
#  Apache 2.0.

# This script is for use with "retrain_fast.sh"; it combines the original model
# that you trained on top of, with the single layer model you trained, so that
# you can do joint backpropagation.

# Begin configuration options.
cmd=run.pl
# End configuration options.


echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 <original-nnet-dir> <new-nnet-dir> <combined-nnet-dir>"
  echo "where <original-nnet-dir> will typically be a normal neural net from another corpus,"
  echo "and <new-nnet-dir> will usually be a single-layer neural net trained on top of it by"
  echo "dumping the activations (e.g. using steps/online/nnet2/dump_nnet_activations.sh, I"
  echo "think no such script exists for non-online), and then training using"
  echo "steps/nnet2/retrain_fast.sh."
  echo "e.g.: $0 ../../swbd/s5b/exp/nnet2_online/nnet_gpu_online exp/nnet2_swbd_online/nnet_gpu_online exp/nnet2_swbd_online/nnet_gpu_online_combined"
fi


src1=$1
src2=$2
dir=$3

for f in $src1/final.mdl $src2/tree $src2/final.mdl; do
   [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1;
done


mkdir -p $dir/log

info=$dir/nnet_info
nnet-am-info $src1/final.mdl >$info
nc=$(grep num-components $info | awk '{print $2}');
if grep SumGroupComponent $info >/dev/null; then 
  nc_truncate=$[$nc-3]  # we did mix-up: remove AffineComponent,
                        # SumGroupComponent, SoftmaxComponent
else
                        # we didn't mix-up:
  nc_truncate=$[$nc-2]  # remove AffineComponent, SoftmaxComponent
fi

$cmd $dir/log/get_raw_nnet.log \
 nnet-to-raw-nnet --truncate=$nc_truncate $src1/final.mdl $dir/first_nnet.raw || exit 1;

$cmd $dir/log/append_nnet.log \
  nnet-insert --randomize-next-component=false --insert-at=0 \
  $src2/final.mdl $dir/first_nnet.raw $dir/final.mdl || exit 1;

$cleanup && rm $dir/first_nnet.raw

# Copy the tree etc., 

cp $src2/tree $dir || exit 1;

# Copy feature-related things from src1 where we built the initial model.
# Note: if you've done anything like mess with the feature-extraction configs,
# or changed the feature type, you have to keep track of that yourself.
for f in final.mat cmvn_opts splice_opts; do
  if [ -f $src1/$f ]; then
    cp $src1/$f $dir || exit 1;
  fi
done

echo "$0: created appended model in $dir"
