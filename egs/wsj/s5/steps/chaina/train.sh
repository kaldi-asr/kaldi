#!/bin/bash

# Copyright   2019  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.


# Begin configuration section
stage=0
leaky_hmm_coefficient=0.1
xent_regularize=0.1
apply_deriv_weights=false   # you might want to set this to true in unsupervised training
                            # scenarios.
memory_compression_level=2  # Enables us to use larger minibatch size than we
                            # otherwise could, but may not be optimal for speed
                            # (--> set to 0 if you have plenty of memory.
dropout_schedule=
srand=0
max_param_change=2.0
use_gpu=yes   # can be "yes", "no", "optional", "wait"

common_opts=           # Options passed through to nnet3-chaina-train and nnet3-chaina-combine

unadapted_top_weight=0.5
unadapted_bottom_weight=0.5

num_epochs=4.0   #  Note: each epoch may actually contain multiple repetitions of
                 #  the data, for various reasons:
                 #    using the --num-repeats option in process_egs.sh
                 #    data augmentation
                 #    different data shifts (this includes 3 different shifts
                 #    of the data if frame_subsampling_factor=3 (see $dir/init/info.txt)

num_jobs_initial=1
num_jobs_final=1
initial_effective_lrate=0.001
final_effective_lrate=0.0001
num_groups_per_minibatch=32  # note: if chunks_per_group=4, this would mean 128
                             # chunks per minibatch.

max_iters_combine=80
max_models_combine=20

# End configuration section



echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 2 ]; then
  echo "Usage: $0  [options] <egs-dir>  <model-dir>"
  echo " e.g.: $0 exp/chaina/tdnn1a_sp/egs  exp/chaina/tdnn1a_sp"
  echo ""
  echo " TODO: more documentation"
  exit 1
fi

egs_dir=$1
dir=$2

set -e -u  # die on failed command or undefined variable

steps/chaina/validate_randomized_egs.sh $egs_dir

for f in $dir/init/info.txt $dir/init/bottom.raw; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done


frame_subsampling_factor=$(awk '/^frame_subsampling_factor/ {print $2}' <$dir/init/info.txt)
num_scp_files=$(awk '/^num_scp_files/ {print $2}' <$dir/init/info.txt)

steps/chaina/internal/get_train_schedule.py \
  --frame-subsampling-factor=$frame_subsampling_factor \
  --num-jobs-initial=$num_jobs_initial \
  --num-jobs-final=$num_jobs_final \
  --num-epochs=$num_epochs \
  --num-scp-files=$num_scp_files \
  --frame-subsampling-factor=$frame_subsampling_factor \
  --initial-effective-lrate=$initial_effective_lrate \
  --final-effective-lrate=$final_effective_lrate \
  --schedule-out=$dir/schedule.txt



num_iters=$(wc -l <$dir/schedule.txt)
langs=$(awk '/^langs/ { $1=""; print; }' <$dir/0/info.txt)

mkdir -p $dir/log


# Copy models with initial learning rate and dropout options from $dir/init to $dir/0
mkdir -p $dir/0
lrate=$(awk ' {if(NR-1==0) { print;exit(0);}}' <$dir/schedule.txt | cut -f 5)
dropout_str=$(awk ' {if(NR-1==0) { print;exit(0);}}' <$dir/schedule.txt | cut -f 4)
run.pl $dir/log/init_bottom_model.log \
  nnet3-copy --learning-rate=$lrate --edits="$dropout_str" $dir/init/bottom.raw $dir/0/bottom.raw
for lang in $langs; do
  run.pl $dir/log/init_model_$lang.log \
         nnet3-am-copy --learning-rate=$lrate --edits="$dropout_str" $dir/init/$lang.mdl $dir/0/$lang.mdl
done


iter=0

echo "exiting early"
exit 0


# Note: the .ark files are not actually consumed directly downstream (only via
# the top-level .scp files), but we check them anyway for now.
for f in $dir/train.scp $dir/info.txt \
         $dir/heldout_subset.{ark,scp} $dir/train_subset.{ark,scp} \
         $dir/train.1.scp $dir/train.1.ark; do
  if ! [ -f $f -a -s $f ]; then
    echo "$0: expected file $f to exist and be nonempty."
    exit 1
  fi
done


if [ $(awk '/^dir_type/ { print $2; }' <$dir/info.txt) != "processed_chaina_egs" ]; then
  grep dir_type $dir/info.txt
  echo "$0: dir_type should be processed_chaina_egs in $dir/info.txt"
  exit 1
fi

lang=$(awk '/^lang / {print $2; }' <$dir/info.txt)

for f in $dir/misc/$lang.{trans_mdl,normalization.fst,den.fst}; do
  if ! [ -f $f -a -s $f ]; then
    echo "$0: expected file $f to exist and be nonempty."
    exit 1
  fi
done

echo "$0: sucessfully validated processed egs in $dir"
