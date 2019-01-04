#!/bin/bash

# Copyright   2019  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.
#
# This script takes nnet examples dumped by steps/chaina/get_raw_egs.sh and
# combines the chunks into groups by speaker (to the extent possible; it may
# need to combine speakers in some cases), locally randomizes the result, and
# dumps the resulting egs to disk.  Chunks of these will later be globally
# randomized (at the scp level) by steps/chaina/randomize_egs.sh


# Begin configuration section.
cmd=run.pl
chunks_per_spk=4
num_repeats=2  # number of times we repeat the same chunks with different
               # grouping.  Recommend 1 or 2; must divide chunks_per_spk
compress=true   # set this to false to disable compression (e.g. if you want to see whether
                # results are affected).


num_heldout_groups=200    # The number of groups (i.e. groups of chunks) that
                          # will go in the held-out set and the train subset
                          # (heldout_subset.scp and train_subset.scp).  The real
                          # point of train_subset.scp, and the reason we can't
                          # just use a subset of train.scp, is that it contains
                          # egs that are statistically comparable to
                          # heldout_subset.scp, so their prob can be
                          # meaningfully compared with those from
                          # heldout_subset.scp.  Note: the number (e.g. 200) is
                          # *after* merging chunks into groups of size
                          # $chunks_per_spk.


shuffle_buffer_size=5000   # Size of buffer (containing grouped egs) to use
                           # for random shuffle.

stage=0
nj=5             # the number of parallel jobs to run.
srand=0

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 2 ]; then
  echo "Usage: $0 [opts] <raw-egs-dir> <processed-egs-dir>"
  echo " e.g.: $0 --chunks-per-spk 4 exp/chaina/tdnn1a_sp/raw_egs exp/chaina/tdnn1a_sp/processed_egs"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options (alternative to this"
  echo "                                                   # command line)"
  echo "  --cmd (utils/run.pl;utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --chunks-per-spk <n;4>                           # Number of chunks (preferentially, from a single speaker"
  echo "                                                   # to combine into each example.  This grouping of"
  echo "                                                   # egs is part of the 'chaina' framework; the adaptation"
  echo "                                                   # parameters will be estimated from these groups."
  echo "  --num-repeats <n;2>                              # Number of times we group the same chunks into different"
  echo "                                                   # groups.  For now only the values 1 and 2 are"
  echo "                                                   # recommended, due to the very simple way we choose"
  echo "                                                   # the groups (it's consecutive)."
  echo "  --nj       <num-jobs;5>                          # Number of jobs to run in parallel.  Usually quite a"
  echo "                                                   # small number, as we'll be limited by disk access"
  echo "                                                   # speed."
  echo "  --compress <bool;true>                           # True if you want the egs to be compressed"
  echo "                                                   # (e.g. you may set to false for debugging purposes, to"
  echo "                                                   # check that the compression is not hurting)."
  echo "  --num-heldout-egs <n;200>                        # Number of egs to put in train_subset.scp and heldout_subset.scp."
  echo "                                                   # These will be used for diagnostics.  Note: this number is"
  echo "                                                   # the number of  grouped egs, after merging --chunks-per-spk"
  echo "                                                   # chunks into a single eg."
  echo "                                                   # ... may be a comma separated list, but we advise a single"
  echo "                                                   #  number in most cases, due to interaction with the need "
  echo "                                                   # to group egs from the same speaker into groups."
  echo "  --stage <stage|0>                                # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."
  exit 1;
fi

raw_egs_dir=$1
dir=$2

# die on error or undefined variable.
set -e -u

for f in $raw_egs_dir/all.scp $raw_egs_dir/info.txt $raw_egs_dir/misc/utt2spk; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist."
    exit 1
  fi
done

if ! awk '/dir_type /{if ($2 != "raw_chaina_dir") exit(1); }'; then
  echo "$0: input directory $raw_egs_dir does not seem to be of the right type."
fi



mkdir -p $dir/temp $dir/log


if [ $stage -le 0 ]; then
  echo "$0: choosing egs to merge"

  utt2uniq_opt=
  [ -f $raw_egs_dir/misc/utt2uniq ] && utt2uniq_opt="--utt2uniq=$raw_egs_dir/misc/utt2uniq"

  $cmd $dir/log/choose_egs_to_merge.log steps/chaina/internal/choose_egs_to_merge.py \
    --chunks-per-spk=$chunks_per_spk \
    --num-repeats=$num_repeats \
    --num-heldout-groups=$num_heldout_groups \
    $utt2uniq_opt \
    --scp-in=$raw_egs_dir/all.scp \
    --training-data-out=$dir/temp/train.list \
    --heldout-subset-out=$dir/temp/heldout_subset.list \
    --training-subset-out=$dir/temp/train_subset.list
fi

if [ $stage -le 1 ]; then


  for name in heldout_subset train_subset; do
    echo "$0: merging and shuffling $train egs"

    # Linearize these lists and add keys to make it an scp format.
    awk '{for (n=1;n<=NF;n++) { count++; print count "-" $n; }' <$dir/temp/${name}.list >$dir/temp/${name}.scp

    $cmd $dir/log/merge_${name}_egs.log \
       nnet3-chain-merge-egs --compress=$compress scp:$dir/temp/${name}.scp ark:- \| \
       nnet3-chain-shuffle-egs --srand=$srand $ark:- ark,scp:$dir/${name}.ark,$dir/${name}.scp
  done

  # Split up the training list into multiple smaller lists, as it could be long.
  utils/split_scp.pl $dir/train.list  $(for j in $(seq $nj); do echo $dir/temp/train.$j.list; done)
  for j in $(seq $nj); do
    awk '{for (n=1;n<=NF;n++) { count++; print count "-" $n; }' <$dir/temp/train.$j.list >$dir/temp/train.$j.scp
  done

  if [ -e $dir/storage ]; then
    # Make soft links to storage directories, if distributing this way..  See
    # utils/create_split_dir.pl.
    echo "$0: creating data links"
    utils/create_data_link.pl $(for j in $(seq $nj); do echo $dir/train.$j.ark; done)
  fi


  $cmd JOB=1:$nj $dir/log/merge_train_egs.JOB.log \
     nnet3-chain-merge-egs --compress=$compress scp:$dir/temp/train.JOB.scp ark:- \| \
     nnet3-chain-shuffle-egs --shuffle-buffer-size=$shuffle_buffer_size \
         --srand=\$[JOB+$srand] ark:- ark,scp:$dir/train.JOB.ark,$dir/train.JOB.scp

  cat $(for j in $(seq $nj); do echo $dir/train.$j.scp; done) > $dir/train.scp
fi



echo "$0: Finished processing egs"
