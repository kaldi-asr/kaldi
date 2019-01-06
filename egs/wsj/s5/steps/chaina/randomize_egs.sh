#!/bin/bash

# Copyright   2019  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.
#
# This script takes nnet examples dumped by steps/chaina/process_egs.sh,
# globally randomizes the egs, and divides into multiple .scp files.  This is
# the form of egs which is consumed by the training script.  All this is done
# only by manipulating the contents of .scp files.  To keep locality of disk
# access, we only randomize blocks of egs (e.g.  blocks containing 128 groups of
# sequences).  This doesn't defeat randomization, because both process_egs.sh
# and the training script use nnet3-shuffle-egs to do more local randomization.

# Later on, we'll have a multilingual/multi-input-dir version fo this script
# that combines egs from various data sources and possibly multiple languages.
# This version assumes there is just one language.

# Begin configuration section.
cmd=run.pl

groups_per_block=128     # The 'groups' are the egs in the scp file from
                         # process_egs.sh, containing '--chunks-per-group' sequences
                         # each.

frames_per_job=3000000   # The number of frames of data we want to process per
                         # training job (will determine how long each job takes,
                         # and the frequency of model averaging.  This was
                         # previously called --frames-per-iter, but
                         # --frames-per-job is clearer as each job does this
                         # many.

num_groups_combine=1000  # the number of groups from the training set that we
                         # randomly choose as input to nnet3-chain-combine;
                         # these will go to combine.scp.  train_subset.scp and
                         # heldout_subset.scp are, for now, just copied over
                         # from the input.

# Later we may provide a mechanism to change the language name; for now we
# just copy it from the input.


srand=0
stage=0

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 2 ]; then
  echo "Usage: $0 [opts] <processed-egs-dir> <randomized-egs-dir>"
  echo " e.g.: $0 --frames-per-job 200000 exp/chaina/tdnn1a_sp/processed_egs exp/chaina/tdnn1a_sp/egs"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options (alternative to this"
  echo "                                                   # command line)"
  echo "  --cmd (utils/run.pl;utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --groups-per-block <n;128>                       # The number of groups (i.e. previously merged egs"
  echo "                                                   # containing --chunks-per-group chunks) to to consider "
  echo "                                                   # as one block, where whole blocks are randomized;"
  echo "                                                   # smaller means more complete randomization but less"
  echo "                                                   # local disk access."
  echo "  --frames-per-job <n;3000000>                     # The number of input frames (not counting context)"
  echo "                                                   # that we aim to have in each scp file after"
  echo "                                                   # randomization and splitting."
  echo "  --num-groups-combine <n;1000>                    # The number of randomly chosen groups to"
  echo "                                                   # put in the subset in 'combine.scp' which will"
  echo "                                                   # be used in nnet3-chaina-combine to decide which"
  echo "                                                   # models to average over."
  echo "  --stage <stage|0>                                # Used to run this script from somewhere in"
  echo "                                                   # the middle."
  echo "  --srand <srand|0>                                # Random seed, affects randomization."
  exit 1;
fi

processed_egs_dir=$1
dir=$2

# die on error or undefined variable.
set -e -u

if ! steps/chaina/validate_processed_egs.sh $processed_egs_dir; then
  echo "$0: could not validate input directory $processed_egs_dir"
  exit 1
fi

# Work out how many groups per job and how many frames per job we'll have

frames_per_group_avg=$(awk '/^frames_per_chunk_avg/ { fpc=$2; } /^chunks_per_group/ { print int(fpc * $2); }')
if ! [ $frames_per_group_avg -gt 0 ]; then
  echo "$0: error getting frames per group.";
fi

num_groups=$(wc -l <$processed_egs_dir/train.scp)

num_scp_files=$[[ (frames_per_group_avg + frames_per_job / 2) / frames_per_job ]]
[ $num_scp_files -eq 0 ] && num_scp_files=1

frames_per_scp_file=$[[(frames_per_group_avg * num_groups) / num_scp_files]]
groups_per_scp_file=$[[ num_groups / num_scp_files]]


mkdir -p $dir/temp

if [ -d $dir/misc ]; then
  rm -r $dir/misc
fi

mkdir -p $dir/misc
cp $processed_egs_dir/misc/* $dir/misc


# We want to globally randomize the order of these blocks of (e.g.) 128 lines of
# the input train.scp, and then split up into $num_scp_files groups.  we could
# do this in a specially-written python script, but instead we do it with a
# combination of existing Kaldi and UNIX utilities.

awk '{block=sprintf("%05d", NR / groups_per_block); group_id=$1; print group_id, block;}' \
    <$processed_egs_dir/train.scp >$dir/temp/key2block

# get list of blocks
awk '{print $2}' | uniq <$dir/temp/key2block > $dir/temp/blocks
# get randomized-order list of blocks
utils/shuffle_list.pl --srand "$srand" <$dir/temp/blocks > $dir/temp/blocks_rand
# Map block-ids to randomized-order block-ids
paste $dir/temp/blocks $dir/temp/blocks_rand > $dir/temp/block2rand


# The following command first maps block-ids to randomized-order block-ids, then
# sorts the keys by these randomized-order block-ids while otherwise maintaining
# stable sorting (-s) which keeps the keys in the blocks in the same order.
utils/apply_map.pl -f 2 $dir/temp/block2rand <$dir/temp/key2block | \
  sort -k2 -s > $dir/temp/key2block_rand


# The following command just changes the order of train.scp to
# match the order in key2block_rand (which has the order of blocks
# of lines randomly moved around).
awk '{print $1, $1}' $dir/temp/key2block_rand | \
  utils/apply_map.pl -f 2 $processed_egs_dir/train.scp \
                     >$dir/temp/train.scp_rand


# The following command splits up $dir/temp/train.scp_rand (the randomized-order
# version of train.scp), while keeping distinct blocks in separate scp files,
# thanks to the --utt2spk option.
utils/split_scp.pl --utt2spk=$dir/temp/key2block_rand \
   $dir/temp/train.scp_rand \
   $(for i in $(seq $num_scp_files); do echo $dir/train.$i.scp; done)


cp $processed_egs_dir/heldout_subset.scp $processed_egs_dir/train_subset.scp $dir/



cat $processed_egs_dir/info.txt  | awk '
  /^dir_type/ { print "dir_type randomized_chaina_egs"; next; }
  /^lang / { print "langs", $2; next }
  /^num_input_frames/ { print $2 * num_repeats; next; } # approximate; ignores held-out egs.
   {print;}
  END{print "chunks_per_group " chunks_per_group; print "num_repeats " num_repeats;}' >$dir/info.txt

cat <<EOF >>$dir/info.txt
num_scp_files $num_scp_files
frames_per_scp_file $frames_per_scp_file
groups_per_scp_file $groups_per_scp_file
EOF

# Note: frame_per_job, after rounding, becomes frames_per_scp_file.


if ! cat $dir/info.txt | awk '{if (NF == 1) exit(1);}'; then
  echo "$0: we failed to obtain at least one of the fields in $dir/info.txt"
  exit 1
fi


echo "$0: Finished randomizing egs"
