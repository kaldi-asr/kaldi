#!/bin/bash

# Copyright   2019  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.
# Copyright   2019  Idiap Research Institute (Author: Srikanth Madikeri).  Apache 2.0.
#
# This script takes nnet examples dumped by steps/chain/process_egs.sh,
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
num_blocks=256

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
  echo " e.g.: $0 --frames-per-job 2000000 exp/chain/tdnn1a_sp/processed_egs exp/chain/tdnn1a_sp/egs"
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
  echo "                                                   # be used in nnet3-chain-combine to decide which"
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

if ! steps/chain2/validate_processed_egs.sh $processed_egs_dir; then
  echo "$0: could not validate input directory $processed_egs_dir"
  exit 1
fi

# Work out how many groups per job and how many frames per job we'll have

info_in=$processed_egs_dir/info.txt

# num_scp_files is the number of archives
num_input_frames=$(awk '/^num_input_frames/ { nif=$2; print nif}' $info_in)
frames_per_chunk_avg=$(awk '/^frames_per_chunk_avg/ { fpc=$2; print fpc}' $info_in)
num_chunks=$(awk '/^num_chunks/ { nc=$2; print nc}' $info_in)
num_scp_files=$[(num_chunks * frames_per_chunk_avg)/frames_per_job +1]
[ $num_scp_files -eq 0 ] && num_scp_files=1

frames_per_scp_file=$[(num_chunks*frames_per_chunk_avg)/num_scp_files] # because it may be slightly different from frames_per_job


mkdir -p $dir/temp

if [ -d $dir/misc ]; then
  rm -r $dir/misc
fi

mkdir -p $dir/misc
cp $processed_egs_dir/misc/* $dir/misc

utils/shuffle_list.pl  $processed_egs_dir/train.scp > $dir/temp/train.scp
utils/split_scp.pl $dir/temp/train.scp $(for i in $(seq $num_blocks); do echo $dir/temp/train.$i.scp; done)
for i in `seq $num_blocks`; do
    utils/split_scp.pl <(utils/shuffle_list.pl $dir/temp/train.$i.scp) $(for j in $(seq $num_scp_files); do echo $dir/temp/train.$i.$j.scp; done)
done
for j in `seq $num_scp_files`; do
    cat $dir/temp/train.*.$j.scp | utils/shuffle_list.pl > $dir/train.$j.scp
done
rm -rf $dir/temp &

cp $processed_egs_dir/heldout_subset.scp $processed_egs_dir/train_subset.scp $dir/


# note: there is only one language in $processed_egs_dir (any
# merging would be done at the randomization stage but that is not supported yet).

lang=$(awk '/^lang / { print $2; }' <$processed_egs_dir/info.txt)

# We'll store info files per language, containing the part of the information
# that is language-specific, plus a single global info.txt containing stuff that
# is not language specific.
# This will get more complicated once we actually support multiple languages,
# and when we allow multiple input processed egs dirs for the same language.

grep -v -E '^dir_type|^lang|^feat_dim' <$processed_egs_dir/info.txt | \
  cat <(echo "dir_type randomized_chain_egs") - > $dir/info_$lang.txt


cat <<EOF >$dir/info.txt
dir_type randomized_chain_egs
num_scp_files $num_scp_files
langs $lang
frames_per_scp_file $frames_per_scp_file
EOF
# frames_per_job, after rounding, becomes frames_per_scp_file.

# note: frames_per_chunk_avg will be present in the info.txt file as well as
# the per-language files.
grep -E '^feat_dim|^frames_per_chunk_avg' <$processed_egs_dir/info.txt >>$dir/info.txt



if ! cat $dir/info.txt | awk '{if (NF == 1) exit(1);}'; then
  echo "$0: we failed to obtain at least one of the fields in $dir/info.txt"
  exit 1
fi


wait;
echo "$0: Finished randomizing egs"
