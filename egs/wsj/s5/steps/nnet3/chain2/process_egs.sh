#!/bin/bash

# Copyright   2019  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.
# Copyright   2019  Idiap Research Institute (Author: Srikanth Madikeri).  Apache 2.0.
#
# This script takes nnet examples dumped by steps/chain/get_raw_egs.sh and
# combines the chunks into groups by speaker (to the extent possible; it may
# need to combine speakers in some cases), locally randomizes the result, and
# dumps the resulting egs to disk.  Chunks of these will later be globally
# randomized (at the scp level) by steps/chaina/randomize_egs.sh


# Begin configuration section.
cmd=run.pl
num_repeats=1  # number of times we repeat the same chunks with different
               # grouping.  
compress=true   # set this to false to disable compression (e.g. if you want to see whether
                # results are affected).

num_utts_subset=300     # number of utterances in validation and training
                        # subsets used for shrinkage and diagnostics.


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
  echo " e.g.: $0 exp/chaina/tdnn1a_sp/raw_egs exp/chaina/tdnn1a_sp/processed_egs"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options (alternative to this"
  echo "                                                   # command line)"
  echo "  --cmd (utils/run.pl;utils/queue.pl <queue opts>) # how to run jobs."
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
  echo "                                                   # the number of  grouped egs, after merging --chunks-per-group"
  echo "                                                   # chunks into a single eg."
  echo "                                                   # ... may be a comma separated list, but we advise a single"
  echo "                                                   #  number in most cases, due to interaction with the need "
  echo "                                                   # to group egs from the same speaker into groups."
  echo "  --stage <stage|0>                                # Used to run this script from somewhere in"
  echo "                                                   # the middle."
  exit 1;
fi

raw_egs_dir=$1
dir=$2

# die on error or undefined variable.
set -e -u

if ! steps/chain2/validate_raw_egs.sh $raw_egs_dir; then
  echo "$0: failed to validate input directory $raw_egs_dir"
  exit 1
fi


mkdir -p $dir/temp $dir/log


if [ $stage -le 0 ]; then
  echo "$0: choosing heldout_subset and train_subset"

  utt2uniq_opt=
  if [ -f $raw_egs_dir/misc/utt2uniq ]; then
      utt2uniq_opt="--utt2uniq=$raw_egs_dir/misc/utt2uniq"
      echo "$0: File $raw_egs_dir/misc/utt2uniq exists, so ensuring the hold-out set" \
           "includes all perturbed versions of the same source utterance."
      utils/utt2spk_to_spk2utt.pl $raw_egs_dir/misc/utt2uniq 2>/dev/null | \
          utils/shuffle_list.pl 2>/dev/null | \
            awk -v max_utt=$num_utts_subset '{
                for (n=2;n<=NF;n++) print $n;
                printed += NF-1;
                if (printed >= max_utt) nextfile; }' \
          | fgrep -f - $raw_egs_dir/all.scp | sort -k1,1 > $dir/temp/heldout_subset.list
  else
      awk '{print $1}' $raw_egs_dir/misc/utt2spk | \
        utils/shuffle_list.pl 2>/dev/null | \
        head -$num_utts_subset |  fgrep -f - $raw_egs_dir/all.scp | sort -k1,1 > $dir/temp/heldout_subset.list
  fi

  awk '{print $1}' $raw_egs_dir/misc/utt2spk | \
     utils/filter_scp.pl --exclude $dir/temp/heldout_subset.list | \
     utils/shuffle_list.pl 2>/dev/null | \
     head -$num_utts_subset | fgrep -f - $raw_egs_dir/all.scp | sort -k1,1 > $dir/temp/train_subset.list

  awk '{print $1}' $raw_egs_dir/misc/utt2spk | \
     utils/filter_scp.pl --exclude $dir/temp/heldout_subset.list | fgrep -f - $raw_egs_dir/all.scp > $dir/temp/train.list
  fi
len_valid_uttlist=$(wc -l < $dir/temp/heldout_subset.list)
len_trainsub_uttlist=$(wc -l <$dir/temp/train_subset.list)

if [ $stage -le 1 ]; then

  for name in heldout_subset train_subset; do
    echo "$0: merging and shuffling $name egs"

    cp $dir/temp/${name}.list $dir/temp/${name}.scp

    $cmd $dir/log/shuffle_${name}_egs.log \
      nnet3-chain-shuffle-egs --srand=$srand scp:$dir/temp/${name}.scp ark,scp:$dir/${name}.ark,$dir/${name}.scp
  done

  # Split up the training list into multiple smaller lists, as it could be long.
  utils/split_scp.pl $dir/temp/train.list  $(for j in $(seq $nj); do echo $dir/temp/train.$j.scp; done)

  if [ -e $dir/storage ]; then
    # Make soft links to storage directories, if distributing this way..  See
    # utils/create_split_dir.pl.
    echo "$0: creating data links"
    utils/create_data_link.pl $(for j in $(seq $nj); do echo $dir/train.$j.ark; done) || true
  fi

  $cmd JOB=1:$nj $dir/log/shuffle_train_egs.JOB.log \
     nnet3-chain-shuffle-egs --buffer-size=$shuffle_buffer_size \
         --srand=\$[JOB+$srand] scp:$dir/temp/train.JOB.scp ark,scp:$dir/train.JOB.ark,$dir/train.JOB.scp || exit 1;
  cat $(for j in $(seq $nj); do echo $dir/train.$j.scp; done) > $dir/train.scp
fi

cat $raw_egs_dir/info.txt  | awk  -v num_repeats=$num_repeats \
   '
  /^dir_type / { print "dir_type processed_chain_egs"; next; }
  /^num_input_frames / { print "num_input_frames "$2 * num_repeats; next; } # approximate; ignores held-out egs.
  /^num_chunks / { print "num_chunks " $2 * num_repeats; next; }
   {print;}
  END{print "num_repeats " num_repeats;}' >$dir/info.txt



if ! cat $dir/info.txt | awk '{if (NF == 1) exit(1);}'; then
  echo "$0: we failed to obtain at least one of the fields in $dir/info.txt"
  exit 1
fi

cp -r $raw_egs_dir/misc/ $dir/


echo "$0: Finished processing egs"
