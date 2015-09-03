#!/bin/bash

# Copyright 2012-2014 Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# This is modified from ../../nnet2/get_egs.sh. 
# This script combines the
# nnet-example extraction with the feature extraction directly from wave files;
# it uses the program online2-wav-dump-feature to do all parts of feature
# extraction: MFCC/PLP/fbank, possibly plus pitch, plus iVectors.  This script
# is intended mostly for cross-system training for online decoding, where you
# initialize the nnet from an existing, larger system.


# Begin configuration section.
cmd=run.pl
num_utts_subset=300    # number of utterances in validation and training
                       # subsets used for shrinkage and diagnostics
num_valid_frames_combine=0 # #valid frames for combination weights at the very end.
num_train_frames_combine=10000 # # train frames for the above.
num_frames_diagnostic=4000 # number of frames for "compute_prob" jobs
samples_per_iter=400000 # each iteration of training, see this many samples
                        # per job.  This is just a guideline; it will pick a number
                        # that divides the number of samples in the entire data.
transform_dir=     # If supplied, overrides alidir
num_jobs_nnet=16    # Number of neural net jobs to run in parallel
stage=0
io_opts="-tc 5" # for jobs with a lot of I/O, limits the number running at one time. 
random_copy=false

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 4 ]; then
  echo "Usage: steps/online/nnet2/get_egs.sh [opts] <data> <ali-dir> <online-nnet-dir> <exp-dir>"
  echo " e.g.: steps/online/nnet2/get_egs.sh data/train exp/tri3_ali exp/nnet2_online/nnet_a_gpu_online/ exp/tri4_nnet"
  echo "In <online-nnet-dir>, it looks for final.mdl (need to compute required left and right context),"
  echo "and a configuration file conf/online_nnet2_decoding.conf which describes the features."
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --cmd (utils/run.pl;utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --num-jobs-nnet <num-jobs;16>                    # Number of parallel jobs to use for main neural net"
  echo "                                                   # training (will affect results as well as speed; try 8, 16)"
  echo "                                                   # Note: if you increase this, you may want to also increase"
  echo "                                                   # the learning rate."
  echo "  --samples-per-iter <#samples;400000>             # Number of samples of data to process per iteration, per"
  echo "                                                   # process."
  echo "  --feat-type <lda|raw>                            # (by default it tries to guess).  The feature type you want"
  echo "                                                   # to use as input to the neural net."
  echo "  --splice-width <width;4>                         # Number of frames on each side to append for feature input"
  echo "                                                   # (note: we splice processed, typically 40-dimensional frames"
  echo "  --num-frames-diagnostic <#frames;4000>           # Number of frames used in computing (train,valid) diagnostics"
  echo "  --num-valid-frames-combine <#frames;10000>       # Number of frames used in getting combination weights at the"
  echo "                                                   # very end."
  echo "  --stage <stage|0>                                # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."
  
  exit 1;
fi

data=$1
alidir=$2
online_nnet_dir=$3
dir=$4


mdl=$online_nnet_dir/final.mdl # only needed for left and right context.
feature_conf=$online_nnet_dir/conf/online_nnet2_decoding.conf

for f in $data/wav.scp $alidir/ali.1.gz $alidir/final.mdl $alidir/tree $feature_conf $mdl; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

nj=`cat $alidir/num_jobs` || exit 1;  # number of jobs in alignment dir...

sdata=$data/split$nj
utils/split_data.sh $data $nj

mkdir -p $dir/log
cp $alidir/tree $dir
grep -v '^--endpoint' $feature_conf >$dir/feature.conf || exit 1;

# Get list of validation utterances. 
mkdir -p $dir/valid $dir/train_subset

awk '{print $1}' $data/utt2spk | utils/shuffle_list.pl | head -$num_utts_subset \
    > $dir/valid/uttlist || exit 1;

if [ -f $data/utt2uniq ]; then
  echo "File $data/utt2uniq exists, so augmenting valid/uttlist to"
  echo "include all perturbed versions of the same 'real' utterances."
  mv $dir/valid/uttlist $dir/valid/uttlist.tmp
  utils/utt2spk_to_spk2utt.pl $data/utt2uniq > $dir/uniq2utt
  cat $dir/valid/uttlist.tmp | utils/apply_map.pl $data/utt2uniq | \
    sort | uniq | utils/apply_map.pl $dir/uniq2utt | \
    awk '{for(n=1;n<=NF;n++) print $n;}' | sort  > $dir/valid/uttlist
  rm $dir/uniq2utt $dir/valid/uttlist.tmp
fi

awk '{print $1}' $data/utt2spk | utils/filter_scp.pl --exclude $dir/valid/uttlist | \
   utils/shuffle_list.pl | head -$num_utts_subset > $dir/train_subset/uttlist || exit 1;


for subdir in valid train_subset; do
  # In order for the iVector extraction to work right, we need to process all
  # utterances of the speakers which have utterances in valid/uttlist, and the
  # same for train_subset/uttlist.  We produce $dir/valid/uttlist_extended which
  # will contain all utterances of all speakers which have utterances in
  # $dir/valid/uttlist, and the same for $dir/train_subset/.

  utils/filter_scp.pl $dir/$subdir/uttlist <$data/utt2spk | awk '{print $2}' > $dir/$subdir/spklist || exit 1;
  utils/filter_scp.pl -f 2 $dir/$subdir/spklist <$data/utt2spk >$dir/$subdir/utt2spk || exit 1;
  utils/utt2spk_to_spk2utt.pl <$dir/$subdir/utt2spk >$dir/$subdir/spk2utt || exit 1;
  awk '{print $1}' <$dir/$subdir/utt2spk >$dir/$subdir/uttlist_extended || exit 1;
  rm $dir/$subdir/spklist
done

if [ -f $data/segments ]; then
  # note: in the feature extraction, because the program online2-wav-dump-features is sensitive to the
  # previous utterances within a speaker, we do the filtering after extracting the features.
  echo "$0 [info]: segments file exists: using that."
  feats="ark,s,cs:extract-segments scp:$sdata/JOB/wav.scp $sdata/JOB/segments ark:- | online2-wav-dump-features --config=$dir/feature.conf ark:$sdata/JOB/spk2utt ark,s,cs:- ark:- | subset-feats --exclude=$dir/valid/uttlist ark:- ark:- |"
  valid_feats="ark,s,cs:utils/filter_scp.pl $dir/valid/uttlist_extended $data/segments  | extract-segments scp:$data/wav.scp - ark:- | online2-wav-dump-features --config=$dir/feature.conf ark:$dir/valid/spk2utt ark,s,cs:- ark:- | subset-feats --include=$dir/valid/uttlist ark:- ark:- |"
  train_subset_feats="ark,s,cs:utils/filter_scp.pl $dir/train_subset/uttlist_extended $data/segments  | extract-segments scp:$data/wav.scp - ark:- | online2-wav-dump-features --config=$dir/feature.conf ark:$dir/train_subset/spk2utt ark,s,cs:- ark:- | subset-feats --include=$dir/train_subset/uttlist ark:- ark:- |"
else
  echo "$0 [info]: no segments file exists, using wav.scp."
  feats="ark,s,cs:online2-wav-dump-features --config=$dir/feature.conf ark:$sdata/JOB/spk2utt scp:$sdata/JOB/wav.scp ark:- | subset-feats --exclude=$dir/valid/uttlist ark:- ark:- |"
  valid_feats="ark,s,cs:utils/filter_scp.pl $dir/valid/uttlist_extended $data/wav.scp | online2-wav-dump-features --config=$dir/feature.conf ark:$dir/valid/spk2utt scp:- ark:- | subset-feats --include=$dir/valid/uttlist ark:- ark:- |"
  train_subset_feats="ark,s,cs:utils/filter_scp.pl $dir/train_subset/uttlist_extended $data/wav.scp | online2-wav-dump-features --config=$dir/feature.conf ark:$dir/train_subset/spk2utt scp:- ark:- | subset-feats --include=$dir/train_subset/uttlist ark:- ark:- |"
fi

ivector_dim=$(online2-wav-dump-features --config=$dir/feature.conf --print-ivector-dim=true) || exit 1;

! [ $ivector_dim -ge 0 ] && echo "$0: error getting iVector dim" && exit 1;


if [ $stage -le 0 ]; then
  echo "$0: working out number of frames of training data"
  num_frames=$(steps/nnet2/get_num_frames.sh $data)
  echo $num_frames > $dir/num_frames
else
  num_frames=`cat $dir/num_frames` || exit 1;
fi

# Working out number of iterations per epoch.
iters_per_epoch=`perl -e "print int($num_frames/($samples_per_iter * $num_jobs_nnet) + 0.5);"` || exit 1;
[ $iters_per_epoch -eq 0 ] && iters_per_epoch=1
samples_per_iter_real=$[$num_frames/($num_jobs_nnet*$iters_per_epoch)]
echo "$0: Every epoch, splitting the data up into $iters_per_epoch iterations,"
echo "$0: giving samples-per-iteration of $samples_per_iter_real (you requested $samples_per_iter)."

# Making soft links to storage directories.  This is a no-up unless
# the subdirectory $dir/egs/storage/ exists.  See utils/create_split_dir.pl
for x in `seq 1 $num_jobs_nnet`; do
  for y in `seq 0 $[$iters_per_epoch-1]`; do
    utils/create_data_link.pl $dir/egs/egs.$x.$y.ark
    utils/create_data_link.pl $dir/egs/egs_tmp.$x.$y.ark
  done
  for y in `seq 1 $nj`; do
    utils/create_data_link.pl $dir/egs/egs_orig.$x.$y.ark
  done
done

remove () { for x in $*; do [ -L $x ] && rm $(readlink -f $x); rm $x; done }

set -o pipefail
left_context=$(nnet-am-info $mdl | grep '^left-context' | awk '{print $2}') || exit 1;
right_context=$(nnet-am-info $mdl | grep '^right-context' | awk '{print $2}') || exit 1;
nnet_context_opts="--left-context=$left_context --right-context=$right_context"
set +o pipefail

mkdir -p $dir/egs

if [ $stage -le 2 ]; then
  rm $dir/.error 2>/dev/null
  
  echo "$0: extracting validation and training-subset alignments."
  set -o pipefail;
  for id in $(seq $nj); do gunzip -c $alidir/ali.$id.gz; done | \
    copy-int-vector ark:- ark,t:- | \
    utils/filter_scp.pl <(cat $dir/valid/uttlist $dir/train_subset/uttlist) | \
    gzip -c >$dir/ali_special.gz || exit 1;
  set +o pipefail; # unset the pipefail option.
  
  echo "Getting validation and training subset examples."
  $cmd $dir/log/create_valid_subset.log \
    nnet-get-egs $ivectors_opt $nnet_context_opts "$valid_feats" \
     "ark,s,cs:gunzip -c $dir/ali_special.gz | ali-to-pdf $alidir/final.mdl ark:- ark:- | ali-to-post ark:- ark:- |" \
     "ark:$dir/egs/valid_all.egs" || touch $dir/.error &
  $cmd $dir/log/create_train_subset.log \
    nnet-get-egs $ivectors_opt $nnet_context_opts "$train_subset_feats" \
    "ark,s,cs:gunzip -c $dir/ali_special.gz | ali-to-pdf $alidir/final.mdl ark:- ark:- | ali-to-post ark:- ark:- |" \
     "ark:$dir/egs/train_subset_all.egs" || touch $dir/.error &
  wait;
  [ -f $dir/.error ] && exit 1;
  echo "Getting subsets of validation examples for diagnostics and combination."
  $cmd $dir/log/create_valid_subset_combine.log \
    nnet-subset-egs --n=$num_valid_frames_combine ark:$dir/egs/valid_all.egs \
        ark:$dir/egs/valid_combine.egs || touch $dir/.error &
  $cmd $dir/log/create_valid_subset_diagnostic.log \
    nnet-subset-egs --n=$num_frames_diagnostic ark:$dir/egs/valid_all.egs \
    ark:$dir/egs/valid_diagnostic.egs || touch $dir/.error &

  $cmd $dir/log/create_train_subset_combine.log \
    nnet-subset-egs --n=$num_train_frames_combine ark:$dir/egs/train_subset_all.egs \
    ark:$dir/egs/train_combine.egs || touch $dir/.error &
  $cmd $dir/log/create_train_subset_diagnostic.log \
    nnet-subset-egs --n=$num_frames_diagnostic ark:$dir/egs/train_subset_all.egs \
    ark:$dir/egs/train_diagnostic.egs || touch $dir/.error &
  wait
  [ -f $dir/.error ] && echo "Error detected while creating egs" && exit 1;
  cat $dir/egs/valid_combine.egs $dir/egs/train_combine.egs > $dir/egs/combine.egs

  for f in $dir/egs/{combine,train_diagnostic,valid_diagnostic}.egs; do
    [ ! -s $f ] && echo "No examples in file $f" && exit 1;
  done
  rm $dir/egs/valid_all.egs $dir/egs/train_subset_all.egs $dir/egs/{train,valid}_combine.egs $dir/ali_special.gz
fi

if [ $stage -le 3 ]; then

  # Other scripts might need to know the following info:
  echo $num_jobs_nnet >$dir/egs/num_jobs_nnet
  echo $iters_per_epoch >$dir/egs/iters_per_epoch
  echo $samples_per_iter_real >$dir/egs/samples_per_iter

  echo "Creating training examples";
  # in $dir/egs, create $num_jobs_nnet separate files with training examples.
  # The order is not randomized at this point.

  egs_list=
  for n in `seq 1 $num_jobs_nnet`; do
    egs_list="$egs_list ark:$dir/egs/egs_orig.$n.JOB.ark"
  done
  echo "Generating training examples on disk"
  # The examples will go round-robin to egs_list.
  $cmd $io_opts JOB=1:$nj $dir/log/get_egs.JOB.log \
    nnet-get-egs $ivectors_opt $nnet_context_opts "$feats" \
    "ark,s,cs:gunzip -c $alidir/ali.JOB.gz | ali-to-pdf $alidir/final.mdl ark:- ark:- | ali-to-post ark:- ark:- |" ark:- \| \
    nnet-copy-egs ark:- $egs_list || exit 1;
fi

if [ $stage -le 4 ]; then
  echo "$0: rearranging examples into parts for different parallel jobs"
  # combine all the "egs_orig.JOB.*.scp" (over the $nj splits of the data) and
  # then split into multiple parts egs.JOB.*.scp for different parts of the
  # data, 0 .. $iters_per_epoch-1.

  if [ $iters_per_epoch -eq 1 ]; then
    echo "$0: Since iters-per-epoch == 1, just concatenating the data."
    for n in `seq 1 $num_jobs_nnet`; do
      cat $dir/egs/egs_orig.$n.*.ark > $dir/egs/egs_tmp.$n.0.ark || exit 1;
      remove $dir/egs/egs_orig.$n.*.ark 
    done
  else # We'll have to split it up using nnet-copy-egs.
    egs_list=
    for n in `seq 0 $[$iters_per_epoch-1]`; do
      egs_list="$egs_list ark:$dir/egs/egs_tmp.JOB.$n.ark"
    done
    # note, the "|| true" below is a workaround for NFS bugs
    # we encountered running this script with Debian-7, NFS-v4.
    $cmd $io_opts JOB=1:$num_jobs_nnet $dir/log/split_egs.JOB.log \
      nnet-copy-egs --random=$random_copy --srand=JOB \
        "ark:cat $dir/egs/egs_orig.JOB.*.ark|" $egs_list || exit 1;
    remove $dir/egs/egs_orig.*.*.ark  2>/dev/null
  fi
fi

if [ $stage -le 5 ]; then
  # Next, shuffle the order of the examples in each of those files.
  # Each one should not be too large, so we can do this in memory.
  echo "Shuffling the order of training examples"
  echo "(in order to avoid stressing the disk, these won't all run at once)."

  for n in `seq 0 $[$iters_per_epoch-1]`; do
    $cmd $io_opts JOB=1:$num_jobs_nnet $dir/log/shuffle.$n.JOB.log \
      nnet-shuffle-egs "--srand=\$[JOB+($num_jobs_nnet*$n)]" \
      ark:$dir/egs/egs_tmp.JOB.$n.ark ark:$dir/egs/egs.JOB.$n.ark 
    remove $dir/egs/egs_tmp.*.$n.ark
  done
fi

echo "$0: Finished preparing training examples"
