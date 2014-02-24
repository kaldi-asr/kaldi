#!/bin/bash

# Copyright 2012 Johns Hopkins University (Author: Daniel Povey)
# Copyright 2014 Vimal Manohar
# Apache 2.0.

# This script, which will generally be called from other neural-net training
# scripts, extracts the training examples used to train the neural net (and also
# the validation examples used for diagnostics), and puts them in separate archives.
# This is similar to the script steps/nnet2/get_egs.sh, but this also extracts 
# frames from unsupervsied data. Decode directory for unsupervised data which
# has the best path done along with posteriors (can be done using local/combine_posteriors.sh)

set -o pipefail

# Begin configuration section.
cmd=run.pl
feat_type=
num_utts_subset=300    # number of utterances in validation and training
                       # subsets used for shrinkage and diagnostics
num_valid_frames_combine=0 # #valid frames for combination weights at the very end.
num_train_frames_combine=10000 # # train frames for the above.
num_frames_diagnostic=4000 # number of frames for "compute_prob" jobs
samples_per_iter=400000 # each iteration of training, see this many samples
                        # per job.  This is just a guideline; it will pick a number
                        # that divides the number of samples in the entire data.
transform_dir_sup=     # If supplied, overrides alidir
transform_dir_unsup=   
num_jobs_nnet=16    # Number of neural net jobs to run in parallel
stage=-10
io_opts="-tc 5" # for jobs with a lot of I/O, limits the number running at one time. 
splice_width=4 # meaning +- 4 frames on each side for second LDA
spk_vecs_dir_sup=
spk_vecs_dir_unsup=
random_copy=false
weight_threshold=0.7    # Threshold on confidence factor of an unsupervised data 
                        # frame for it to not be ignored
supervised_copies=3     # Make x copies of supervised data.
use_frame_selection=true
use_frame_weights=false   # TODO: Not coded

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 6 ]; then
  echo "Usage: steps/nnet2/get_egs_semi_supervised.sh [opts] <supervised-data-dir> <unsupervised-data-dir> <lang> <supervised-ali-dir> <unsupervised-lat-dir> <exp-dir>"
  echo " e.g.: steps/nnet2/get_egs_semi_supervised.sh data/train data/train_unt data/lang exp/tri3_ali exp/tri3/dev_unt exp/tri4_nnet"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --num-jobs-nnet <num-jobs|16>                    # Number of parallel jobs to use for main neural net"
  echo "                                                   # training (will affect results as well as speed; try 8, 16)"
  echo "                                                   # Note: if you increase this, you may want to also increase"
  echo "                                                   # the learning rate."
  echo "  --samples-per-iter <#samples|400000>             # Number of samples of data to process per iteration, per"
  echo "                                                   # process."
  echo "  --splice-width <width|4>                         # Number of frames on each side to append for feature input"
  echo "                                                   # (note: we splice processed, typically 40-dimensional frames"
  echo "  --num-frames-diagnostic <#frames|4000>           # Number of frames used in computing (train,valid) diagnostics"
  echo "  --num-valid-frames-combine <#frames|10000>       # Number of frames used in getting combination weights at the"
  echo "                                                   # very end."
  echo "  --stage <stage|0>                                # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."
  echo "  --weight-threshold <weight|0.7>                  # Threshold on the confidence factor below which the "
  echo "                                                   # a frame of unsupervised training data is ignored."
  echo "  --supervised-copies <#copies|3>                  # Make copies of supervised data"
  echo "  --transform-dir-sup                              # Directory with transforms for supervised training data"
  echo "  --transform-dir-unsup                            # Directory with transforms for unsupervised training data"
  
  exit 1;
fi

data_sup=$1
data_unsup=$2
lang=$3
alidir=$4
latdir=$5
dir=$6

# Check some files.
for f in $data_sup/feats.scp $data_unsup/feats.scp $lang/L.fst $alidir/ali.1.gz $alidir/final.mdl $alidir/tree $latdir/best_path_ali.1.gz $latdir/weights.1.gz $latdir/../final.mdl $latdir/../tree; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

# Set some variables.
oov=`cat $lang/oov.int`
num_leaves=`tree-info $alidir/tree 2>/dev/null | awk '{print $2}'` || exit 1;
silphonelist=`cat $lang/phones/silence.csl` || exit 1;
[ -z $num_leaves ] && echo "\$num_leaves is unset" && exit 1
[ "$num_leaves" -eq "0" ] && echo "\$num_leaves is 0" && exit 1

nj_sup=`cat $alidir/num_jobs` || exit 1;  # number of jobs in alignment dir...
nj_unsup=`cat $latdir/num_jobs` || exit 1;  # number of jobs in decode dir
# in this dir we'll have just one job.
sdata_sup=$data_sup/split$nj_sup
sdata_unsup=$data_unsup/split$nj_unsup
utils/split_data.sh $data_sup $nj_sup
utils/split_data.sh $data_unsup $nj_unsup

mkdir -p $dir/log
echo $nj_sup > $dir/num_jobs_sup
echo $nj_unsup > $dir/num_jobs_unsup

cp $alidir/tree $dir

awk '{print $1}' $data_sup/utt2spk | utils/shuffle_list.pl | head -$num_utts_subset > $dir/valid_uttlist

# TODO (Vimal 22-Jan-14): Might need to deal unsupervised data separately 
if [ -f $data_sup/utt2uniq ]; then
  echo "File $data_sup/utt2uniq exists, so augmenting valid_uttlist to"
  echo "include all perturbed versions of the same 'real' utterances."
  mv $dir/valid_uttlist $dir/valid_uttlist.tmp
  utils/utt2spk_to_spk2utt.pl $data_sup/utt2uniq > $dir/uniq2utt
  cat $dir/valid_uttlist.tmp | utils/apply_map.pl $data_sup/utt2uniq | \
    sort | uniq | utils/apply_map.pl $dir/uniq2utt | \
    awk '{for(n=1;n<=NF;n++) print $n;}' | sort  > $dir/valid_uttlist
  rm $dir/uniq2utt $dir/valid_uttlist.tmp
fi

# TODO (Vimal 22-Jan-14): Might need to deal unsupervised data separately 
awk '{print $1}' $data_sup/utt2spk | utils/filter_scp.pl --exclude $dir/valid_uttlist | \
     head -$num_utts_subset > $dir/train_subset_uttlist

[ -z "$transform_dir_sup" ] && transform_dir_sup=$alidir
[ -z "$transform_dir_unsup" ] && transform_dir_unsup=$latdir
norm_vars=`cat $alidir/norm_vars 2>/dev/null` || norm_vars=false # cmn/cmvn option, default false.
norm_vars_unsup=`cat $latdir/norm_vars 2>/dev/null` || norm_vars_unsup=false

if [ "$norm_vars" != "$norm_vars_unsup" ]; then
  echo "ERROR: Features mismatch for supervised and unsupervised data!"
  echo "Variance normalization $norm_vars for supervised data vs $norm_vars_unsup for unsupervised data"
  exit 1
fi
cp $alidir/norm_vars $dir 2>/dev/null

## Set up features. 
if [ -z $feat_type ]; then
  if [ -f $alidir/final.mat ] && [ ! -f $transform_dir_sup/raw_trans.1 ]; then feat_type=lda; else feat_type=raw; fi
fi

echo "$0: feature type is $feat_type"

case $feat_type in
  raw) feats_sup="ark,s,cs:utils/filter_scp.pl --exclude $dir/valid_uttlist $sdata_sup/JOB/feats.scp | apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$sdata_sup/JOB/utt2spk scp:$sdata_sup/JOB/cmvn.scp scp:- ark:- |"
    feats_unsup="ark,s,cs:cat $sdata_unsup/JOB/feats.scp | apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$sdata_unsup/JOB/utt2spk scp:$sdata_unsup/JOB/cmvn.scp scp:- ark:- |"
    valid_feats="ark,s,cs:utils/filter_scp.pl $dir/valid_uttlist $data_sup/feats.scp | apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$data_sup/utt2spk scp:$data_sup/cmvn.scp scp:- ark:- |"
    train_subset_feats="ark,s,cs:utils/filter_scp.pl $dir/train_subset_uttlist $data_sup/feats.scp | apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$data_sup/utt2spk scp:$data_sup/cmvn.scp scp:- ark:- |"
   ;;
  lda) 
    splice_opts=`cat $alidir/splice_opts 2>/dev/null`
    #splice_opts_unsup=`cat $latdir/../splice_opts 2>/dev/null`
    #if [ "$splice_opts" -ne "$splice_opts_unsup" ]; then
    #  echo "ERROR: Features mismatch for supervised and unsupervised data!"
    #  echo "Splice options $splice_opts for supervised data vs $splice_opts_unsup for unsupervised data"
    #  exit 1
    #fi
    cp $alidir/splice_opts $dir/splice_opts 2>/dev/null
  
    #if [ "`diff $alidir/final.mat $latdir/../final.mat &> /dev/null; echo $?`" -ne "0" ]; then
    #  echo "ERROR: Features mismatch for supervised and unsupervised data!"
    #  echo "LDA matrices $alidir/final.mat for supervised data and $latdir/../final.mat for unsupervised data don't match"
    #  exit 1
    #fi

    cp $alidir/final.mat $dir    
    feats_sup="ark,s,cs:utils/filter_scp.pl --exclude $dir/valid_uttlist $sdata_sup/JOB/feats.scp | apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$sdata_sup/JOB/utt2spk scp:$sdata_sup/JOB/cmvn.scp scp:- ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
    feats_unsup="ark,s,cs:cat $sdata_unsup/JOB/feats.scp | apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$sdata_unsup/JOB/utt2spk scp:$sdata_unsup/JOB/cmvn.scp scp:- ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
    valid_feats="ark,s,cs:utils/filter_scp.pl $dir/valid_uttlist $data_sup/feats.scp | apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$data_sup/utt2spk scp:$data_sup/cmvn.scp scp:- ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
    train_subset_feats="ark,s,cs:utils/filter_scp.pl $dir/train_subset_uttlist $data_sup/feats.scp | apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$data_sup/utt2spk scp:$data_sup/cmvn.scp scp:- ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
    ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac

if [ -f $transform_dir_sup/trans.1 ] && [ $feat_type != "raw" ]; then
  echo "$0: using transforms from $transform_dir_sup for supervised data"
  feats_sup="$feats_sup transform-feats --utt2spk=ark:$sdata_sup/JOB/utt2spk ark:$transform_dir_sup/trans.JOB ark:- ark:- |"
  valid_feats="$valid_feats transform-feats --utt2spk=ark:$data_sup/utt2spk 'ark:cat $transform_dir_sup/trans.*|' ark:- ark:- |"
  train_subset_feats="$train_subset_feats transform-feats --utt2spk=ark:$data_sup/utt2spk 'ark:cat $transform_dir_sup/trans.*|' ark:- ark:- |"
fi
if [ -f $transform_dir_sup/raw_trans.1 ] && [ $feat_type == "raw" ]; then
  echo "$0: using raw-fMLLR transforms from $transform_dir_sup"
  feats_sup="$feats_sup transform-feats --utt2spk=ark:$sdata_sup/JOB/utt2spk ark:$transform_dir_sup/raw_trans.JOB ark:- ark:- |"
  valid_feats="$valid_feats transform-feats --utt2spk=ark:$data_sup/utt2spk 'ark:cat $transform_dir_sup/raw_trans.*|' ark:- ark:- |"
  train_subset_feats="$train_subset_feats transform-feats --utt2spk=ark:$data_sup/utt2spk 'ark:cat $transform_dir_sup/raw_trans.*|' ark:- ark:- |"
fi

if [ -f $transform_dir_unsup/trans.1 ] && [ $feat_type != "raw" ]; then
  echo "$0: using transforms from $transform_dir_unsup for supervised data"
  feats_unsup="$feats_unsup transform-feats --utt2spk=ark:$sdata_unsup/JOB/utt2spk ark:$transform_dir_unsup/trans.JOB ark:- ark:- |"
fi
if [ -f $transform_dir_unsup/raw_trans.1 ] && [ $feat_type == "raw" ]; then
  echo "$0: using raw-fMLLR transforms from $transform_dir_unsup"
  feats_unsup="$feats_unsup transform-feats --utt2spk=ark:$sdata_unsup/JOB/utt2spk ark:$transform_dir_unsup/raw_trans.JOB ark:- ark:- |"
fi

if [ $stage -le 0 ]; then
  echo "$0: working out number of frames of training data"
  num_sup_frames=`feat-to-len scp:$data_sup/feats.scp ark,t:- | awk '{x += $2;} END{print x;}'` || exit 1;
  num_unsup_frames=`feat-to-len scp:$data_unsup/feats.scp ark,t:- | awk '{x += $2;} END{print x;}'` || exit 1;
  num_frames=$(perl -e "print STDOUT ($num_sup_frames * $supervised_copies + $num_unsup_frames)")
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

# Making soft links to storage directories.
for x in `seq 1 $num_jobs_nnet`; do
  for y in `seq 0 $[$iters_per_epoch-1]`; do
    utils/create_data_link.pl $dir/egs/egs.$x.$y.ark
    utils/create_data_link.pl $dir/egs/egs_tmp.$x.$y.ark
  done
  for y in `seq 1 $nj_sup`; do
    utils/create_data_link.pl $dir/egs/egs_orig.$x.$y.ark
  done
done

nnet_context_opts="--left-context=$splice_width --right-context=$splice_width"
mkdir -p $dir/egs

if [ ! -z $spk_vecs_dir_sup ]; then
  [ ! -f $spk_vecs_dir_sup/vecs.1 ] && echo "No such file $spk_vecs_dir_sup/vecs.1" && exit 1;
  spk_vecs_opt_sup=("--spk-vecs=ark:cat $spk_vecs_dir_sup/vecs.*|" "--utt2spk=ark:$data_sup/utt2spk")
else
  spk_vecs_opt_sup=()
fi

if [ ! -z $spk_vecs_dir_unsup ]; then
  [ ! -f $spk_vecs_dir_unsup/vecs.1 ] && echo "No such file $spk_vecs_dir_unsup/vecs.1" && exit 1;
  spk_vecs_opt_unsup=("--spk-vecs=ark:cat $spk_vecs_dir_unsup/vecs.*|" "--utt2spk=ark:$data_unsup/utt2spk")
else
  spk_vecs_opt_unsup=()
fi

if [ $stage -le 2 ]; then
  echo "Getting validation and training subset examples."
  rm $dir/.error 2>/dev/null
  $cmd $dir/log/create_valid_subset.log \
    nnet-get-egs $nnet_context_opts "${spk_vecs_opt_sup[@]}" "$valid_feats" \
     "ark,cs:gunzip -c $alidir/ali.*.gz | ali-to-pdf $alidir/final.mdl ark:- ark:- | ali-to-post ark:- ark:- |" \
     "ark:$dir/egs/valid_all.egs" || touch $dir/.error &
  $cmd $dir/log/create_train_subset.log \
    nnet-get-egs $nnet_context_opts "${spk_vecs_opt_sup[@]}" "$train_subset_feats" \
     "ark,cs:gunzip -c $alidir/ali.*.gz | ali-to-pdf $alidir/final.mdl ark:- ark:- | ali-to-post ark:- ark:- |" \
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
  cat $dir/egs/valid_combine.egs $dir/egs/train_combine.egs > $dir/egs/combine.egs

  for f in $dir/egs/{combine,train_diagnostic,valid_diagnostic}.egs; do
    [ ! -s $f ] && echo "No examples in file $f" && exit 1;
  done
  rm $dir/egs/valid_all.egs $dir/egs/train_subset_all.egs $dir/egs/{train,valid}_combine.egs
fi

if [ $stage -le 3 ]; then
  mkdir -p $dir/temp

  # Other scripts might need to know the following info:
  echo $num_jobs_nnet >$dir/egs/num_jobs_nnet
  echo $iters_per_epoch >$dir/egs/iters_per_epoch
  echo $samples_per_iter_real >$dir/egs/samples_per_iter

  echo "Creating training examples";
  # in $dir/egs, create $num_jobs_nnet separate files with training examples.
  # The order is not randomized at this point.

  echo "Generating training examples on disk"
  # The examples will go round-robin to egs_list.

  egs_list=
  for n in `seq 1 $num_jobs_nnet`; do
    egs_list="$egs_list ark:$dir/egs/egs_orig.$n.JOB.ark"
  done

  $cmd $io_opts JOB=1:$nj_unsup $dir/log/get_weighted_egs.JOB.log \
    nnet-get-weighted-egs $nnet_context_opts "${spk_vecs_opt_unsup[@]}" \
    --weight-threshold=$weight_threshold --use-frame-weights=$use_frame_weights \
    --use-frame-selection=$use_frame_selection "$feats_unsup" \
    "ark,s,cs:gunzip -c $latdir/best_path_ali.JOB.gz | convert-ali $latdir/../final.mdl $alidir/final.mdl $dir/tree ark:- ark:- | ali-to-pdf $alidir/final.mdl ark:- ark:- | ali-to-post ark:- ark:- |" \
    "ark,s,cs:gunzip -c $latdir/weights.JOB.gz |" ark:- \| \
    nnet-copy-egs ark:- $egs_list || exit 1;

  for (( i=0; i<supervised_copies; i++ )); do
    nj_start=$((nj_unsup + i * nj_sup))
    nj_end=$((nj_unsup + (i+1) * nj_sup))
    
    egs_list=
    for n in `seq 1 $num_jobs_nnet`; do
      egs_list="$egs_list ark:$dir/egs/egs_orig.$n.$i.JOB.ark"
    done
    
    $cmd $io_opts JOB=1:$nj_sup $dir/log/get_egs.$i.JOB.log \
      nnet-get-egs $nnet_context_opts "${spk_vecs_opt_sup[@]}" "$feats_sup" \
      "ark,s,cs:gunzip -c $alidir/ali.JOB.gz | ali-to-pdf $alidir/final.mdl ark:- ark:- | ali-to-post ark:- ark:- |" ark:- \| \
      nnet-copy-egs --random=true --srand=\$\(\(JOB+$nj_sup*$i\)\) ark:- $egs_list || exit 1;

    #for n in `seq 1 $num_jobs_nnet`; do 
    #  for j in `seq 1 $nj_sup`; do
    #    mv $dir/egs/egs_orig.$n.$i.$j.ark $dir/egs/egs_orig.$n.$((nj_unsup+i*nj_sup+j)).ark || exit 1
    #  done
    #done
  done;
fi

if [ $stage -le 4 ]; then
  echo "$0: rearranging examples into parts for different parallel jobs"
  # combine all the "egs_orig.JOB.*.scp" (over the splits of the data) and
  # then split into multiple parts egs.JOB.*.scp for different parts of the
  # data, 0 .. $iters_per_epoch-1.

  if [ $iters_per_epoch -eq 1 ]; then
    echo "$0: Since iters-per-epoch == 1, just concatenating the data."
    for n in `seq 1 $num_jobs_nnet`; do
      cat $dir/egs/egs_orig.$n.*.ark > $dir/egs/egs_tmp.$n.0.ark || exit 1;
      rm $dir/egs/egs_orig.$n.*.ark  # don't "|| exit 1", due to NFS bugs...
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
        "ark:cat $dir/egs/egs_orig.JOB.*.ark|" $egs_list '&&' \
        '(' rm $dir/egs/egs_orig.JOB.*.ark '||' true ')' || exit 1;
  fi
fi

if [ $stage -le 5 ]; then
  # Next, shuffle the order of the examples in each of those files.
  # Each one should not be too large, so we can do this in memory.
  echo "Shuffling the order of training examples"
  echo "(in order to avoid stressing the disk, these won't all run at once)."


  # note, the "|| true" below is a workaround for NFS bugs
  # we encountered running this script with Debian-7, NFS-v4.
  for n in `seq 0 $[$iters_per_epoch-1]`; do
    $cmd $io_opts JOB=1:$num_jobs_nnet $dir/log/shuffle.$n.JOB.log \
      nnet-shuffle-egs "--srand=\$[JOB+($num_jobs_nnet*$n)]" \
      ark:$dir/egs/egs_tmp.JOB.$n.ark ark:$dir/egs/egs.JOB.$n.ark '&&' \
      '(' rm $dir/egs/egs_tmp.JOB.$n.ark '||' true ')' || exit 1;
  done
fi

echo "$0: Finished preparing training examples"
