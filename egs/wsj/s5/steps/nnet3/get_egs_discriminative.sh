#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.
# Copyright 2014-2015   Vimal Manohar

# This script dumps examples MPE or MMI or state-level minimum bayes risk (sMBR)
# training of neural nets.
# Criterion supported are mpe, smbr and mmi

# Begin configuration section.
cmd=run.pl
feat_type=raw     # set it to 'lda' to use LDA features.
frames_per_eg=150 # number of frames of labels per example.  more->less disk space and
                  # less time preparing egs, but more I/O during training.
                  # note: the script may reduce this if reduce_frames_per_eg is true.
frames_overlap_per_eg=30 # number of supervised frames of overlap that we aim for per eg.
                  # can be useful to avoid wasted data if you're using --left-deriv-truncate
                  # and --right-deriv-truncate.
frame_subsampling_factor=1 # ratio between input and output frame-rate of nnet.
                           # this should be read from the nnet. For now, it is taken as an option
left_context=4    # amount of left-context per eg (i.e. extra frames of input features
                  # not present in the output supervision).
right_context=4   # amount of right-context per eg.
valid_left_context=   # amount of left_context for validation egs, typically used in
                      # recurrent architectures to ensure matched condition with
                      # training egs
valid_right_context=  # amount of right_context for validation egs
adjust_priors=true
priors_left_context=   # amount of left_context for priors egs
priors_right_context=   # amount of right_context for priors egs
compress=true   # set this to false to disable compression (e.g. if you want to see whether
                # results are affected).
num_utts_subset=80     # number of utterances in validation and training
                        # subsets used for shrinkage and diagnostics.

frames_per_iter=400000 # each iteration of training, see this many frames
                       # per job.  This is just a guideline; it will pick a number
                       # that divides the number of samples in the entire data.

determinize=true
minimize=true
remove_output_symbols=true
remove_epsilons=true
collapse_transition_ids=true
acwt=0.1

stage=0
max_jobs_run=15
max_shuffle_jobs_run=15

transform_dir= # If this is a SAT system, directory for transforms
online_ivector_dir=
cmvn_opts=  # can be used for specifying CMVN options, if feature type is not lda (if lda,
            # it doesn't make sense to use different options than were used as input to the
            # LDA transform).  This is used to turn off CMVN in the online-nnet experiments.

num_priors_subset=100
num_archives_priors=10

# End configuration section.


echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 6 ]; then
  echo "Usage: $0 [opts] <data> <lang> <ali-dir> <denlat-dir> <src-model-file> <degs-dir>"
  echo " e.g.: $0 data/train data/lang exp/tri3_ali exp/tri4_nnet_denlats exp/tri4/final.mdl exp/tri4_mpe/degs"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs (probably would be good to add -tc 5 or so if using"
  echo "                                                   # GridEngine (to avoid excessive NFS traffic)."
  echo "  --samples-per-iter <#samples|400000>             # Number of samples of data to process per iteration, per"
  echo "                                                   # process."
  echo "  --stage <stage|-8>                               # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."
  echo "  --online-ivector-dir <dir|"">                    # Directory for online-estimated iVectors, used in the"
  echo "                                                   # online-neural-net setup."
  exit 1;
fi

data=$1
lang=$2
alidir=$3
denlatdir=$4
src_model=$5
dir=$6

extra_files=
[ ! -z $online_ivector_dir ] && \
  extra_files="$online_ivector_dir/ivector_period $online_ivector_dir/ivector_online.scp"

# Check some files.
for f in $data/feats.scp $lang/L.fst $alidir/ali.1.gz $alidir/num_jobs $alidir/tree \
         $denlatdir/lat.1.gz $denlatdir/num_jobs $src_model $extra_files; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

mkdir -p $dir/log $dir/info || exit 1;

utils/lang/check_phones_compatible.sh $lang/phones.txt $alidir/phones.txt || exit 1;
cp $lang/phones.txt $dir || exit 1;

nj=$(cat $denlatdir/num_jobs) || exit 1;

sdata=$data/split$nj
utils/split_data.sh $data $nj

# Get list of validation utterances.
awk '{print $1}' $data/utt2spk | utils/shuffle_list.pl | head -$num_utts_subset \
    > $dir/valid_uttlist || exit 1;

if [ -f $data/utt2uniq ]; then  # this matters if you use data augmentation.
  echo "File $data/utt2uniq exists, so augmenting valid_uttlist to"
  echo "include all perturbed versions of the same 'real' utterances."
  mv $dir/valid_uttlist $dir/valid_uttlist.tmp
  utils/utt2spk_to_spk2utt.pl $data/utt2uniq > $dir/uniq2utt
  cat $dir/valid_uttlist.tmp | utils/apply_map.pl $data/utt2uniq | \
    sort | uniq | utils/apply_map.pl $dir/uniq2utt | \
    awk '{for(n=1;n<=NF;n++) print $n;}' | sort  > $dir/valid_uttlist
  rm $dir/uniq2utt $dir/valid_uttlist.tmp
fi

awk '{print $1}' $data/utt2spk | utils/filter_scp.pl --exclude $dir/valid_uttlist | \
   utils/shuffle_list.pl | head -$num_utts_subset > $dir/train_subset_uttlist || exit 1;

[ -z "$transform_dir" ] && transform_dir=$alidir

if [ $stage -le 1 ]; then
  nj_ali=$(cat $alidir/num_jobs)
  alis=$(for n in $(seq $nj_ali); do echo -n "$alidir/ali.$n.gz "; done)
  $cmd $dir/log/copy_alignments.log \
    copy-int-vector "ark:gunzip -c $alis|" \
    ark,scp:$dir/ali.ark,$dir/ali.scp || exit 1;
fi

prior_ali_rspecifier="ark,s,cs:utils/filter_scp.pl $dir/priors_uttlist $dir/ali.scp | ali-to-pdf $alidir/final.mdl scp:- ark:- |"

if [ -f $transform_dir/trans.1 ] && [ $feat_type != "raw" ]; then
  echo "$0: using transforms from $transform_dir"
  if [ $stage -le 0 ]; then
    $cmd $dir/log/copy_transforms.log \
      copy-feats "ark:cat $transform_dir/trans.* |" "ark,scp:$dir/trans.ark,$dir/trans.scp"
  fi
fi
if [ -f $transform_dir/raw_trans.1 ] && [ $feat_type == "raw" ]; then
  echo "$0: using raw transforms from $transform_dir"
  if [ $stage -le 0 ]; then
    $cmd $dir/log/copy_transforms.log \
      copy-feats "ark:cat $transform_dir/raw_trans.* |" "ark,scp:$dir/trans.ark,$dir/trans.scp"
  fi
fi

silphonelist=`cat $lang/phones/silence.csl` || exit 1;
cp $alidir/tree $dir
cp $lang/phones/silence.csl $dir/info/
cp $src_model $dir/final.mdl || exit 1

# Get list of utterances for prior computation.
awk '{print $1}' $data/utt2spk | utils/filter_scp.pl --exclude $dir/valid_uttlist | \
  utils/shuffle_list.pl | head -$num_priors_subset \
  > $dir/priors_uttlist || exit 1;

## We don't support deltas here, only LDA or raw (mainly because deltas are less
## frequently used).
if [ -z $feat_type ]; then
  if [ -f $alidir/final.mat ] && [ ! -f $transform_dir/raw_trans.1 ]; then feat_type=lda; else feat_type=raw; fi
fi
echo "$0: feature type is $feat_type"

case $feat_type in
  raw) feats="ark,s,cs:utils/filter_scp.pl --exclude $dir/valid_uttlist $sdata/JOB/feats.scp | apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:- ark:- |"
    valid_feats="ark,s,cs:utils/filter_scp.pl $dir/valid_uttlist $data/feats.scp | apply-cmvn $cmvn_opts --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:- ark:- |"
    train_subset_feats="ark,s,cs:utils/filter_scp.pl $dir/train_subset_uttlist $data/feats.scp | apply-cmvn $cmvn_opts --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:- ark:- |"
    priors_feats="ark,s,cs:utils/filter_scp.pl $dir/priors_uttlist $data/feats.scp | apply-cmvn $cmvn_opts --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:- ark:- |"
    echo $cmvn_opts > $dir/cmvn_opts
   ;;
  lda)
    splice_opts=`cat $alidir/splice_opts 2>/dev/null`
    cp $alidir/splice_opts $dir 2>/dev/null
    cp $alidir/final.mat $dir
    [ ! -z "$cmvn_opts" ] && \
       echo "You cannot supply --cmvn-opts option if feature type is LDA." && exit 1;
    cmvn_opts=`cat $alidir/cmvn_opts 2>/dev/null`
    cp $alidir/cmvn_opts $dir 2>/dev/null
    feats="ark,s,cs:utils/filter_scp.pl --exclude $dir/valid_uttlist $sdata/JOB/feats.scp | apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:- ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
    valid_feats="ark,s,cs:utils/filter_scp.pl $dir/valid_uttlist $data/feats.scp | apply-cmvn $cmvn_opts --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:- ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
    train_subset_feats="ark,s,cs:utils/filter_scp.pl $dir/train_subset_uttlist $data/feats.scp | apply-cmvn $cmvn_opts --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:- ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
    priors_feats="ark,s,cs:utils/filter_scp.pl $dir/priors_uttlist $data/feats.scp | apply-cmvn $cmvn_opts --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:- ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
    ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac

if [ -f $dir/trans.scp ]; then
  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk scp:$dir/trans.scp ark:- ark:- |"
  valid_feats="$valid_feats transform-feats --utt2spk=ark:$data/utt2spk scp:$dir/trans.scp|' ark:- ark:- |"
  train_subset_feats="$train_subset_feats transform-feats --utt2spk=ark:$data/utt2spk scp:$dir/trans.scp|' ark:- ark:- |"
  priors_feats="$priors_feats transform-feats --utt2spk=ark:$data/utt2spk scp:$dir/trans.scp|' ark:- ark:- |"
fi

if [ ! -z $online_ivector_dir ]; then
  ivector_period=$(cat $online_ivector_dir/ivector_period)
  ivector_dim=$(feat-to-dim scp:$online_ivector_dir/ivector_online.scp -) || exit 1;
  echo $ivector_dim >$dir/info/ivector_dim

  ivector_opt="--ivectors='ark,s,cs:utils/filter_scp.pl $sdata/JOB/utt2spk $online_ivector_dir/ivector_online.scp | subsample-feats --n=-$ivector_period scp:- ark:- |'"
  valid_ivector_opt="--ivectors='ark,s,cs:utils/filter_scp.pl $dir/valid_uttlist $online_ivector_dir/ivector_online.scp | subsample-feats --n=-$ivector_period scp:- ark:- |'"
  train_subset_ivector_opt="--ivectors='ark,s,cs:utils/filter_scp.pl $dir/train_subset_uttlist $online_ivector_dir/ivector_online.scp | subsample-feats --n=-$ivector_period scp:- ark:- |'"
  priors_ivector_opt="--ivectors='ark,s,cs:utils/filter_scp.pl $dir/priors_uttlist $online_ivector_dir/ivector_online.scp | subsample-feats --n=-$ivector_period scp:- ark:- |'"
fi

if [ $stage -le 2 ]; then
  echo "$0: working out number of frames of training data"
  num_frames=$(steps/nnet2/get_num_frames.sh $data)
  echo $num_frames > $dir/info/num_frames
  echo "$0: working out feature dim"
  feats_one="$(echo $feats | sed s/JOB/1/g)"
  feat_dim=$(feat-to-dim "$feats_one" -) || exit 1;
  echo $feat_dim > $dir/info/feat_dim
else
  num_frames=$(cat $dir/info/num_frames) || exit 1;
  feat_dim=$(cat $dir/info/feat_dim) || exit 1;
fi

# Working out total number of archives. Add one on the assumption the
# num-frames won't divide exactly, and we want to round up.
num_archives=$[$num_frames/$frames_per_iter+1]

# We may have to first create a smaller number of larger archives, with number
# $num_archives_intermediate, if $num_archives is more than the maximum number
# of open filehandles that the system allows per process (ulimit -n).
max_open_filehandles=$(ulimit -n) || exit 1
num_archives_intermediate=$num_archives
archives_multiple=1
while [ $[$num_archives_intermediate+4] -gt $max_open_filehandles ]; do
  archives_multiple=$[$archives_multiple+1]
  num_archives_intermediate=$[$num_archives/$archives_multiple] || exit 1;
done
# now make sure num_archives is an exact multiple of archives_multiple.
num_archives=$[$archives_multiple*$num_archives_intermediate] || exit 1;

echo $num_archives >$dir/info/num_archives
echo $frames_per_eg >$dir/info/frames_per_eg
# Work out the number of egs per archive
egs_per_archive=$[$num_frames/($frames_per_eg*$num_archives)] || exit 1;
! [ $egs_per_archive -le $frames_per_iter ] && \
  echo "$0: script error: egs_per_archive=$egs_per_archive not <= frames_per_iter=$frames_per_iter" \
  && exit 1;

echo $egs_per_archive > $dir/info/egs_per_archive

echo "$0: creating $num_archives archives, each with $egs_per_archive egs, with"
echo "$0:   $frames_per_eg labels per example, and (left,right) context = ($left_context,$right_context)"


if [ -e $dir/storage ]; then
  # Make soft links to storage directories, if distributing this way..  See
  # utils/create_split_dir.pl.
  echo "$0: creating data links"
  utils/create_data_link.pl $(for x in $(seq $num_archives); do echo $dir/degs.$x.ark; done)
  for x in $(seq $num_archives_intermediate); do
    utils/create_data_link.pl $(for y in $(seq $nj); do echo $dir/degs_orig.$y.$x.ark; done)
  done
fi

if [ $stage -le 3 ]; then
  echo "$0: copying training lattices"

  $cmd --max-jobs-run 6 JOB=1:$nj $dir/log/lattice_copy.JOB.log \
    lattice-copy --write-compact=false --include="cat $dir/valid_uttlist $dir/train_subset_uttlist |" --ignore-missing \
    "ark:gunzip -c $denlatdir/lat.JOB.gz|" ark,scp:$dir/lat_special.JOB.ark,$dir/lat_special.JOB.scp || exit 1;

  for id in $(seq $nj); do cat $dir/lat_special.$id.scp; done > $dir/lat_special.scp
fi

splitter_opts="--supervision-splitter.determinize=$determinize --supervision-splitter.minimize=$minimize --supervision-splitter.remove_output_symbols=$remove_output_symbols --supervision-splitter.remove_epsilons=$remove_epsilons --supervision-splitter.collapse-transition-ids=$collapse_transition_ids --supervision-splitter.acoustic-scale=$acwt"

[ -z $valid_left_context ] &&  valid_left_context=$left_context;
[ -z $valid_right_context ] &&  valid_right_context=$right_context;

[ -z $priors_left_context ] &&  priors_left_context=$left_context;
[ -z $priors_right_context ] &&  priors_right_context=$right_context;

left_context=$[left_context+frame_subsampling_factor/2]
right_context=$[right_context+frame_subsampling_factor/2]

egs_opts="--left-context=$left_context --right-context=$right_context --num-frames=$frames_per_eg --num-frames-overlap=$frames_overlap_per_eg --frame-subsampling-factor=$frame_subsampling_factor --compress=$compress $splitter_opts"

valid_left_context=$[valid_left_context+frame_subsampling_factor/2]
valid_right_context=$[valid_right_context+frame_subsampling_factor/2]

# don't do the overlap thing for the validation data.
valid_egs_opts="--left-context=$valid_left_context --right-context=$valid_right_context --num-frames=$frames_per_eg --frame-subsampling-factor=$frame_subsampling_factor --compress=$compress $splitter_opts"

priors_left_context=$[priors_left_context+frame_subsampling_factor/2]
priors_right_context=$[priors_right_context+frame_subsampling_factor/2]

# don't do the overlap thing for the priors computation data.
priors_egs_opts="--left-context=$priors_left_context --right-context=$priors_right_context --num-frames=1 --compress=$compress"

supervision_all_opts="--frame-subsampling-factor=$frame_subsampling_factor"

echo $left_context > $dir/info/left_context
echo $right_context > $dir/info/right_context

echo $priors_left_context > $dir/info/priors_left_context
echo $priors_right_context > $dir/info/priors_right_context

echo $frame_subsampling_factor > $dir/info/frame_subsampling_factor


if [ "$frame_subsampling_factor" != 1 ]; then
  if $adjust_priors; then
    echo "$0: setting --adjust-priors false since adjusting priors is not supported (and does not make sense) for chain models"
    adjust_priors=false
  fi
fi

(
  if $adjust_priors && [ $stage -le 10 ]; then
    if [ ! -f $dir/ali.scp ]; then
      nj_ali=$(cat $alidir/num_jobs)
      alis=$(for n in $(seq $nj_ali); do echo -n "$alidir/ali.$n.gz "; done)
      $cmd $dir/log/copy_alignments.log \
        copy-int-vector "ark:gunzip -c $alis|" \
        ark,scp:$dir/ali.ark,$dir/ali.scp || exit 1;
    fi

    priors_egs_list=
    for y in `seq $num_archives_priors`; do
      utils/create_data_link.pl $dir/priors_egs.$y.ark
      priors_egs_list="$priors_egs_list ark:$dir/priors_egs.$y.ark"
    done

    echo "$0: dumping egs for prior adjustment in the background."

    num_pdfs=`am-info $alidir/final.mdl | grep pdfs | awk '{print $NF}' 2>/dev/null` || exit 1

    $cmd $dir/log/create_priors_subset.log \
      nnet3-get-egs --num-pdfs=$num_pdfs $priors_ivector_opt $priors_egs_opts "$priors_feats" \
      "$prior_ali_rspecifier ali-to-post ark:- ark:- |" \
      ark:- \| nnet3-copy-egs ark:- $priors_egs_list || \
      { touch $dir/.error; echo "Error in creating priors subset. See $dir/log/create_priors_subset.log"; exit 1; }

    sleep 3;

    echo $num_archives_priors >$dir/info/num_archives_priors
  else
    echo 0 > $dir/info/num_archives_priors
  fi
) &

if [ $stage -le 4 ]; then
  echo "$0: Getting validation and training subset examples."
  rm $dir/.error 2>/dev/null
  echo "$0: ... extracting validation and training-subset alignments."

  #utils/filter_scp.pl <(cat $dir/valid_uttlist $dir/train_subset_uttlist) \
  #  <$dir/lat.scp >$dir/lat_special.scp

  utils/filter_scp.pl <(cat $dir/valid_uttlist $dir/train_subset_uttlist) \
    <$dir/ali.scp >$dir/ali_special.scp

  $cmd $dir/log/create_valid_subset.log \
    discriminative-get-supervision $supervision_all_opts \
    scp:$dir/ali_special.scp scp:$dir/lat_special.scp ark:- \| \
    nnet3-discriminative-get-egs $valid_ivector_opt $valid_egs_opts \
    $dir/final.mdl "$valid_feats" ark,s,cs:- "ark:$dir/valid_diagnostic.degs" || touch $dir/.error &

  $cmd $dir/log/create_train_subset.log \
    discriminative-get-supervision $supervision_all_opts \
    scp:$dir/ali_special.scp scp:$dir/lat_special.scp ark:- \| \
    nnet3-discriminative-get-egs $train_subset_ivector_opt $egs_opts \
    $dir/final.mdl "$train_subset_feats" ark,s,cs:- "ark:$dir/train_diagnostic.degs" || touch $dir/.error &
  wait;
  [ -f $dir/.error ] && echo "Error detected while creating train/valid egs" && exit 1
  echo "... Getting subsets of validation examples for diagnostics and combination."

  for f in $dir/{train_diagnostic,valid_diagnostic}.degs; do
    [ ! -s $f ] && echo "No examples in file $f" && exit 1;
  done
fi

if [ $stage -le 5 ]; then
  # create degs_orig.*.*.ark; the first index goes to $nj,
  # the second to $num_archives_intermediate.

  degs_list=
  for n in $(seq $num_archives_intermediate); do
    degs_list="$degs_list ark:$dir/degs_orig.JOB.$n.ark"
  done
  echo "$0: Generating training examples on disk"

  # The examples will go round-robin to degs_list.
  # To make it efficient we need to use a large 'nj', like 40, and in that case
  # there can be too many small files to deal with, because the total number of
  # files is the product of 'nj' by 'num_archives_intermediate', which might be
  # quite large.
  $cmd --max-jobs-run $max_jobs_run JOB=1:$nj $dir/log/get_egs.JOB.log \
    discriminative-get-supervision $supervision_all_opts \
    "scp:utils/filter_scp.pl $sdata/JOB/utt2spk $dir/ali.scp |" \
    "ark,s,cs:gunzip -c $denlatdir/lat.JOB.gz |" ark:- \| \
    nnet3-discriminative-get-egs $ivector_opt $egs_opts \
    $dir/final.mdl "$feats" ark,s,cs:- ark:- \| \
    nnet3-discriminative-copy-egs --random=true --srand=JOB ark:- $degs_list || exit 1;
fi

if [ $stage -le 6 ]; then
  echo "$0: recombining and shuffling order of archives on disk"
  # combine all the "degs_orig.*.JOB.scp" (over the $nj splits of the data) and
  # shuffle the order, writing to the degs.JOB.ark

  # the input is a concatenation over the input jobs.
  degs_list=
  for n in $(seq $nj); do
    degs_list="$degs_list $dir/degs_orig.$n.JOB.ark"
  done

  if [ $archives_multiple == 1 ]; then # normal case.
    $cmd --max-jobs-run $max_shuffle_jobs_run --mem 8G JOB=1:$num_archives_intermediate $dir/log/shuffle.JOB.log \
      nnet3-discriminative-shuffle-egs --srand=JOB "ark:cat $degs_list|" ark:$dir/degs.JOB.ark  || exit 1;
  else
    # we need to shuffle the 'intermediate archives' and then split into the
    # final archives.  we create soft links to manage this splitting, because
    # otherwise managing the output names is quite difficult (and we don't want
    # to submit separate queue jobs for each intermediate archive, because then
    # the --max-jobs-run option is hard to enforce).
    output_archives=$(for y in $(seq $archives_multiple); do echo -n "ark:$dir/degs.JOB.$y.ark "; done)
    for x in $(seq $num_archives_intermediate); do
      for y in $(seq $archives_multiple); do
        archive_index=$[($x-1)*$archives_multiple+$y]
        # degs.intermediate_archive.{1,2,...}.ark will point to degs.archive.ark
        ln -sf degs.$archive_index.ark $dir/degs.$x.$y.ark || exit 1
      done
    done
    $cmd --max-jobs-run $max_shuffle_jobs_run --mem 8G JOB=1:$num_archives_intermediate $dir/log/shuffle.JOB.log \
      nnet3-discriminative-shuffle-egs --srand=JOB "ark:cat $degs_list|" ark:- \| \
      nnet3-discriminative-copy-egs ark:- $output_archives || exit 1;
  fi
fi

if [ $stage -le 7 ]; then
  echo "$0: removing temporary archives"
  for x in $(seq $nj); do
    for y in $(seq $num_archives_intermediate); do
      file=$dir/degs_orig.$x.$y.ark
      [ -L $file ] && rm $(readlink -f $file)
      rm $file
    done
  done
  if [ $archives_multiple -gt 1 ]; then
    # there are some extra soft links that we should delete.
    for f in $dir/degs.*.*.ark; do rm $f; done
  fi
  echo "$0: removing temporary lattices"
  rm $dir/lat.*
  echo "$0: removing temporary alignments and transforms"
  # Ignore errors below because trans.* might not exist.
  rm $dir/{ali,trans}.{ark,scp} 2>/dev/null
fi

wait

echo "$0: Finished preparing training examples"
