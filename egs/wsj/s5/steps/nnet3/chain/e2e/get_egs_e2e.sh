#!/usr/bin/env bash

# Copyright 2012-2015 Johns Hopkins University (Author: Daniel Povey)
# Copyright   2017  Hossein Hadian
# Apache 2.0.
#


# This is simlilar to chain/get_egs.sh except it
# uses training FSTs (instead of lattices) to generate end2end egs.
# It calls nnet3-chain-e2e-get-egs binary


# Begin configuration section.
cmd=run.pl
normalize_egs=true
frame_subsampling_factor=3 # frames-per-second of features we train on divided
                           # by frames-per-second at output of chain model
left_context=4    # amount of left-context per eg (i.e. extra frames of input features
                  # not present in the output supervision).
right_context=4   # amount of right-context per eg.
left_context_initial=-1    # if >=0, left-context for first chunk of an utterance
right_context_final=-1     # if >=0, right-context for last chunk of an utterance
compress=true   # set this to false to disable compression (e.g. if you want to see whether
                # results are affected).

num_utts_subset=1400     # number of utterances in validation and training
                        # subsets used for shrinkage and diagnostics.
num_valid_egs_combine=0  # #validation examples for combination weights at the very end.
num_train_egs_combine=250 # number of train examples for the above.
num_egs_diagnostic=300 # number of frames for "compute_prob" jobs
frames_per_iter=400000 # each iteration of training, see this many frames per
                       # job, measured at the sampling rate of the features
                       # used.  This is just a guideline; it will pick a number
                       # that divides the number of samples in the entire data.

stage=0
nj=15         # This should be set to the maximum number of jobs you are
              # comfortable to run in parallel; you can increase it if your disk
              # speed is greater and you have more machines.
max_shuffle_jobs_run=50  # the shuffle jobs now include the nnet3-chain-normalize-egs command,
                         # which is fairly CPU intensive, so we can run quite a few at once
                         # without overloading the disks.
srand=0     # rand seed for nnet3-chain-get-egs, nnet3-chain-copy-egs and nnet3-chain-shuffle-egs
online_ivector_dir=  # can be used if we are including speaker information as iVectors.
cmvn_opts=  # can be used for specifying CMVN options, if feature type is not lda (if lda,
            # it doesn't make sense to use different options than were used as input to the
            # LDA transform).  This is used to turn off CMVN in the online-nnet experiments.
online_cmvn=false # Set to 'true' to replace 'apply-cmvn' by 'apply-cmvn-online' in the nnet3 input.
                  # The configuration is passed externally via '$cmvn_opts' given to train.py,
                  # typically as: --cmvn-opts="--config conf/online_cmvn.conf".
                  # The global_cmvn.stats are computed by this script from the features.
                  # Note: the online cmvn for ivector extractor it is controlled separately in
                  #       steps/online/nnet2/train_ivector_extractor.sh by --online-cmvn-iextractor


echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 4 ]; then
  echo "Usage: $0 [opts] <data> <chain-dir> <fsts-dir> <egs-dir>"
  echo " e.g.: $0 data/train exp/chain/e2e exp/chain/e2e/egs"
  echo ""
  echo "From <chain-dir>, 0.trans_mdl (the transition-model), tree (the tree)"
  echo "and normalization.fst (the normalization FST, derived from the denominator FST)"
  echo "are read."
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --nj <nj>                                        # The maximum number of jobs you want to run in"
  echo "                                                   # parallel (increase this only if you have good disk and"
  echo "                                                   # network speed).  default=6"
  echo "  --cmd (utils/run.pl;utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --frames-per-iter <#samples;400000>              # Number of frames of data to process per iteration, per"
  echo "                                                   # process."
  echo "  --feat-type <lda|raw>                            # (raw is the default).  The feature type you want"
  echo "                                                   # to use as input to the neural net."
  echo "  --frame-subsampling-factor <factor;3>            # factor by which num-frames at nnet output is reduced "
  echo "  --left-context <int;4>                           # Number of frames on left side to append for feature input"
  echo "  --right-context <int;4>                          # Number of frames on right side to append for feature input"
  echo "  --left-context-initial <int;-1>                  # If >= 0, left-context for first chunk of an utterance"
  echo "  --right-context-final <int;-1>                   # If >= 0, right-context for last chunk of an utterance"
  echo "  --num-egs-diagnostic <#frames;4000>              # Number of egs used in computing (train,valid) diagnostics"
  echo "  --num-valid-egs-combine <#frames;10000>          # Number of egss used in getting combination weights at the"
  echo "                                                   # very end."
  echo "  --stage <stage|0>                                # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."

  exit 1;
fi

data=$1
chaindir=$2
fstdir=$3
dir=$4

# Check some files.
[ ! -z "$online_ivector_dir" ] && \
  extra_files="$online_ivector_dir/ivector_online.scp $online_ivector_dir/ivector_period"

for f in $data/feats.scp $data/allowed_lengths.txt \
         $chaindir/{0.trans_mdl,tree,normalization.fst} $extra_files; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

sdata=$data/split$nj
utils/split_data.sh $data $nj

mkdir -p $dir/log $dir/info

# Get list of validation utterances.

frame_shift=$(utils/data/get_frame_shift.sh $data)
utils/data/get_utt2dur.sh $data

frames_per_eg=$(cat $data/allowed_lengths.txt | tr '\n' , | sed 's/,$//')

[ ! -f "$data/utt2len" ] && feat-to-len scp:$data/feats.scp ark,t:$data/utt2len

cat $data/utt2len | \
  awk '{print $1}' | \
  utils/shuffle_list.pl 2>/dev/null | head -$num_utts_subset > $dir/valid_uttlist


len_uttlist=`wc -l $dir/valid_uttlist | awk '{print $1}'`
if [ $len_uttlist -lt $num_utts_subset ]; then
  echo "Number of utterances which have length at least $frames_per_eg is really low. Please check your data." && exit 1;
fi

if [ -f $data/utt2uniq ]; then  # this matters if you use data augmentation.
  # because of this stage we can again have utts with lengths less than
  # frames_per_eg
  echo "File $data/utt2uniq exists, so augmenting valid_uttlist to"
  echo "include all perturbed versions of the same 'real' utterances."
  mv $dir/valid_uttlist $dir/valid_uttlist.tmp
  utils/utt2spk_to_spk2utt.pl $data/utt2uniq > $dir/uniq2utt
  cat $dir/valid_uttlist.tmp | utils/apply_map.pl $data/utt2uniq | \
    sort | uniq | utils/apply_map.pl $dir/uniq2utt | \
    awk '{for(n=1;n<=NF;n++) print $n;}' | sort  > $dir/valid_uttlist
  rm $dir/uniq2utt $dir/valid_uttlist.tmp
fi

# awk -v mf_len=222 '{if ($2 == mf_len) print $1}' | \
cat $data/utt2len | \
  awk '{print $1}' | \
   utils/filter_scp.pl --exclude $dir/valid_uttlist | \
   utils/shuffle_list.pl 2>/dev/null | head -$num_utts_subset > $dir/train_subset_uttlist
len_uttlist=`wc -l $dir/train_subset_uttlist | awk '{print $1}'`
if [ $len_uttlist -lt $num_utts_subset ]; then
  echo "Number of utterances which have length at least $frames_per_eg is really low. Please check your data." && exit 1;
fi

## Set up features.

# get the global_cmvn stats for online-cmvn,
if $online_cmvn; then
  # create global_cmvn.stats,
  #
  # caution: the top-level nnet training script should copy
  # 'global_cmvn.stats' and 'online_cmvn' to its own dir.
  if ! matrix-sum --binary=false scp:$data/cmvn.scp - >$dir/global_cmvn.stats 2>/dev/null; then
    echo "$0: Error summing cmvn stats"
    exit 1
  fi
  touch $dir/online_cmvn
else
  [ -f $dir/online_cmvn ] && rm $dir/online_cmvn
fi

# create the feature pipelines,
if ! $online_cmvn; then
  # the original front-end with 'apply-cmvn',
  echo "$0: feature type is raw, with 'apply-cmvn'"
  feats="ark,s,cs:utils/filter_scp.pl --exclude $dir/valid_uttlist $sdata/JOB/feats.scp | apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:- ark:- |"
  valid_feats="ark,s,cs:utils/filter_scp.pl $dir/valid_uttlist $data/feats.scp | apply-cmvn $cmvn_opts --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:- ark:- |"
  train_subset_feats="ark,s,cs:utils/filter_scp.pl $dir/train_subset_uttlist $data/feats.scp | apply-cmvn $cmvn_opts --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:- ark:- |"
else
  # the alternative front-end with 'apply-cmvn-online',
  # - the $cmvn_opts can be set to '--config=conf/online_cmvn.conf' which is the setup of ivector-extractor,
  echo "$0: feature type is raw, with 'apply-cmvn-online'"
  feats="ark,s,cs:utils/filter_scp.pl --exclude $dir/valid_uttlist $sdata/JOB/feats.scp | apply-cmvn-online $cmvn_opts --spk2utt=ark:$sdata/JOB/spk2utt $dir/global_cmvn.stats scp:- ark:- |"
  valid_feats="ark,s,cs:utils/filter_scp.pl $dir/valid_uttlist $data/feats.scp | apply-cmvn-online $cmvn_opts --spk2utt=ark:$data/spk2utt $dir/global_cmvn.stats scp:- ark:- |"
  train_subset_feats="ark,s,cs:utils/filter_scp.pl $dir/train_subset_uttlist $data/feats.scp | apply-cmvn-online $cmvn_opts --spk2utt=ark:$data/spk2utt $dir/global_cmvn.stats scp:- ark:- |"
fi
echo $cmvn_opts >$dir/cmvn_opts # caution: the top-level nnet training script should copy this to its own dir now.

if [ ! -z "$online_ivector_dir" ]; then
  ivector_dim=$(feat-to-dim scp:$online_ivector_dir/ivector_online.scp -) || exit 1;
  echo $ivector_dim > $dir/info/ivector_dim
  steps/nnet2/get_ivector_id.sh $online_ivector_dir > $dir/info/final.ie.id || exit 1
  ivector_period=$(cat $online_ivector_dir/ivector_period) || exit 1;
  ivector_opts="--online-ivectors=scp:$online_ivector_dir/ivector_online.scp --online-ivector-period=$ivector_period"
else
  ivector_opts=""
  echo 0 >$dir/info/ivector_dim
fi

if [ $stage -le 1 ]; then
  echo "$0: working out number of frames of training data"
  num_frames=$(steps/nnet2/get_num_frames.sh $data)
  echo $num_frames > $dir/info/num_frames
  echo "$0: working out feature dim"
  feats_one="$(echo $feats | sed s/JOB/1/g)"
  if ! feat_dim=$(feat-to-dim "$feats_one" - 2>/dev/null); then
    echo "Command failed (getting feature dim): feat-to-dim \"$feats_one\""
    exit 1
  fi
  echo $feat_dim > $dir/info/feat_dim
else
  num_frames=$(cat $dir/info/num_frames) || exit 1;
  feat_dim=$(cat $dir/info/feat_dim) || exit 1;
fi

# the + 1 is to round up, not down... we assume it doesn't divide exactly.
num_archives=$[$num_frames/$frames_per_iter+1]

# We may have to first create a smaller number of larger archives, with number
# $num_archives_intermediate, if $num_archives is more than the maximum number
# of open filehandles that the system allows per process (ulimit -n).
max_open_filehandles=500 #$(ulimit -n) || exit 1
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
if [ $left_context_initial -ge 0 ] || [ $right_context_final -ge 0 ]; then
  echo "$0:   ... and (left-context-initial,right-context-final) = ($left_context_initial,$right_context_final)"
fi


if [ -e $dir/storage ]; then
  # Make soft links to storage directories, if distributing this way..  See
  # utils/create_split_dir.pl.
  echo "$0: creating data links"
  utils/create_data_link.pl $(for x in $(seq $num_archives); do echo $dir/cegs.$x.ark; done)
  for x in $(seq $num_archives_intermediate); do
    utils/create_data_link.pl $(for y in $(seq $nj); do echo $dir/cegs_orig.$y.$x.ark; done)
  done
fi


egs_opts="--left-context=$left_context --right-context=$right_context --num-frames=$frames_per_eg --frame-subsampling-factor=$frame_subsampling_factor --compress=$compress"
[ $left_context_initial -ge 0 ] && egs_opts="$egs_opts --left-context-initial=$left_context_initial"
[ $right_context_final -ge 0 ] && egs_opts="$egs_opts --right-context-final=$right_context_final"


echo $left_context > $dir/info/left_context
echo $right_context > $dir/info/right_context
echo $left_context_initial > $dir/info/left_context_initial
echo $right_context_final > $dir/info/right_context_final

num_fst_jobs=$(cat $fstdir/num_jobs) || exit 1;
for id in $(seq $num_fst_jobs); do cat $fstdir/fst.$id.scp; done > $fstdir/fst.scp

if [ $stage -le 3 ]; then
  echo "$0: Getting validation and training subset examples."
  rm $dir/.error 2>/dev/null

  # do the filtering just once, as fst.scp may be long.
  utils/filter_scp.pl <(cat $dir/valid_uttlist $dir/train_subset_uttlist) \
    <$fstdir/fst.scp >$fstdir/fst_special.scp
  if $normalize_egs; then
    norm_opt=$chaindir/normalization.fst
  else
    norm_opt=
  fi
  $cmd $dir/log/create_valid_subset.log \
    utils/filter_scp.pl $dir/valid_uttlist $fstdir/fst_special.scp \| \
    fstcopy scp:- ark:- \| \
    nnet3-chain-e2e-get-egs $ivector_opts --srand=$srand \
      $egs_opts $norm_opt \
      "$valid_feats" ark,s,cs:- $chaindir/0.trans_mdl "ark:$dir/valid_all.cegs" || touch $dir/.error &
  $cmd $dir/log/create_train_subset.log \
    utils/filter_scp.pl $dir/train_subset_uttlist $fstdir/fst_special.scp \| \
    fstcopy scp:- ark:- \| \
    nnet3-chain-e2e-get-egs $ivector_opts --srand=$srand \
      $egs_opts $norm_opt \
      "$train_subset_feats" ark,s,cs:- $chaindir/0.trans_mdl "ark:$dir/train_subset_all.cegs" || touch $dir/.error &
  wait;
  [ -f $dir/.error ] && echo "Error detected while creating train/valid egs" && exit 1
  echo "... Getting subsets of validation examples for diagnostics and combination."
  $cmd $dir/log/create_valid_subset_combine.log \
    nnet3-chain-subset-egs --n=$num_valid_egs_combine ark:$dir/valid_all.cegs \
    ark:$dir/valid_combine.cegs || touch $dir/.error &
  $cmd $dir/log/create_valid_subset_diagnostic.log \
    nnet3-chain-subset-egs --n=$num_egs_diagnostic ark:$dir/valid_all.cegs \
    ark:$dir/valid_diagnostic.cegs || touch $dir/.error &

  $cmd $dir/log/create_train_subset_combine.log \
    nnet3-chain-subset-egs --n=$num_train_egs_combine ark:$dir/train_subset_all.cegs \
    ark:$dir/train_combine.cegs || touch $dir/.error &
  $cmd $dir/log/create_train_subset_diagnostic.log \
    nnet3-chain-subset-egs --n=$num_egs_diagnostic ark:$dir/train_subset_all.cegs \
    ark:$dir/train_diagnostic.cegs || touch $dir/.error &
  wait
  sleep 5  # wait for file system to sync.
  cat $dir/valid_combine.cegs $dir/train_combine.cegs > $dir/combine.cegs

  for f in $dir/{combine,train_diagnostic,valid_diagnostic}.cegs; do
    [ ! -s $f ] && echo "No examples in file $f" && exit 1;
  done

  #rm $dir/valid_all.cegs $dir/train_subset_all.cegs $dir/{train,valid}_combine.cegs
  #exit 0
fi

echo "num_archives_intermediate:" $num_archives_intermediate
echo "num_archives: $num_archives"
echo "archives_multiple: $archives_multiple"

if [ $stage -le 4 ]; then
  # create cegs_orig.*.*.ark; the first index goes to $nj,
  # the second to $num_archives_intermediate.

  egs_list=
  for n in $(seq $num_archives_intermediate); do
    egs_list="$egs_list ark:$dir/cegs_orig.JOB.$n.ark"
  done
  echo "$0: Generating training examples on disk"

  # The examples will go round-robin to egs_list.  Note: we omit the
  # 'normalization.fst' argument while creating temporary egs: the phase of egs
  # preparation that involves the normalization FST is quite CPU-intensive and
  # it's more convenient to do it later, in the 'shuffle' stage.  Otherwise to
  # make it efficient we need to use a large 'nj', like 40, and in that case
  # there can be too many small files to deal with, because the total number of
  # files is the product of 'nj' by 'num_archives_intermediate', which might be
  # quite large.
  $cmd JOB=1:$nj $dir/log/get_egs.JOB.log \
    utils/filter_scp.pl $sdata/JOB/utt2spk $fstdir/fst.scp \| \
    fstcopy scp:- ark:- \| \
    nnet3-chain-e2e-get-egs $ivector_opts --srand=\$[JOB+$srand] $egs_opts \
     "$feats" ark,s,cs:- $chaindir/0.trans_mdl ark:- \| \
    nnet3-chain-copy-egs --random=true --srand=\$[JOB+$srand] ark:- $egs_list || exit 1;
fi

if [ $stage -le 5 ]; then
  echo "$0: recombining and shuffling order of archives on disk"
  # combine all the "egs_orig.*.JOB.scp" (over the $nj splits of the data) and
  # shuffle the order, writing to the egs.JOB.ark

  # the input is a concatenation over the input jobs.
  egs_list=
  for n in $(seq $nj); do
    egs_list="$egs_list $dir/cegs_orig.$n.JOB.ark"
  done

  if [ $archives_multiple == 1 ]; then # normal case.
    if $normalize_egs; then
      $cmd --max-jobs-run $max_shuffle_jobs_run --mem 8G JOB=1:$num_archives_intermediate $dir/log/shuffle.JOB.log \
        nnet3-chain-normalize-egs $chaindir/normalization.fst "ark:cat $egs_list|" ark:- \| \
        nnet3-chain-shuffle-egs --srand=\$[JOB+$srand] ark:- ark:$dir/cegs.JOB.ark  || exit 1;
    else
      $cmd --max-jobs-run $max_shuffle_jobs_run --mem 8G JOB=1:$num_archives_intermediate $dir/log/shuffle.JOB.log \
        nnet3-chain-shuffle-egs --srand=\$[JOB+$srand] "ark:cat $egs_list|" ark:$dir/cegs.JOB.ark  || exit 1;
    fi
  else
    # we need to shuffle the 'intermediate archives' and then split into the
    # final archives.  we create soft links to manage this splitting, because
    # otherwise managing the output names is quite difficult (and we don't want
    # to submit separate queue jobs for each intermediate archive, because then
    # the --max-jobs-run option is hard to enforce).
    output_archives="$(for y in $(seq $archives_multiple); do echo ark:$dir/cegs.JOB.$y.ark; done)"
    for x in $(seq $num_archives_intermediate); do
      for y in $(seq $archives_multiple); do
        archive_index=$[($x-1)*$archives_multiple+$y]
        # egs.intermediate_archive.{1,2,...}.ark will point to egs.archive.ark
        ln -sf cegs.$archive_index.ark $dir/cegs.$x.$y.ark || exit 1
      done
    done
    $cmd --max-jobs-run $max_shuffle_jobs_run --mem 8G JOB=1:$num_archives_intermediate $dir/log/shuffle.JOB.log \
      nnet3-chain-normalize-egs $chaindir/normalization.fst "ark:cat $egs_list|" ark:- \| \
      nnet3-chain-shuffle-egs --srand=\$[JOB+$srand] ark:- ark:- \| \
      nnet3-chain-copy-egs ark:- $output_archives || exit 1;
  fi
fi

if [ $stage -le 6 ]; then
  echo "$0: removing temporary archives"
  (
    cd $dir
    for f in $(ls -l . | grep 'cegs_orig' | awk '{ X=NF-1; Y=NF-2; if ($X == "->")  print $Y, $NF; }'); do rm $f; done
    # the next statement removes them if we weren't using the soft links to a
    # 'storage' directory.
    rm cegs_orig.*.ark 2>/dev/null
  )
  if [ $archives_multiple -gt 1 ]; then
    # there are some extra soft links that we should delete.
    for f in $dir/cegs.*.*.ark; do rm $f; done
  fi
  echo "$0: removing temporary alignments"
  rm $dir/ali.{ark,scp} 2>/dev/null

fi

echo "$0: Finished preparing training examples"
