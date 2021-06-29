#!/usr/bin/env bash

# Copyright 2012-2015 Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.
#
# This script, which will generally be called from other neural-net training
# scripts, extracts the training examples used to train the 'chain' system
# (and also the validation examples used for diagnostics), and puts them in
# separate archives.
#
# This script dumps egs with many frames of labels, controlled by the
# frames_per_eg config variable (default: 25), plus left and right context.
# Because CTC training involves alignment of data, we can't meaningfully train
# frame by frame.   The supervision approach involves the time alignment, though--
# it is just applied in a loose way, where each symbol can appear in the
# frame-range that it was in in the alignment, extended by a certain margin.
#


# Begin configuration section.
cmd=run.pl
frames_per_eg=25   # number of feature frames example (not counting added context).
                   # more->less disk space and less time preparing egs, but more
                   # I/O during training.
frames_overlap_per_eg=0  # number of supervised frames of overlap that we aim for per eg.
                  # can be useful to avoid wasted data if you're using --left-deriv-truncate
                  # and --right-deriv-truncate.
frame_subsampling_factor=3 # frames-per-second of features we train on divided
                           # by frames-per-second at output of chain model
alignment_subsampling_factor=3 # frames-per-second of input alignments divided
                               # by frames-per-second at output of chain model
left_context=4    # amount of left-context per eg (i.e. extra frames of input features
                  # not present in the output supervision).
right_context=4   # amount of right-context per eg.
constrained=true  # 'constrained=true' is the traditional setup; 'constrained=false'
                  # gives you the 'unconstrained' egs creation in which the time
                  # boundaries are not enforced inside chunks.

left_context_initial=-1    # if >=0, left-context for first chunk of an utterance
right_context_final=-1     # if >=0, right-context for last chunk of an utterance
compress=true   # set this to false to disable compression (e.g. if you want to see whether
                # results are affected).

num_utts_subset=300     # number of utterances in validation and training
                        # subsets used for shrinkage and diagnostics.
num_valid_egs_combine=0  # #validation examples for combination weights at the very end.
num_train_egs_combine=1000 # number of train examples for the above.
num_egs_diagnostic=400 # number of frames for "compute_prob" jobs
frames_per_iter=400000 # each iteration of training, see this many frames per
                       # job, measured at the sampling rate of the features
                       # used.  This is just a guideline; it will pick a number
                       # that divides the number of samples in the entire data.

right_tolerance=  # chain right tolerance == max label delay.
left_tolerance=

stage=0
max_jobs_run=15         # This should be set to the maximum number of nnet3-chain-get-egs jobs you are
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
lattice_lm_scale=     # If supplied, the graph/lm weight of the lattices will be
                      # used (with this scale) in generating supervisions
                      # This is 0 by default for conventional supervised training,
                      # but may be close to 1 for the unsupervised part of the data
                      # in semi-supervised training. The optimum is usually
                      # 0.5 for unsupervised data.
lattice_prune_beam=         # If supplied, the lattices will be pruned to this beam,
                            # before being used to get supervisions.
acwt=0.1   # For pruning
deriv_weights_scp=
generate_egs_scp=false

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 4 ]; then
  echo "Usage: $0 [opts] <data> <chain-dir> <lattice-dir> <egs-dir>"
  echo " e.g.: $0 data/train exp/tri4_nnet exp/tri3_lats exp/tri4_nnet/egs"
  echo ""
  echo "From <chain-dir>, 0.trans_mdl (the transition-model), tree (the tree)"
  echo "and normalization.fst (the normalization FST, derived from the denominator FST)"
  echo "are read."
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --max-jobs-run <max-jobs-run>                    # The maximum number of jobs you want to run in"
  echo "                                                   # parallel (increase this only if you have good disk and"
  echo "                                                   # network speed).  default=6"
  echo "  --cmd (utils/run.pl;utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --frames-per-iter <#samples;400000>              # Number of frames of data to process per iteration, per"
  echo "                                                   # process."
  echo "  --frame-subsampling-factor <factor;3>            # factor by which num-frames at nnet output is reduced "
  echo "  --frames-per-eg <frames;25>                      # number of supervised frames per eg on disk"
  echo "  --frames-overlap-per-eg <frames;25>              # number of supervised frames of overlap between egs"
  echo "  --left-context <int;4>                           # Number of frames on left side to append for feature input"
  echo "  --right-context <int;4>                          # Number of frames on right side to append for feature input"
  echo "  --left-context-initial <int;-1>                  # If >= 0, left-context for first chunk of an utterance"
  echo "  --right-context-final <int;-1>                   # If >= 0, right-context for last chunk of an utterance"
  echo "  --num-egs-diagnostic <#frames;4000>              # Number of egs used in computing (train,valid) diagnostics"
  echo "  --num-valid-egs-combine <#frames;10000>          # Number of egs used in getting combination weights at the"
  echo "                                                   # very end."
  echo "  --lattice-lm-scale <float>                       # If supplied, the graph/lm weight of the lattices will be "
  echo "                                                   # used (with this scale) in generating supervisions"
  echo "  --lattice-prune-beam <float>                     # If supplied, the lattices will be pruned to this beam, "
  echo "                                                   # before being used to get supervisions."
  echo "  --acwt <float;0.1>                               # Acoustic scale -- affects pruning"
  echo "  --deriv-weights-scp <str>                        # If supplied, adds per-frame weights to the supervision."
  echo "  --generate-egs-scp <bool;false>                  # Generates scp files -- Required if the egs will be "
  echo "                                                   # used for multilingual/multitask training."
  echo "  --stage <stage|0>                                # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."

  exit 1;
fi

data=$1
chaindir=$2
latdir=$3
dir=$4

# Check some files.
[ ! -z "$online_ivector_dir" ] && \
  extra_files="$online_ivector_dir/ivector_online.scp $online_ivector_dir/ivector_period"

for f in $data/feats.scp $latdir/lat.1.gz $latdir/final.mdl \
         $chaindir/{0.trans_mdl,tree,normalization.fst} $extra_files; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

nj=$(cat $latdir/num_jobs) || exit 1
if [ -f $latdir/per_utt ]; then
  sdata=$data/split${nj}utt
  utils/split_data.sh --per-utt $data $nj
else
  sdata=$data/split$nj
  utils/split_data.sh $data $nj
fi

mkdir -p $dir/log $dir/info

# Get list of validation utterances.
frame_shift=$(utils/data/get_frame_shift.sh $data) || exit 1

if [ -f $data/utt2uniq ]; then
  # Must hold out all augmented versions of the same utterance.
  echo "$0: File $data/utt2uniq exists, so ensuring the hold-out set" \
       "includes all perturbed versions of the same source utterance."
  utils/utt2spk_to_spk2utt.pl $data/utt2uniq 2>/dev/null | \
      utils/shuffle_list.pl 2>/dev/null | \
    awk -v max_utt=$num_utts_subset '{
        for (n=2;n<=NF;n++) print $n;
        printed += NF-1;
        if (printed >= max_utt) exit(0); }' |
    sort > $dir/valid_uttlist
else
  awk '{print $1}' $data/utt2spk | \
    utils/shuffle_list.pl 2>/dev/null | \
    head -$num_utts_subset > $dir/valid_uttlist
fi
len_valid_uttlist=$(wc -l < $dir/valid_uttlist)

awk '{print $1}' $data/utt2spk | \
   utils/filter_scp.pl --exclude $dir/valid_uttlist | \
   utils/shuffle_list.pl 2>/dev/null | \
   head -$num_utts_subset > $dir/train_subset_uttlist
len_trainsub_uttlist=$(wc -l <$dir/train_subset_uttlist)

if [[ $len_valid_uttlist -lt $num_utts_subset ||
      $len_trainsub_uttlist -lt $num_utts_subset ]]; then
  echo "$0: Number of utterances is very small. Please check your data." && exit 1;
fi

echo "$0: Holding out $len_valid_uttlist utterances in validation set and" \
     "$len_trainsub_uttlist in training diagnostic set, out of total" \
     "$(wc -l < $data/utt2spk)."


echo "$0: creating egs.  To ensure they are not deleted later you can do:  touch $dir/.nodelete"

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
  valid_spk2utt="ark:utils/filter_scp.pl $dir/valid_uttlist $data/utt2spk | utils/utt2spk_to_spk2utt.pl |"
  valid_feats="ark,s,cs:utils/filter_scp.pl $dir/valid_uttlist $data/feats.scp | apply-cmvn-online $cmvn_opts --spk2utt=\"$valid_spk2utt\" $dir/global_cmvn.stats scp:- ark:- |"
  train_subset_spk2utt="ark:utils/filter_scp.pl $dir/train_subset_uttlist $data/utt2spk | utils/utt2spk_to_spk2utt.pl |"
  train_subset_feats="ark,s,cs:utils/filter_scp.pl $dir/train_subset_uttlist $data/feats.scp | apply-cmvn-online $cmvn_opts --spk2utt=\"$train_subset_spk2utt\" $dir/global_cmvn.stats scp:- ark:- |"
fi
echo $cmvn_opts >$dir/cmvn_opts # caution: the top-level nnet training script should copy this to its own dir now.


tree-info $chaindir/tree | grep num-pdfs | awk '{print $2}' > $dir/info/num_pdfs || exit 1

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
# This sometimes gives a misleading answer as GridEngine sometimes changes the
# limit, so we limit it to 512.
max_open_filehandles=$(ulimit -n) || exit 1
[ $max_open_filehandles -gt 512 ] && max_open_filehandles=512
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

[ ! -z "$deriv_weights_scp" ] && egs_opts="$egs_opts --deriv-weights-rspecifier=scp:$deriv_weights_scp"

chain_supervision_all_opts="--lattice-input=true --frame-subsampling-factor=$alignment_subsampling_factor"
[ ! -z $right_tolerance ] && \
  chain_supervision_all_opts="$chain_supervision_all_opts --right-tolerance=$right_tolerance"

[ ! -z $left_tolerance ] && \
  chain_supervision_all_opts="$chain_supervision_all_opts --left-tolerance=$left_tolerance"

if ! $constrained; then
  chain_supervision_all_opts="$chain_supervision_all_opts --convert-to-pdfs=false"
  trans_mdl_opt=--transition-model=$chaindir/0.trans_mdl
else
  trans_mdl_opt=
fi


lats_rspecifier="ark:gunzip -c $latdir/lat.JOB.gz |"
if [ ! -z $lattice_prune_beam ]; then
  if [ "$lattice_prune_beam" == "0" ] || [ "$lattice_prune_beam" == "0.0" ]; then
    lats_rspecifier="$lats_rspecifier lattice-1best --acoustic-scale=$acwt ark:- ark:- |"
  else
    lats_rspecifier="$lats_rspecifier lattice-prune --acoustic-scale=$acwt --beam=$lattice_prune_beam ark:- ark:- |"
  fi
fi

normalization_fst_scale=1.0

if [ ! -z "$lattice_lm_scale" ]; then
  chain_supervision_all_opts="$chain_supervision_all_opts --lm-scale=$lattice_lm_scale"

  normalization_fst_scale=$(perl -e "
  if ($lattice_lm_scale >= 1.0 || $lattice_lm_scale < 0) {
    print STDERR \"Invalid --lattice-lm-scale $lattice_lm_scale\";
    exit(1);
  }
  print (1.0 - $lattice_lm_scale);") || exit 1
fi

echo $left_context > $dir/info/left_context
echo $right_context > $dir/info/right_context
echo $left_context_initial > $dir/info/left_context_initial
echo $right_context_final > $dir/info/right_context_final

if [ $stage -le 2 ]; then
  echo "$0: Getting validation and training subset examples in background."
  rm $dir/.error 2>/dev/null

  (
    $cmd --max-jobs-run 6 JOB=1:$nj $dir/log/lattice_copy.JOB.log \
      lattice-copy --include="cat $dir/valid_uttlist $dir/train_subset_uttlist |" --ignore-missing \
      "$lats_rspecifier" \
      ark,scp:$dir/lat_special.JOB.ark,$dir/lat_special.JOB.scp || exit 1

    for id in $(seq $nj); do cat $dir/lat_special.$id.scp; done > $dir/lat_special.scp

    $cmd $dir/log/create_valid_subset.log \
      utils/filter_scp.pl $dir/valid_uttlist $dir/lat_special.scp \| \
      lattice-align-phones --replace-output-symbols=true $latdir/final.mdl scp:- ark:- \| \
      chain-get-supervision $chain_supervision_all_opts $chaindir/tree $chaindir/0.trans_mdl \
        ark:- ark:- \| \
      nnet3-chain-get-egs $ivector_opts --srand=$srand \
         $egs_opts --normalization-fst-scale=$normalization_fst_scale \
         $trans_mdl_opt $chaindir/normalization.fst \
        "$valid_feats" ark,s,cs:- "ark:$dir/valid_all.cegs" || exit 1
    $cmd $dir/log/create_train_subset.log \
      utils/filter_scp.pl $dir/train_subset_uttlist $dir/lat_special.scp \| \
      lattice-align-phones --replace-output-symbols=true $latdir/final.mdl scp:- ark:- \| \
      chain-get-supervision $chain_supervision_all_opts \
        $chaindir/tree $chaindir/0.trans_mdl ark:- ark:- \| \
      nnet3-chain-get-egs $ivector_opts --srand=$srand \
        $egs_opts --normalization-fst-scale=$normalization_fst_scale \
        $trans_mdl_opt $chaindir/normalization.fst \
        "$train_subset_feats" ark,s,cs:- "ark:$dir/train_subset_all.cegs" || exit 1
    sleep 5  # wait for file system to sync.
    echo "$0: Getting subsets of validation examples for diagnostics and combination."
    if $generate_egs_scp; then
      valid_diagnostic_output="ark,scp:$dir/valid_diagnostic.cegs,$dir/valid_diagnostic.scp"
      train_diagnostic_output="ark,scp:$dir/train_diagnostic.cegs,$dir/train_diagnostic.scp"
    else
      valid_diagnostic_output="ark:$dir/valid_diagnostic.cegs"
      train_diagnostic_output="ark:$dir/train_diagnostic.cegs"
    fi
    $cmd $dir/log/create_valid_subset_combine.log \
      nnet3-chain-subset-egs --n=$num_valid_egs_combine ark:$dir/valid_all.cegs \
      ark:$dir/valid_combine.cegs || exit 1
    $cmd $dir/log/create_valid_subset_diagnostic.log \
      nnet3-chain-subset-egs --n=$num_egs_diagnostic ark:$dir/valid_all.cegs \
      $valid_diagnostic_output || exit 1

    $cmd $dir/log/create_train_subset_combine.log \
      nnet3-chain-subset-egs --n=$num_train_egs_combine ark:$dir/train_subset_all.cegs \
      ark:$dir/train_combine.cegs || exit 1
    $cmd $dir/log/create_train_subset_diagnostic.log \
      nnet3-chain-subset-egs --n=$num_egs_diagnostic ark:$dir/train_subset_all.cegs \
      $train_diagnostic_output || exit 1
    sleep 5  # wait for file system to sync.
    if $generate_egs_scp; then
      cat $dir/valid_combine.cegs $dir/train_combine.cegs | \
        nnet3-chain-copy-egs ark:- ark,scp:$dir/combine.cegs,$dir/combine.scp
    else
      cat $dir/valid_combine.cegs $dir/train_combine.cegs > $dir/combine.cegs
    fi

    for f in $dir/{combine,train_diagnostic,valid_diagnostic}.cegs; do
      [ ! -s $f ] && echo "$0: No examples in file $f" && exit 1;
    done
    rm $dir/valid_all.cegs $dir/train_subset_all.cegs $dir/{train,valid}_combine.cegs
  ) || touch $dir/.error &
fi

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

  $cmd --max-jobs-run $max_jobs_run JOB=1:$nj $dir/log/get_egs.JOB.log \
    lattice-align-phones --replace-output-symbols=true $latdir/final.mdl \
      "$lats_rspecifier" ark:- \| \
    chain-get-supervision $chain_supervision_all_opts \
      $chaindir/tree $chaindir/0.trans_mdl ark:- ark:- \| \
    nnet3-chain-get-egs $ivector_opts --srand=\$[JOB+$srand] $egs_opts \
      --num-frames-overlap=$frames_overlap_per_eg $trans_mdl_opt \
     "$feats" ark,s,cs:- ark:- \| \
    nnet3-chain-copy-egs --random=true --srand=\$[JOB+$srand] ark:- $egs_list || exit 1;
fi

if [ -f $dir/.error ]; then
  echo "$0: Error detected while creating train/valid egs" && exit 1
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
    if $generate_egs_scp; then
      output_archive="ark,scp:$dir/cegs.JOB.ark,$dir/cegs.JOB.scp"
    else
      output_archive="ark:$dir/cegs.JOB.ark"
    fi
    $cmd --max-jobs-run $max_shuffle_jobs_run --mem 8G \
      JOB=1:$num_archives_intermediate $dir/log/shuffle.JOB.log \
      nnet3-chain-normalize-egs --normalization-fst-scale=$normalization_fst_scale \
        $chaindir/normalization.fst "ark:cat $egs_list|" ark:- \| \
      nnet3-chain-shuffle-egs --srand=\$[JOB+$srand] ark:- $output_archive || exit 1;

    if $generate_egs_scp; then
      #concatenate cegs.JOB.scp in single cegs.scp
      for j in $(seq $num_archives_intermediate); do
        cat $dir/cegs.$j.scp || exit 1;
      done > $dir/cegs.scp || exit 1;
      for f in $dir/cegs.*.scp; do rm $f; done
    fi
  else
    # we need to shuffle the 'intermediate archives' and then split into the
    # final archives.  we create soft links to manage this splitting, because
    # otherwise managing the output names is quite difficult (and we don't want
    # to submit separate queue jobs for each intermediate archive, because then
    # the --max-jobs-run option is hard to enforce).
    if $generate_egs_scp; then
      output_archives="$(for y in $(seq $archives_multiple); do echo ark,scp:$dir/cegs.JOB.$y.ark,$dir/cegs.JOB.$y.scp; done)"
    else
      output_archives="$(for y in $(seq $archives_multiple); do echo ark:$dir/cegs.JOB.$y.ark; done)"
    fi
    for x in $(seq $num_archives_intermediate); do
      for y in $(seq $archives_multiple); do
        archive_index=$[($x-1)*$archives_multiple+$y]
        # egs.intermediate_archive.{1,2,...}.ark will point to egs.archive.ark
        ln -sf cegs.$archive_index.ark $dir/cegs.$x.$y.ark || exit 1
      done
    done
    $cmd --max-jobs-run $max_shuffle_jobs_run --mem 8G \
      JOB=1:$num_archives_intermediate $dir/log/shuffle.JOB.log \
      nnet3-chain-normalize-egs --normalization-fst-scale=$normalization_fst_scale \
        $chaindir/normalization.fst "ark:cat $egs_list|" ark:- \| \
      nnet3-chain-shuffle-egs --srand=\$[JOB+$srand] ark:- ark:- \| \
      nnet3-chain-copy-egs ark:- $output_archives || exit 1;

    if $generate_egs_scp; then
      #concatenate cegs.JOB.scp in single cegs.scp
      rm -rf $dir/cegs.scp
      for j in $(seq $num_archives_intermediate); do
        for y in $(seq $archives_multiple); do
          cat $dir/cegs.$j.$y.scp || exit 1;
        done
      done > $dir/cegs.scp || exit 1;
      for f in $dir/cegs.*.*.scp; do rm $f; done
    fi
  fi
fi

wait
if [ -f $dir/.error ]; then
  echo "$0: Error detected while creating train/valid egs" && exit 1
fi

if [ $stage -le 6 ]; then
  echo "$0: Removing temporary archives, alignments and lattices"
  (
    cd $dir
    for f in $(ls -l . | grep 'cegs_orig' | awk '{ X=NF-1; Y=NF-2; if ($X == "->")  print $Y, $NF; }'); do rm $f; done
    # the next statement removes them if we weren't using the soft links to a
    # 'storage' directory.
    rm cegs_orig.*.ark 2>/dev/null
  )
  if ! $generate_egs_scp && [ $archives_multiple -gt 1 ]; then
    # there are some extra soft links that we should delete.
    for f in $dir/cegs.*.*.ark; do rm $f; done
  fi
  rm $dir/ali.{ark,scp} 2>/dev/null
  rm $dir/lat_special.*.{ark,scp} 2>/dev/null
fi

echo "$0: Finished preparing training examples"
