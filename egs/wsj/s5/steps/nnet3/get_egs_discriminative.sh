#!/bin/bash

# Copyright 2012-2016   Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.
# Copyright 2014-2015   Vimal Manohar

# Note: you may find it more convenient to use the newer script get_degs.sh, which
# combines decoding and example-creation in one step without writing lattices.

# This script dumps examples MPE or MMI or state-level minimum bayes risk (sMBR)
# training of neural nets.
# Criterion supported are mpe, smbr and mmi

# Begin configuration section.
cmd=run.pl
frames_per_eg=150 # number of frames of labels per example.  more->less disk space and
                  # less time preparing egs, but more I/O during training.
                  # Note: may in general be a comma-separated string of alternative
                  # durations; the first one (the principal num-frames) is preferred.
frames_overlap_per_eg=30 # number of supervised frames of overlap that we aim for per eg.
                  # can be useful to avoid wasted data if you're using --left-deriv-truncate
                  # and --right-deriv-truncate.
frame_subsampling_factor=1 # ratio between input and output frame-rate of nnet.
                           # this should be read from the nnet. For now, it is taken as an option
left_context=4    # amount of left-context per eg (i.e. extra frames of input features
                  # not present in the output supervision).
right_context=4   # amount of right-context per eg.
left_context_initial=-1    # if >=0, left-context for first chunk of an utterance
right_context_final=-1     # if >=0, right-context for last chunk of an utterance
adjust_priors=true
compress=true   # set this to false to disable compression (e.g. if you want to see whether
                # results are affected).
num_utts_subset=80     # number of utterances in validation and training
                        # subsets used for shrinkage and diagnostics.

frames_per_iter=400000 # each iteration of training, see this many frames
                       # per job.  This is just a guideline; it will pick a number
                       # that divides the number of samples in the entire data.

acwt=0.1

stage=0
max_jobs_run=15
max_shuffle_jobs_run=15

online_ivector_dir=
cmvn_opts=  # can be used for specifying CMVN options, if feature type is not lda (if lda,
            # it doesn't make sense to use different options than were used as input to the
            # LDA transform).  This is used to turn off CMVN in the online-nnet experiments.

num_priors_subset=1000  #  number of utterances used to calibrate the per-state
                        #  priors.  Note: these don't have to be held out from
                        #  the training data.
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
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs (probably would be good to add --max-jobs-run 5 or so if using"
  echo "                                                   # GridEngine (to avoid excessive NFS traffic)."
  echo "  --samples-per-iter <#samples|400000>             # Number of samples of data to process per iteration, per"
  echo "                                                   # process."
  echo "  --stage <stage|-8>                               # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."
  echo "  --online-ivector-dir <dir|"">                    # Directory for online-estimated iVectors, used in the"
  echo "                                                   # online-neural-net setup."
  echo "  --left-context <int;4>                           # Number of frames on left side to append for feature input"
  echo "  --right-context <int;4>                          # Number of frames on right side to append for feature input"
  echo "  --left-context-initial <int;-1>                  # If >= 0, left-context for first chunk of an utterance"
  echo "  --right-context-final <int;-1>                   # If >= 0, right-context for last chunk of an utterance"
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

if [ $stage -le 1 ]; then
  nj_ali=$(cat $alidir/num_jobs)
  alis=$(for n in $(seq $nj_ali); do echo -n "$alidir/ali.$n.gz "; done)
  $cmd $dir/log/copy_alignments.log \
    copy-int-vector "ark:gunzip -c $alis|" \
    ark,scp:$dir/ali.ark,$dir/ali.scp || exit 1;
fi

prior_ali_rspecifier="ark,s,cs:utils/filter_scp.pl $dir/priors_uttlist $dir/ali.scp | ali-to-pdf $alidir/final.mdl scp:- ark:- |"

silphonelist=`cat $lang/phones/silence.csl` || exit 1;
cp $alidir/tree $dir
cp $lang/phones/silence.csl $dir/info/
cp $src_model $dir/final.mdl || exit 1

# Get list of utterances for prior computation.
awk '{print $1}' $data/utt2spk | utils/filter_scp.pl --exclude $dir/valid_uttlist | \
  utils/shuffle_list.pl | head -$num_priors_subset \
  > $dir/priors_uttlist || exit 1;

feats="ark,s,cs:utils/filter_scp.pl --exclude $dir/valid_uttlist $sdata/JOB/feats.scp | apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:- ark:- |"
valid_feats="ark,s,cs:utils/filter_scp.pl $dir/valid_uttlist $data/feats.scp | apply-cmvn $cmvn_opts --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:- ark:- |"
train_subset_feats="ark,s,cs:utils/filter_scp.pl $dir/train_subset_uttlist $data/feats.scp | apply-cmvn $cmvn_opts --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:- ark:- |"
priors_feats="ark,s,cs:utils/filter_scp.pl $dir/priors_uttlist $data/feats.scp | apply-cmvn $cmvn_opts --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:- ark:- |"
echo $cmvn_opts > $dir/cmvn_opts

if [ ! -z $online_ivector_dir ]; then
  ivector_period=$(cat $online_ivector_dir/ivector_period)
  ivector_dim=$(feat-to-dim scp:$online_ivector_dir/ivector_online.scp -) || exit 1;
  echo $ivector_dim >$dir/info/ivector_dim
  steps/nnet2/get_ivector_id.sh $online_ivector_dir > $dir/info/final.ie.id || exit 1
  ivector_opts="--online-ivectors=scp:$online_ivector_dir/ivector_online.scp --online-ivector-period=$ivector_period"
else
  ivector_opts=""
fi

if [ $stage -le 2 ]; then
  echo "$0: working out number of frames of training data"
  num_frames=$(steps/nnet2/get_num_frames.sh $data)
  echo $num_frames > $dir/info/num_frames
  echo "$0: working out feature dim"
  feats_one="$(echo $feats | sed s:JOB:1:g)"
  if feat_dim=$(feat-to-dim "$feats_one" - 2>/dev/null); then
    echo $feat_dim > $dir/info/feat_dim
  else # run without stderr redirection to show the error.
    feat-to-dim "$feats_one" -; exit 1
  fi
fi

# Work out total number of archives. Add one on the assumption the
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

# the first field in frames_per_eg (which is a comma-separated list of numbers)
# is the 'principal' frames-per-eg, and for purposes of working out the number
# of archives we assume that this will be the average number of frames per eg.
frames_per_eg_principal=$(echo $frames_per_eg | cut -d, -f1)

# Work out the number of egs per archive
egs_per_archive=$[$num_frames/($frames_per_eg_principal*$num_archives)] || exit 1;
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



# If frame_subsampling_factor > 0, we will later be shifting the egs slightly to
# the left or right as part of training, so we see (e.g.) all shifts of the data
# modulo 3... we need to extend the l/r context slightly to account for this, to
# ensure we see the entire context that the model requires.
left_context=$[left_context+frame_subsampling_factor/2]
right_context=$[right_context+frame_subsampling_factor/2]
[ $left_context_initial -ge 0 ] && left_context_initial=$[left_context_initial+frame_subsampling_factor/2]
[ $right_context_final -ge 0 ] && right_context_final=$[right_context_final+frame_subsampling_factor/2]

egs_opts="--left-context=$left_context --right-context=$right_context --num-frames=$frames_per_eg --compress=$compress --frame-subsampling-factor=$frame_subsampling_factor --acoustic-scale=$acwt"
[ $left_context_initial -ge 0 ] && egs_opts="$egs_opts --left-context-initial=$left_context_initial"
[ $right_context_final -ge 0 ] && egs_opts="$egs_opts --right-context-final=$right_context_final"


# don't do the overlap thing for the priors computation data-- but do use the
# same num-frames for the eg, which would be much more efficient in case it's a
# recurrent model and has a lot of frames of context.  In any case we're not
# doing SGD so there is no benefit in having short chunks.
priors_egs_opts="--left-context=$left_context --right-context=$right_context --num-frames=$frames_per_eg --compress=$compress"
[ $left_context_initial -ge 0 ] && priors_egs_opts="$priors_egs_opts --left-context-initial=$left_context_initial"
[ $right_context_final -ge 0 ] && priors_egs_opts="$priors_egs_opts --right-context-final=$right_context_final"


echo $left_context > $dir/info/left_context
echo $right_context > $dir/info/right_context
echo $left_context_initial > $dir/info/left_context_initial
echo $right_context_final > $dir/info/right_context_final

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
      nnet3-get-egs --num-pdfs=$num_pdfs $ivector_opts $priors_egs_opts "$priors_feats" \
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
    nnet3-discriminative-get-egs $ivector_opts $egs_opts \
    $dir/final.mdl "$valid_feats" scp:$dir/lat_special.scp \
    scp:$dir/ali_special.scp "ark:$dir/valid_diagnostic.degs" || touch $dir/.error &

  $cmd $dir/log/create_train_subset.log \
    nnet3-discriminative-get-egs $ivector_opts $egs_opts \
    $dir/final.mdl "$train_subset_feats" scp:$dir/lat_special.scp \
    scp:$dir/ali_special.scp  "ark:$dir/train_diagnostic.degs" || touch $dir/.error &
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
    nnet3-discriminative-get-egs $ivector_opts $egs_opts \
      --num-frames-overlap=$frames_overlap_per_eg \
      $dir/final.mdl "$feats" "ark,s,cs:gunzip -c $denlatdir/lat.JOB.gz |" \
      "scp:utils/filter_scp.pl $sdata/JOB/utt2spk $dir/ali.scp |" ark:- \| \
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
      [ -L $file ] && rm $(utils/make_absolute.sh $file)
      rm $file
    done
  done
  if [ $archives_multiple -gt 1 ]; then
    # there are some extra soft links that we should delete.
    for f in $dir/degs.*.*.ark; do rm $f; done
  fi
  echo "$0: removing temporary lattices"
  rm $dir/lat.*
  echo "$0: removing temporary alignments"
  rm $dir/ali.{ark,scp} 2>/dev/null
fi

wait

echo "$0: Finished preparing training examples"
