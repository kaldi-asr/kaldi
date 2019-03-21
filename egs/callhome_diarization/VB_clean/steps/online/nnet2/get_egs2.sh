#!/bin/bash

# Copyright 2012-2014 Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.
#
# This is modified from ../../nnet2/get_egs2.sh.  [note: get_egs2.sh is as get_egs.sh,
# but uses the newer, more compact way of writing egs. where we write multiple
# frames of labels in order to share the context.]
# This script combines the
# nnet-example extraction with the feature extraction directly from wave files;
# it uses the program online2-wav-dump-feature to do all parts of feature
# extraction: MFCC/PLP/fbank, possibly plus pitch, plus iVectors.  This script
# is intended mostly for cross-system training for online decoding, where you
# initialize the nnet from an existing, larger system.
#

# Begin configuration section.
cmd=run.pl
frames_per_eg=8   # number of frames of labels per example.  more->less disk space and
                  # less time preparing egs, but more I/O during training.
                  # note: the script may reduce this if reduce_frames_per_eg is true.

reduce_frames_per_eg=true  # If true, this script may reduce the frames_per_eg
                           # if there is only one archive and even with the
                           # reduced frames_pe_eg, the number of
                           # samples_per_iter that would result is less than or
                           # equal to the user-specified value.
num_utts_subset=300     # number of utterances in validation and training
                        # subsets used for shrinkage and diagnostics.
num_valid_frames_combine=0 # #valid frames for combination weights at the very end.
num_train_frames_combine=10000 # # train frames for the above.
num_frames_diagnostic=4000 # number of frames for "compute_prob" jobs
samples_per_iter=400000 # each iteration of training, see this many samples
                        # per job.  This is just a guideline; it will pick a number
                        # that divides the number of samples in the entire data.

stage=0
io_opts="--max-jobs-run 5" # for jobs with a lot of I/O, limits the number running at one time. 
random_copy=false

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 4 ]; then
  echo "Usage: $0 [opts] <data> <ali-dir> <online-nnet-dir> <egs-dir>"
  echo " e.g.: $0 data/train exp/tri3_ali exp/nnet2_online/nnet_a_gpu_online/ exp/nnet2_online/nnet_b/egs"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --cmd (utils/run.pl;utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --samples-per-iter <#samples;400000>             # Number of samples of data to process per iteration, per"
  echo "                                                   # process."
  echo "  --feat-type <lda|raw>                            # (by default it tries to guess).  The feature type you want"
  echo "                                                   # to use as input to the neural net."
  echo "  --frames-per-eg <frames;8>                       # number of frames per eg on disk"
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


for f in $data/wav.scp $alidir/ali.1.gz $alidir/final.mdl $alidir/tree $mdl $feature_conf; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done


nj=`cat $alidir/num_jobs` || exit 1;  # number of jobs in alignment dir...

sdata=$data/split$nj
utils/split_data.sh $data $nj

mkdir -p $dir/log $dir/info
! cmp $alidir/tree $online_nnet_dir/tree && \
   echo "$0: warning, tree from alignment dir does not match tree from online-nnet dir (OK if for multilingual)"
cp $alidir/tree $dir
grep -v '^--endpoint' $feature_conf >$dir/feature.conf || exit 1;
mkdir -p $dir/valid $dir/train_subset

# Get list of validation utterances. 
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



set -o pipefail
left_context=$(nnet-am-info $mdl | grep '^left-context' | awk '{print $2}') || exit 1;
right_context=$(nnet-am-info $mdl | grep '^right-context' | awk '{print $2}') || exit 1;
set +o pipefail


if [ $stage -le 0 ]; then
  echo "$0: working out number of frames of training data"
  num_frames=$(steps/nnet2/get_num_frames.sh $data)
  echo $num_frames > $dir/info/num_frames
else
  num_frames=`cat $dir/info/num_frames` || exit 1;
fi

# the + 1 is to round up, not down... we assume it doesn't divide exactly.
num_archives=$[$num_frames/($frames_per_eg*$samples_per_iter)+1]
# (for small data)- while reduce_frames_per_eg == true and the number of
# archives is 1 and would still be 1 if we reduced frames_per_eg by 1, reduce it
# by 1.
reduced=false
while $reduce_frames_per_eg && [ $frames_per_eg -gt 1 ] && \
  [ $[$num_frames/(($frames_per_eg-1)*$samples_per_iter)] -eq 0 ]; do
  frames_per_eg=$[$frames_per_eg-1]
  num_archives=1
  reduced=true
done
$reduced && echo "$0: reduced frames_per_eg to $frames_per_eg because amount of data is small."

echo $num_archives >$dir/info/num_archives
echo $frames_per_eg >$dir/info/frames_per_eg

# Working out number of egs per archive
egs_per_archive=$[$num_frames/($frames_per_eg*$num_archives)]
! [ $egs_per_archive -le $samples_per_iter ] && \
  echo "$0: script error: egs_per_archive=$egs_per_archive not <= samples_per_iter=$samples_per_iter" \
  && exit 1;

echo $egs_per_archive > $dir/info/egs_per_archive

echo "$0: creating $num_archives archives, each with $egs_per_archive egs, with"
echo "$0:   $frames_per_eg labels per example, and (left,right) context = ($left_context,$right_context)"

# Making soft links to storage directories.  This is a no-up unless
# the subdirectory $dir/storage/ exists.  See utils/create_split_dir.pl
for x in `seq $num_archives`; do
  utils/create_data_link.pl $dir/egs.$x.ark
  for y in `seq $nj`; do
    utils/create_data_link.pl $dir/egs_orig.$x.$y.ark
  done
done

nnet_context_opts="--left-context=$left_context --right-context=$right_context"

if [ $stage -le 2 ]; then
  echo "$0: Getting validation and training subset examples."
  rm $dir/.error 2>/dev/null
  echo "$0: ... extracting validation and training-subset alignments."
  set -o pipefail;
  for id in $(seq $nj); do gunzip -c $alidir/ali.$id.gz; done | \
    copy-int-vector ark:- ark,t:- | \
    utils/filter_scp.pl <(cat $dir/valid/uttlist $dir/train_subset/uttlist) | \
    gzip -c >$dir/ali_special.gz || exit 1;
  set +o pipefail; # unset the pipefail option.

  $cmd $dir/log/create_valid_subset.log \
    nnet-get-egs $ivectors_opt $nnet_context_opts "$valid_feats" \
    "ark,s,cs:gunzip -c $dir/ali_special.gz | ali-to-pdf $alidir/final.mdl ark:- ark:- | ali-to-post ark:- ark:- |" \
     "ark:$dir/valid_all.egs" || touch $dir/.error &
  $cmd $dir/log/create_train_subset.log \
    nnet-get-egs $ivectors_opt $nnet_context_opts "$train_subset_feats" \
     "ark,s,cs:gunzip -c $dir/ali_special.gz | ali-to-pdf $alidir/final.mdl ark:- ark:- | ali-to-post ark:- ark:- |" \
     "ark:$dir/train_subset_all.egs" || touch $dir/.error &
  wait;
  [ -f $dir/.error ] && echo "Error detected while creating train/valid egs" && exit 1;
  echo "... Getting subsets of validation examples for diagnostics and combination."
  $cmd $dir/log/create_valid_subset_combine.log \
    nnet-subset-egs --n=$num_valid_frames_combine ark:$dir/valid_all.egs \
        ark:$dir/valid_combine.egs || touch $dir/.error &
  $cmd $dir/log/create_valid_subset_diagnostic.log \
    nnet-subset-egs --n=$num_frames_diagnostic ark:$dir/valid_all.egs \
    ark:$dir/valid_diagnostic.egs || touch $dir/.error &

  $cmd $dir/log/create_train_subset_combine.log \
    nnet-subset-egs --n=$num_train_frames_combine ark:$dir/train_subset_all.egs \
    ark:$dir/train_combine.egs || touch $dir/.error &
  $cmd $dir/log/create_train_subset_diagnostic.log \
    nnet-subset-egs --n=$num_frames_diagnostic ark:$dir/train_subset_all.egs \
    ark:$dir/train_diagnostic.egs || touch $dir/.error &
  wait
  sleep 5  # wait for file system to sync.
  cat $dir/valid_combine.egs $dir/train_combine.egs > $dir/combine.egs

  for f in $dir/{combine,train_diagnostic,valid_diagnostic}.egs; do
    [ ! -s $f ] && echo "No examples in file $f" && exit 1;
  done
  rm $dir/valid_all.egs $dir/train_subset_all.egs $dir/{train,valid}_combine.egs $dir/ali_special.gz
fi

if [ $stage -le 3 ]; then
  # create egs_orig.*.*.ark; the first index goes to $num_archives,
  # the second to $nj (which is the number of jobs in the original alignment
  # dir)

  egs_list=
  for n in $(seq $num_archives); do
    egs_list="$egs_list ark:$dir/egs_orig.$n.JOB.ark"
  done
  echo "$0: Generating training examples on disk"
  
  # The examples will go round-robin to egs_list.
  $cmd $io_opts JOB=1:$nj $dir/log/get_egs.JOB.log \
    nnet-get-egs $ivectors_opt $nnet_context_opts --num-frames=$frames_per_eg "$feats" \
    "ark,s,cs:gunzip -c $alidir/ali.JOB.gz | ali-to-pdf $alidir/final.mdl ark:- ark:- | ali-to-post ark:- ark:- |" ark:- \| \
    nnet-copy-egs ark:- $egs_list || exit 1;
fi

if [ $stage -le 4 ]; then
  echo "$0: recombining and shuffling order of archives on disk"
  # combine all the "egs_orig.JOB.*.scp" (over the $nj splits of the data) and
  # shuffle the order, writing to the egs.JOB.ark

  egs_list=
  for n in $(seq $nj); do 
    egs_list="$egs_list $dir/egs_orig.JOB.$n.ark"
  done

  $cmd $io_opts $extra_opts JOB=1:$num_archives $dir/log/shuffle.JOB.log \
    nnet-shuffle-egs --srand=JOB "ark:cat $egs_list|" ark:$dir/egs.JOB.ark  || exit 1;
fi

if [ $stage -le 5 ]; then
  echo "$0: removing temporary archives"
  for x in `seq $num_archives`; do
    for y in `seq $nj`; do
      file=$dir/egs_orig.$x.$y.ark
      [ -L $file ] && rm $(utils/make_absolute.sh $file)
      rm $file
    done
  done
fi

echo "$0: Finished preparing training examples"
