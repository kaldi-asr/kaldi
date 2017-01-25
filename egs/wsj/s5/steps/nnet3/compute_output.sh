#!/bin/bash

# Copyright 2012-2015  Johns Hopkins University (Author: Daniel Povey).
#                2016  Vimal Manohar
# Apache 2.0.

# This script does decoding with a neural-net.  If the neural net was built on
# top of fMLLR transforms from a conventional system, you should provide the
# --transform-dir option.

# Begin configuration section.
stage=1
transform_dir=    # dir to find fMLLR transforms.
nj=4 # number of jobs.  If --transform-dir set, must match that number!
cmd=run.pl
use_gpu=false
frames_per_chunk=50
ivector_scale=1.0
iter=final
extra_left_context=0
extra_right_context=0
extra_left_context_initial=-1
extra_right_context_final=-1
frame_subsampling_factor=1
feat_type=
compress=false
online_ivector_dir=
post_vec=
output_name=
use_raw_nnet=true
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: $0 [options] <data-dir> <nnet-dir> <output-dir>"
  echo "e.g.:   steps/nnet3/compute_output.sh --nj 8 \\"
  echo "--online-ivector-dir exp/nnet3/ivectors_test_eval92 \\"
  echo "    data/test_eval92_hires exp/nnet3/tdnn exp/nnet3/tdnn/output"
  echo "main options (for others, see top of script file)"
  echo "  --transform-dir <decoding-dir>           # directory of previous decoding"
  echo "                                           # where we can find transforms for SAT systems."
  echo "  --config <config-file>                   # config containing options"
  echo "  --nj <nj>                                # number of parallel jobs"
  echo "  --cmd <cmd>                              # Command to run in parallel with"
  echo "  --iter <iter>                            # Iteration of model to decode; default is final."
  exit 1;
fi

data=$1
srcdir=$2
dir=$3

if ! $use_raw_nnet; then
  [ ! -f $srcdir/$iter.mdl ] && echo "$0: no such file $srcdir/$iter.mdl" && exit 1
  prog=nnet3-am-compute
  model="$srcdir/$iter.mdl"
else 
  [ ! -f $srcdir/$iter.raw ] && echo "$0: no such file $srcdir/$iter.raw" && exit 1
  prog=nnet3-compute
  model="nnet3-copy $srcdir/$iter.raw - |"
fi

mkdir -p $dir/log
echo "rename-node old-name=$output_name new-name=output" > $dir/edits.config

if [ ! -z "$output_name" ]; then
  model="$model nnet3-copy --edits-config=$dir/edits.config - - |"
else
  output_name=output
fi

[ ! -z "$online_ivector_dir" ] && \
  extra_files="$online_ivector_dir/ivector_online.scp $online_ivector_dir/ivector_period"

for f in $data/feats.scp $extra_files; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

sdata=$data/split$nj;
cmvn_opts=`cat $srcdir/cmvn_opts` || exit 1;

[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs


## Set up features.
if [ -z "$feat_type" ]; then
  if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=raw; fi
  echo "$0: feature type is $feat_type"
fi

splice_opts=`cat $srcdir/splice_opts 2>/dev/null`

case $feat_type in
  raw) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |";;
  lda) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"
    ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac
if [ ! -z "$transform_dir" ]; then
  echo "$0: using transforms from $transform_dir"
  [ ! -s $transform_dir/num_jobs ] && \
    echo "$0: expected $transform_dir/num_jobs to contain the number of jobs." && exit 1;
  nj_orig=$(cat $transform_dir/num_jobs)

  if [ $feat_type == "raw" ]; then trans=raw_trans;
  else trans=trans; fi
  if [ $feat_type == "lda" ] && \
    ! cmp $transform_dir/../final.mat $srcdir/final.mat && \
    ! cmp $transform_dir/final.mat $srcdir/final.mat; then
    echo "$0: LDA transforms differ between $srcdir and $transform_dir"
    exit 1;
  fi
  if [ ! -f $transform_dir/$trans.1 ]; then
    echo "$0: expected $transform_dir/$trans.1 to exist (--transform-dir option)"
    exit 1;
  fi
  if [ $nj -ne $nj_orig ]; then
    # Copy the transforms into an archive with an index.
    for n in $(seq $nj_orig); do cat $transform_dir/$trans.$n; done | \
       copy-feats ark:- ark,scp:$dir/$trans.ark,$dir/$trans.scp || exit 1;
    feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk scp:$dir/$trans.scp ark:- ark:- |"
  else
    feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$transform_dir/$trans.JOB ark:- ark:- |"
  fi
elif grep 'transform-feats --utt2spk' $srcdir/log/train.1.log >&/dev/null; then
  echo "$0: **WARNING**: you seem to be using a neural net system trained with transforms,"
  echo "  but you are not providing the --transform-dir option in test time."
fi
##

if [ ! -z "$online_ivector_dir" ]; then
  ivector_period=$(cat $online_ivector_dir/ivector_period) || exit 1;
  ivector_opts="--online-ivectors=scp:$online_ivector_dir/ivector_online.scp --online-ivector-period=$ivector_period"
fi

frame_subsampling_opt=
if [ $frame_subsampling_factor -ne 1 ]; then
  # e.g. for 'chain' systems
  frame_subsampling_opt="--frame-subsampling-factor=$frame_subsampling_factor"
fi

if ! $use_raw_nnet; then
  output_wspecifier="ark:| copy-feats --compress=$compress ark:- ark:- | gzip -c > $dir/log_likes.JOB.gz"
else 
  output_wspecifier="ark:| copy-feats --compress=$compress ark:- ark:- | gzip -c > $dir/nnet_output.JOB.gz"

  if [ ! -z $post_vec ]; then
    if [ $stage -le 1 ]; then
      copy-vector --binary=false $post_vec - | \
        awk '{for (i = 2; i < NF; i++) { sum += i; };
      printf ("[");
      for (i = 2; i < NF; i++) { printf " "log(i/sum); };
      print (" ]");}' > $dir/log_priors.vec
    fi

    output_wspecifier="ark:| matrix-add-offset ark:- 'vector-scale --scale=-1.0 $dir/log_priors.vec - |' ark:- | copy-feats --compress=$compress ark:- ark:- | gzip -c > $dir/log_likes.JOB.gz"
  fi
fi

gpu_opt="--use-gpu=no"
gpu_queue_opt=

if $use_gpu; then
  gpu_queue_opt="--gpu 1"
  gpu_opt="--use-gpu=yes"
fi

if [ $stage -le 2 ]; then
  $cmd $gpu_queue_opt JOB=1:$nj $dir/log/compute_output.JOB.log \
    $prog $gpu_opt $ivector_opts $frame_subsampling_opt \
     --frames-per-chunk=$frames_per_chunk \
     --extra-left-context=$extra_left_context \
     --extra-right-context=$extra_right_context \
     --extra-left-context-initial=$extra_left_context_initial \
     --extra-right-context-final=$extra_right_context_final \
     "$model" "$feats" "$output_wspecifier" || exit 1;
fi

exit 0;

