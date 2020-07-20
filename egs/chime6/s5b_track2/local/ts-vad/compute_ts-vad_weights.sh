#!/bin/bash
# Copyright 2012  Brno University of Technology (Author: Karel Vesely)
#           2013  Johns Hopkins University (Author: Daniel Povey)
#           2015  Vijayaditya Peddinti
#           2016  Vimal Manohar
#           2017  Pegah Ghahremani
#           2020  Ivan Medennikov (STC-innovations Ltd)
# Apache 2.0

# Computes TS-VAD weights using raw nnet3 network.

# Begin configuration section.
nj=4
cmd=run.pl
stage=0
# Begin configuration.
srcdir=
frames_per_chunk=50
extra_left_context=0
extra_right_context=0
extra_left_context_initial=-1
extra_right_context_final=-1
online_ivector_dir=
graphs_scp=
max_jobs_run=20
n_spk=4

normalize_transform=
add_deltas=false
delta_opts=
num_threads=1
use_gpu=true
mb_size=128
optimize=false
apply_exp=true
use_subsampling=false
# End configuration options.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: $0 <data-dir> <extractor> <bn-dir>"
   echo "e.g.: $0 data/train exp/nnet4/bnex.raw data_bn/train"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
extractor=$2
dir=$3

if [ -f $dir/.done ]; then
  echo "$0: $dir/.done already exists!"
  exit 0;
fi

[ -z $srcdir ] && srcdir=`dirname $extractor`

mkdir -p $dir/{log,tmp}
sdata=$data/split${nj}
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || \
   split_data.sh $data $nj || exit 1;

extra_files=
if [ ! -z "$online_ivector_dir" ]; then
  steps/nnet2/check_ivectors_compatible.sh $srcdir $online_ivector_dir || exit 1
  extra_files="$online_ivector_dir/ivector_online.scp $online_ivector_dir/ivector_period"
fi

for f in $extractor $data/feats.scp $extra_files; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

delta_opts=`cat $srcdir/delta_opts 2>/dev/null`
[ ! -z "$delta_opts" ] && add_deltas=true

[ -z "$normalize_transform" ] && [ -f $srcdir/normalize.feature_transform ] && normalize_transform=$srcdir/normalize.feature_transform
echo "normalize transform file: $normalize_transform"

cmvn_opts=`cat $srcdir/cmvn_opts 2>/dev/null`

feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"

if [ ! -z "$normalize_transform" ]; then
  feats="$feats nnet-forward $normalize_transform ark:- ark:- |"
fi

if $add_deltas; then
  feats="$feats add-deltas $delta_opts ark:- ark:- |"
fi

ivector_opts=
if [ ! -z "$online_ivector_dir" ]; then
  ivector_period=$(cat $online_ivector_dir/ivector_period) || exit 1;
  ivector_opts="--online-ivectors=scp:$online_ivector_dir/ivector_online.scp --online-ivector-period=$ivector_period"
fi

frame_subsampling_opt=
if [ -f $srcdir/frame_subsampling_factor ] && $use_subsampling ; then
  # e.g. for 'chain' systems
  frame_subsampling_factor=$(cat $srcdir/frame_subsampling_factor)
  frame_subsampling_opt="--frame-subsampling-factor=$frame_subsampling_factor"
  cp $srcdir/frame_subsampling_factor $dir
  if [[ $frame_subsampling_factor -gt 1 ]]; then
    # Assume a chain system, check agrument sanity.
    if [[ ! ($scale_opts == *--self-loop-scale=1.0* &&
             $scale_opts == *--transition-scale=1.0* &&
             $acoustic_scale = '1.0') ]]; then
      echo "$0: ERROR: frame-subsampling-factor is not 1, assuming a chain system."
      echo "... You should pass the following options to this script:"
      echo "  --scale-opts '--transition-scale=1.0 --self-loop-scale=1.0'" \
           "--acoustic_scale 1.0"
    fi
  fi
fi

##
gpu_opt=
thread_string=
if $use_gpu ; then
  thread_string="-batch --minibatch-size=$mb_size"
  gpu_opt="--gpu 1"
  use_gpu=wait
else
  echo "Warning! GPU is disabled, are you okay?"
  thread_string=""
  use_gpu=no
fi

if [ $stage -le 1 ]; then
  $cmd --max-jobs-run $max_jobs_run $gpu_opt JOB=1:$nj $dir/log/nnet3_compute.JOB.log \
    nnet3-compute$thread_string $ivector_opts $frame_subsampling_opt \
      --apply-exp=$apply_exp \
      --frames-per-chunk=$frames_per_chunk \
      --extra-left-context=$extra_left_context \
      --extra-right-context=$extra_right_context \
      --extra-left-context-initial=$extra_left_context_initial \
      --extra-right-context-final=$extra_right_context_final \
      --use-gpu=$use_gpu \
      $extractor "$feats" ark,t:$dir/tmp/outputs.JOB.ark || exit 1;
  cat $dir/tmp/outputs.*.ark > $dir/outputs.ark
  rm $dir/tmp/outputs.*.ark
fi

if [ $stage -le 2 ]; then
  [ -f $dir/weights.ark ] && rm $dir/weights.ark
  for i in `seq $n_spk`; do
    $cmd $dir/log/make_weights.$i.log \
      select-feats $((2*i-1)) ark:$dir/outputs.ark ark:- \| \
      feat-to-post ark:- ark:- \| \
      post-to-weights ark:- ark,t:"| sed s/\ /-$i\ / > $dir/weights.$i.ark" || exit 1;
  done
  cat $dir/weights.*.ark | sort > $dir/weights.ark
  rm $dir/outputs.ark
  rm $dir/weights.*.ark
fi

echo "$0: done extracting weights"
