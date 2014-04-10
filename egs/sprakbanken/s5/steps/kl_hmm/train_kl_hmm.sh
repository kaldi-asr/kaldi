#!/bin/bash

# Copyright 2012-2013 Karel Vesely,
#                     Daniel Povey,
#                     Idiap Research Institute (Author: David Imseng)
# Apache 2.0

# Begin configuration section.  
nnet= # Optionally pre-select network to use for getting state-likelihoods
feature_transform= # Optionally pre-select feature transform (in front of nnet)
model= # Optionally pre-select transition model
class_frame_counts= # Optionally pre-select class-counts used to compute PDF priors 

stage=0 # stage=1 skips lattice generation
nj=32
cmd=$decode_cmd
max_active=7000 # maximum of active tokens
max_mem=50000000 # limit the fst-size to 50MB (larger fsts are minimized)
use_gpu="no" # disable gpu
parallel_opts="" 
tmpdir=
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: $0 [options] <data-dir> <align-dir> <kl-hmm-dir>"
   echo "... where <kl-hmm-dir> is assumed to be a sub-directory of the directory"
   echo " where the DNN + transition model is."
   echo "e.g.: $0 data/train exp/dnn1/kl-hmm-train"
   echo ""
   echo "This script works on plain or modified features (CMN,delta+delta-delta),"
   echo "which are then sent through feature-transform. It works out what type"
   echo "of features you used from content of srcdir."
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo ""
   echo "  --nnet <nnet>                                    # which nnet to use (opt.)"
   echo "  --feature-transform <nnet>                       # select transform in front of nnet (opt.)"
   echo "  --model <model>                                  # which transition model to use (opt.)"
   echo "  --tmpdir >dir>                                   # Temp directory to store the statistics, becuase they can get big (opt.)"
   exit 1;
fi


data=$1
alidir=$2
dir=$3
srcdir=`dirname $dir`; # The model directory is one level up from decoding directory.
sdata=$data/split$nj;

mkdir -p $dir/log
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

if [ -z "$nnet" ]; then # if --nnet <nnet> was not specified on the command line...
  nnet=$srcdir/final.nnet; 
fi
[ -z "$nnet" ] && echo "Error nnet '$nnet' does not exist!" && exit 1;

if [ -z "$model" ]; then # if --model <mdl> was not specified on the command line...
  model=$srcdir/final.mdl;
fi

# find the feature_transform to use
if [ -z "$feature_transform" ]; then
  feature_transform=$srcdir/final.feature_transform
fi
if [ ! -f $feature_transform ]; then
  echo "Missing feature_transform '$feature_transform'"
  exit 1
fi

# check that files exist
for f in $sdata/1/feats.scp $nnet_i $nnet $model; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

# Create the feature stream:
feats="ark,s,cs:copy-feats scp:$sdata/JOB/feats.scp ark:- |"
# Optionally add cmvn
if [ -f $srcdir/norm_vars ]; then
  norm_vars=$(cat $srcdir/norm_vars 2>/dev/null)
  [ ! -f $sdata/1/cmvn.scp ] && echo "$0: cannot find cmvn stats $sdata/1/cmvn.scp" && exit 1
  feats="$feats apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp ark:- ark:- |"
fi
# Optionally add deltas
if [ -f $srcdir/delta_order ]; then
  delta_order=$(cat $srcdir/delta_order)
  feats="$feats add-deltas --delta-order=$delta_order ark:- ark:- |"
fi

ali="ark:gunzip -c $alidir/ali.*.gz |"

if [[ ! -z $tmpdir ]]; then 
    mkdir -p $tmpdir 
else
    tmpdir=$dir
fi

nkl_states=$(hmm-info --print-args=false $alidir/final.mdl | grep pdfs | awk '{ print $NF }')
if [ $stage -le 0 ]; then
  $cmd $parallel_opts JOB=1:$nj $dir/log/acc-stats.JOB.log \
  nnet-kl-hmm-acc --nkl-states=${nkl_states} "ark:nnet-forward --feature-transform=$feature_transform --use-gpu=$use_gpu $nnet \"$feats\" ark:- |" "ark:ali-to-pdf --print-args=false $alidir/final.mdl \"$ali\" ark:- |" $tmpdir/kl-hmm-stats.JOB
fi

sum-matrices $dir/accumulated-kl-hmm-stats $tmpdir/kl-hmm-stats.*

rm $tmpdir/kl-hmm-stats.*

nnet-kl-hmm-mat-to-component $dir/kl-hmm.nnet $dir/accumulated-kl-hmm-stats

nnet-concat $dir/../final.nnet $dir/kl-hmm.nnet $dir/final.nnet

exit 0;
