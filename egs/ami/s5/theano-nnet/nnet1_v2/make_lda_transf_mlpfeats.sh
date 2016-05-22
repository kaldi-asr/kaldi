#!/bin/bash 

# Copyright 2015  Sri Harish Mallidi
# Apache 2.0

# Begin configuration section.
nj=4
cmd=run.pl

nnet_fwdpass_opts=
nnet_fwdpass_tool=theano-nnet/nnet1_v2/nn_fwdpass.py

use_gpu="no"
htk_save=false
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
   echo "usage: $0 [options] <tgt-data-dir> <src-data-dir> <lda-mlpfeats-dir> <log-dir> <abs-path-to-lda-transf-feat-dir>";
   echo "This script compute LDA transformed features of mlp featues"
   echo "Along with my.extras/steps/nnet/estimate_lda_mlpfeats.sh useful for doing"
   echo "LDA on Bottleneck, HATS, TANDEM features"
   echo "options: "
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

data=$1
srcdata=$2
ldadir=$3
logdir=$4
ldatransf_mlpfeadir=$5

######## CONFIGURATION

# copy the dataset metadata from srcdata.
mkdir -p $data || exit 1;
cp $srcdata/* $data 2>/dev/null; rm $data/feats.scp $data/cmvn.scp;

# make $bnfeadir an absolute pathname.
ldatransf_mlpfeadir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $ldatransf_mlpfeadir ${PWD}`

mkdir -p $ldatransf_mlpfeadir || exit 1;
mkdir -p $data || exit 1;
mkdir -p $logdir || exit 1;


srcscp=$srcdata/feats.scp
scp=$data/feats.scp

nnet=$ldadir/feature_extractor_nnet.pklz
feat_preprocess=$ldadir/feat_preprocess.pkl
ldamat=$ldadir/lda.mat 

required="$srcscp $nnet $feat_preprocess $ldamat"
for f in $required; do
  if [ ! -f $f ]; then
    echo "$0: no such file $f"
    exit 1;
  fi
done

name=$(basename $srcdata)
sdata=$srcdata/split$nj
[[ -d $sdata && $srcdata/feats.scp -ot $sdata ]] || split_data.sh $srcdata $nj || exit 1;

#Compute LDA transf feats
$cmd JOB=1:$nj $logdir/make_lda_transf_${name}.JOB.log \
  $nnet_fwdpass_tool $nnet_fwdpass_opts \
    --feat-preprocess=$feat_preprocess \
    --utt2spk-file=$srcdata/utt2spk --cmvn-scp=$srcdata/cmvn.scp \
    $ldadir $sdata/JOB/ \| \
    transform-feats ${ldamat} ark,t:- ark,scp:${ldatransf_mlpfeadir}/lda_transf_$name.JOB.ark,${ldatransf_mlpfeadir}/lda_transf_$name.JOB.scp || exit 1;

# check that the sentence counts match
N0=$(cat $srcdata/feats.scp | wc -l) 
N1=$(cat $ldatransf_mlpfeadir/lda_transf_$name.*.scp | wc -l)
if [[ "$N0" != "$N1" ]]; then
  echo "Error producing features for $name:"
  echo "Original sentences : $N0  LDA transf MLP feats : $N1"
  exit 1;
fi

# concatenate the .scp files
for ((n=1; n<=nj; n++)); do
  cat ${ldatransf_mlpfeadir}/lda_transf_$name.$n.scp >> $data/feats.scp
done

echo "Succeeded creating MLP-BN features for $name ($data)"

# optionally resave in as HTK features:
if [ $htk_save == true ]; then
  echo -n "Resaving as HTK features into ${ldatransf_mlpfeadir}/htk ... "
  mkdir -p $ldatransf_mlpfeadir/htk
  $cmd JOB=1:$nj $logdir/htk_copy_bnfeats.JOB.log \
    copy-feats-to-htk --output-dir=${ldatransf_mlpfeadir}/htk --output-ext=fea scp:$data/feats.scp || exit 1
  echo "DONE!"
fi
