#!/bin/bash 

# Copyright 2012-2015 Brno University of Technology (author: Karel Vesely)
# Apache 2.0
# To be run from .. (one directory up from here)
# see ../run.sh for example

# Begin configuration section.
nj=4
cmd=run.pl
stage=0

remove_last_components=4 # remove N last components from the nnet
nnet_forward_opts=
use_gpu=no
htk_save=false
ivector=            # rx-specifier with i-vectors (ark-with-vectors),

# LDA related
randprune=4.0 # This is approximately the ratio by which we will speed up the
              # LDA and MLLT calculations via randomized pruning.
lda_dim=300


# transform nnet out opts
transf_nnet_out_opts=

# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

set -euo pipefail

if [ $# != 6 ]; then
   echo "usage: $0 [options] <data-dir> <lang-dir> <ali-dir> <multi-stream-opts> <nnet-dir> <lda-dir>";
   echo "options: "
   echo "  --cmd 'queue.pl <queue opts>'   # how to run jobs."
   echo "  --nj <nj>                       # number of parallel jobs"
   echo "  --remove-last-components <N>    # number of NNet Components to remove from the end"
   echo "  --use-gpu (no|yes|optional)     # forwarding on GPU"
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

data=$1
lang=$2
alidir=$3
multi_stream_opts=$4
nndir=$5
ldadir=$6

mkdir -p $ldadir/{log,data}
logdir=$ldadir/log

######## CONFIGURATION

required="$data/feats.scp $nndir/final.nnet $nndir/final.feature_transform"
for f in $required; do
  [ ! -f $f ] && echo "$0: Missing $f" && exit 1;
done

silphonelist=`cat $lang/phones/silence.csl` || exit 1;

name=$(basename $data)
sdata=$data/split$nj
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

# Prepare $ldadir from $nndir 
# do during test we can just $ldadir
## require for feature extraction pipeline
D=$nndir
[ -e $D/norm_vars ] && cp $D/norm_vars $ldadir/
[ -e $D/cmvn_opts ] && cp $D/cmvn_opts $ldadir/
[ -e $D/delta_order ] && cp $D/delta_order $ldadir/
[ -e $D/delta_opts ] && cp $D/delta_opts $ldadir/
[ -e $D/pytel_transform.py ] && cp $D/pytel_transform.py $ldadir/ 
[ -e $D/stream_combination.list ] && cp $D/stream_combination.list $ldadir/ 
[ -e $D/ivector_dim ] && cp $D/ivector_dim $ldadir/

## prepare nnet related
cp $nndir/final.feature_transform $ldadir/
feature_transform=$ldadir/final.feature_transform
nnet=$ldadir/feature_extractor.nnet
nnet-copy --remove-last-components=$remove_last_components $nndir/final.nnet $nnet 2>$logdir/feature_extractor.log || exit 1
nnet-info $nnet >$ldadir/feature_extractor.nnet-info
# from here-on we can just $ldadir

# PREPARE FEATURE EXTRACTION PIPELINE
# import config,
cmvn_opts=
delta_opts=
D=$ldadir
[ -e $D/norm_vars ] && cmvn_opts="--norm-means=true --norm-vars=$(cat $D/norm_vars)" # Bwd-compatibility,
[ -e $D/cmvn_opts ] && cmvn_opts=$(cat $D/cmvn_opts)
[ -e $D/delta_order ] && delta_opts="--delta-order=$(cat $D/delta_order)" # Bwd-compatibility,
[ -e $D/delta_opts ] && delta_opts=$(cat $D/delta_opts)
#
# Create the feature stream,
feats="ark,s,cs:copy-feats scp:$sdata/JOB/feats.scp ark:- |"
# apply-cmvn (optional),
[ ! -z "$cmvn_opts" -a ! -f $sdata/1/cmvn.scp ] && echo "$0: Missing $sdata/1/cmvn.scp" && exit 1
[ ! -z "$cmvn_opts" ] && feats="$feats apply-cmvn $cmvn_opts --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp ark:- ark:- |"
# add-deltas (optional),
[ ! -z "$delta_opts" ] && feats="$feats add-deltas $delta_opts ark:- ark:- |"
# add-pytel transform (optional),
[ -e $D/pytel_transform.py ] && feats="$feats /bin/env python $D/pytel_transform.py |"

# add-ivector (optional),
if [ -e $D/ivector_dim ]; then
  ivector_dim=$(cat $D/ivector_dim)
  [ -z $ivector ] && echo "Missing --ivector, they were used in training! (dim $ivector_dim)" && exit 1
  ivector_dim2=$(copy-vector --print-args=false "$ivector" ark,t:- | head -n1 | awk '{ print NF-3 }') || true
  [ $ivector_dim != $ivector_dim2 ] && "Error, i-vector dimensionality mismatch! (expected $ivector_dim, got $ivector_dim2 in $ivector)" && exit 1
  # Append to feats
  feats="$feats append-vector-to-feats ark:- '$ivector' ark:- |"
fi

# Add Multi-stream options
feats="$feats nnet-forward $feature_transform ark:- ark:- | apply-feature-stream-mask $multi_stream_opts ark:- ark:- |"

# keep track of transf_nnet_out_opts, 
# so no need to supply it
echo "$transf_nnet_out_opts" >$ldadir/transf_nnet_out_opts

#
nnet_feats="$feats nnet-forward $nnet_forward_opts --use-gpu=$use_gpu $nnet ark:- ark:- | transform-nnet-posteriors $transf_nnet_out_opts ark:- ark:- |"

dir=$ldadir
transf=$dir/lda$lda_dim.mat

if [ $stage -le 0 ]; then
echo "Accumulating LDA statistics."
$cmd JOB=1:$nj $logdir/lda_acc.JOB.log \
  ali-to-post "ark:gunzip -c $alidir/ali.*.gz|" ark:- \| \
    weight-silence-post 0.0 $silphonelist $alidir/final.mdl ark:- ark:- \| \
    acc-lda --rand-prune=$randprune $alidir/final.mdl "$nnet_feats" ark:- $dir/data/lda_acc.JOB || exit 1;
fi

lda_acc_str=""
for ((n=1; n<=nj; n++)); do
  lda_acc_str=$lda_acc_str" "$ldadir/data/lda_acc.${n}
done

echo "Estimating LDA matrix."
$cmd $logdir/lda_est.log \
  est-lda --verbose=2 --dim=$lda_dim $transf $lda_acc_str || exit 1;

(cd $dir; ln -s lda${lda_dim}.mat lda.mat; cd -)

exit 0;
