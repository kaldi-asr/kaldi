#!/bin/bash 

# Copyright 2012-2015 Brno University of Technology (author: Karel Vesely)
# Apache 2.0
# To be run from .. (one directory up from here)
# see ../run.sh for example

# Begin configuration section.
nj=4
cmd=run.pl

# nnet related
remove_last_components=4 # remove N last components from the nnet
nnet_fwdpass_opts=
nnet_fwdpass_tool=theano-nnet/nnet1_v2/pca_mlpfeats_acc.py
nnet_outdir= #Non-default location to write nnet_out

# PCA related
est_pca_opts="--dim=25"

# OTHER
seed=777    # seed value used for training data shuffling
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

set -euo pipefail

if [ $# != 3 ]; then
   echo "usage: $0 [options] <data-train> <nnet-dir> <pca-dir>";
   echo "e.g.: $0 data/train exp/dnn exp/pca_dnn"
   echo ""
   echo "This scripts estimates PCA on MLP activations. Useful for doing PCA on Bottleneck, HATS, TANDEM"
   echo "to compute PCA transf mlpfeats, use my.extras/steps/nnet/make_pca_transf_mlpfeats.sh"
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo ""
   echo "  --nnet <nnet>                                    # non-default location of DNN (opt.)"
   echo "  --nnetdir <dir>                                   # non-default dir with DNN/models, can be different"
   echo "                                                   # from parent dir of <decode-dir>' (opt.)"
   echo ""
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

srcdata=$1
nndir=$2
pcadir=$3

logdir=$pcadir/log; mkdir -p $logdir 
[ -z $nnet_outdir ] && nnet_outdir=$pcadir/data
mkdir -p $nnet_outdir

######## CONFIGURATION

required="$srcdata/feats.scp $nndir/final_nnet.pklz $nndir/feat_preprocess.pkl"
for f in $required; do
  [ ! -f $f ] && echo "$0: Missing $f" && exit 1;
done

name=$(basename $srcdata)
sdata=$srcdata/split$nj
[[ -d $sdata && $srcdata/feats.scp -ot $sdata ]] || split_data.sh $srcdata $nj || exit 1;

# Concat feature transform with trimmed MLP:
cp $nndir/feat_preprocess.pkl $pcadir/feat_preprocess.pkl
nnet=$pcadir/feature_extractor_nnet.pklz
# Create trimmed MLP:
python theano-nnet/nnet1_v2/nnet_copy.py \
  --remove-last-components=$remove_last_components \
  $nndir/final_nnet.pklz $nnet 2>$logdir/feature_extractor.log || exit 1
(cd $pcadir; ln -sf feature_extractor_nnet.pklz final_nnet.pklz; cd -) #for nn_fwdpass.py

# Create info
python theano-nnet/nnet1_v2/nnet_info.py $nnet >$logdir/feature_extractor.nnet-info || exit 1;

echo "Geting accumalators"
$cmd JOB=1:$nj $logdir/mlpfeats_acc.JOB.log \
  $nnet_fwdpass_tool $nnet_fwdpass_opts \
    --feat-preprocess=$pcadir/feat_preprocess.pkl \
    --utt2spk-file=$srcdata/utt2spk --cmvn-scp=$srcdata/cmvn.scp \
    $pcadir $sdata/JOB/ $pcadir/data/pca_acc.JOB.pklz || exit 1


pca_acc_str=""
for ((n=1; n<=nj; n++)); do
  pca_acc_str=$pca_acc_str" "$pcadir/data/pca_acc.${n}.pklz
done

echo "Estimating PCA matrix"
$cmd $logdir/pca_est.log \
  python theano-nnet/feature_funcs/est_pca.py "$est_pca_opts" \
    $pcadir/pca.mat $pca_acc_str || exit 1;

exit 0;


