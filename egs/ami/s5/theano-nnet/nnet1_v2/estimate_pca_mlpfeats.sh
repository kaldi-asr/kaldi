#!/bin/bash 

# Copyright 2012-2015 Brno University of Technology (author: Karel Vesely)
# Apache 2.0
# To be run from .. (one directory up from here)
# see ../run.sh for example

# Begin configuration section.
nj=4
cmd=run.pl

remove_last_components=4 # remove N last components from the nnet
nnet_fwdpass_opts=
nnet_fwdpass_tool=theano-nnet/nnet1_v2/nn_fwdpass.py

# PCA related
est_pca_opts="--dim=25"
num_pca_utt=5000

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

######## CONFIGURATION

required="$srcdata/feats.scp $nndir/final_nnet.pklz $nndir/feat_preprocess.pkl"
for f in $required; do
  [ ! -f $f ] && echo "$0: Missing $f" && exit 1;
done

utils/shuffle_list.pl --srand ${seed:-777} $srcdata/feats.scp | head -n $num_pca_utt | sort > $pcadir/feats.${num_pca_utt}.scp || exit 1;
(cd $pcadir; ln -s feats.${num_pca_utt}.scp feats.scp; cd -)

# Concat feature transform with trimmed MLP:
cp $nndir/feat_preprocess.pkl $pcadir/feat_preprocess.pkl
nnet=$pcadir/feature_extractor_nnet.pklz
# Create trimmed MLP:
python theano-nnet/nnet1_v2/nnet_copy.py \
  --remove-last-components=$remove_last_components \
  $nndir/final_nnet.pklz $nnet 2>$logdir/feature_extractor.log || exit 1
(cd $pcadir; ln -s feature_extractor_nnet.pklz final_nnet.pklz; cd -) #for nn_fwdpass.py

# Create info
python theano-nnet/nnet1_v2/nnet_info.py $nnet >$logdir/feature_extractor.nnet-info || exit 1;

echo "Estimating PCA matrix"
$cmd $logdir/pca_est.log \
  $nnet_fwdpass_tool $nnet_fwdpass_opts \
    --feat-preprocess=$pcadir/feat_preprocess.pkl \
    --utt2spk-file=$srcdata/utt2spk --cmvn-scp=$srcdata/cmvn.scp \
    $pcadir $pcadir/ \| \
    est-pca $est_pca_opts ark,t:- $pcadir/pca.mat || exit 1;


