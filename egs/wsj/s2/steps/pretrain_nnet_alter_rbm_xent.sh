#!/bin/bash
# Copyright 2012 Karel Vesely

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# To be run from ..
#
# Deep neural network pre-training,using fbank features, 
# cepstral mean normalization and hamming-dct transform
#
# Two datasets are used: trainset and devset (for early stopping/model selection)


while [ 1 ]; do
  case $1 in
    --nn-depth)
      shift; nn_depth=$1; shift;
      ;;
    --nn-dimhid)
      shift; nn_dimhid=$1; shift;
      ;;
    --rbm-iter)
      shift; iters_rbm=$1; shift;
      ;;
    --rbm-lrate)
      shift; rbm_lrate=$1; shift;
      ;;
    --rbm-lrate-low)
      shift; rbm_lrate_low=$1; shift;
      ;;
    --lrate)
      shift; lrate=$1; shift;
      ;;
    --l2-penalty)
      shift; l2penalty=$1; shift;
      ;;
    --norm-vars)
      shift; norm_vars=$1; shift;
      ;;
    --fea-dim)
      shift; fea_dim=$1; shift;
      ;;
    --splice-lr)
      shift; splice_lr=$1; shift;
      ;;
    --dct-basis)
      shift; dct_basis=$1; shift;
      ;;
    --*)
      echo Unknown option $1; exit 1;
      ;;
    *)
      break;
      ;;
  esac
done


if [ $# != 6 ]; then
   echo "Usage: steps/pretrain_nnet_dev_alter_rbm_xent.sh <data-dir> <data-dev> <lang-dir> <ali-dir> <ali-dev> <exp-dir>"
   echo " e.g.: steps/pretrain_nnet_dev_alter_rbm_xent.sh data/train data/cv data/lang exp/mono_ali exp/mono_ali_cv exp/mono_nnet"
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

data=$1
data_cv=$2
lang=$3
alidir=$4
alidir_cv=$5
dir=$6

if [ ! -f $alidir/final.mdl -o ! -f $alidir/ali.gz ]; then
  echo "Error: alignment dir $alidir does not contain final.mdl and ali.gz"
  exit 1;
fi




######## CONFIGURATION
TRAIN_TOOL="nnet-train-xent-hardlab-frmshuff --bunchsize=64 "

#feature config
echo norm_vars ${norm_vars:=false} #false:CMN, true:CMVN on fbanks
echo fea_dim: ${fea_dim:=23}       #FBANK dimension
echo splice_lr: ${splice_lr:=15}   #left- and right-splice value
echo dct_basis: ${dct_basis:=16}   #number of DCT basis computed from temporal trajectory of single band

#mlp size
echo nn_depth:  ${nn_depth:=10}    #number of hidden layers
echo nn_dimhid: ${nn_dimhid:=1024} #dimension of hidden layers


#global config for trainig
iters_rbm_init=1 #number of iterations with low mmt
echo iters_rbm: ${iters_rbm:=1} #number of iterations with high mmt
iters_rbm_low_lrate=$((2*iters_rbm)) #number of iterations for RBMs with gaussian input
iters_xent=1 #number of iterations of discriminative fine-tuning

start_halving_inc=0.5
end_halving_inc=0.1
halving_factor=0.5

#parameters for RBM pre-training
echo rbm_lrate: ${rbm_lrate:=0.1}
echo rbm_lrate_low: ${rbm_lrate_low:=0.001}
echo rbm_momentum: ${rbm_momentum:=0.5}
echo rbm_momentum_high: ${rbm_momentum_high:=0.9}
echo rbm_l2penalty: ${rbm_l2penalty:=0.0002}

#parameters for discriminative fine-tuning
echo lrate: ${lrate:=0.015} #learning rate
echo l2penalty: ${l2penalty:=0.0} #L2 regularization penalty
########



mkdir -p $dir/{log,nnet}

###### PREPARE ALIGNMENTS ######
echo "Preparing alignments"
#convert ali to pdf
labels_tr="ark:$dir/train.pdf"
ali-to-pdf $alidir/final.mdl "ark:gunzip -c $alidir/ali.gz |" t,$labels_tr 2> $dir/ali2pdf_tr.log || exit 1
#convert ali to pdf (cv set)
labels_cv="ark:$dir/cv.pdf"
ali-to-pdf $alidir_cv/final.mdl "ark:gunzip -c $alidir_cv/ali.gz |" t,$labels_cv 2> $dir/ali2pdf_cv.log || exit 1
#merge the two parts (scheduler expects one file in $labels)
labels="ark:$dir/cur.pdf"
cat $dir/train.pdf $dir/cv.pdf > $dir/cur.pdf

#get the priors, count the class examples from alignments
pdf-to-counts ark:$dir/train.pdf $dir/cur.counts
#copy the old transition model, will be needed by decoder
copy-transition-model --binary=false $alidir/final.mdl $dir/final.mdl
cp $alidir/tree $dir/tree

###### PREPARE FEATURES ######
# shuffle the list
echo "Preparing train/cv lists"
cat $data/feats.scp | scripts/shuffle_list.pl ${seed:-777} > $dir/train.scp
cp $data_cv/feats.scp $dir/cv.scp
# print the list sizes
wc -l $dir/train.scp $dir/cv.scp

#compute per-speaker CMVN
echo "Computing cepstral mean and variance statistics"
cmvn="ark:$dir/cmvn.ark"
cmvn_cv="ark:$dir/cmvn_cv.ark"
compute-cmvn-stats --spk2utt=ark:$data/spk2utt scp:$dir/train.scp $cmvn 2>$dir/cmvn.log || exit 1
compute-cmvn-stats --spk2utt=ark:$data_cv/spk2utt scp:$dir/cv.scp $cmvn_cv 2>$dir/cmvn_cv.log || exit 1
feats_tr="ark:apply-cmvn --print-args=false --norm-vars=$norm_vars --utt2spk=ark:$data/utt2spk $cmvn scp:$dir/train.scp ark:- |"
feats_cv="ark:apply-cmvn --print-args=false --norm-vars=$norm_vars --utt2spk=ark:$data_cv/utt2spk $cmvn_cv scp:$dir/cv.scp ark:- |"

#add splicing
feats_tr="$feats_tr splice-feats --print-args=false --left-context=$splice_lr --right-context=$splice_lr ark:- ark:- |"
feats_cv="$feats_cv splice-feats --print-args=false --left-context=$splice_lr --right-context=$splice_lr ark:- ark:- |"

#generate hamming+dct transform
echo "Preparing Hamming DCT transform"
transf=$dir/hamm_dct.mat
scripts/gen_hamm_mat.py --fea-dim=$fea_dim --splice=$splice_lr > $dir/hamm.mat
scripts/gen_dct_mat.py --fea-dim=$fea_dim --splice=$splice_lr --dct-basis=$dct_basis > $dir/dct.mat
compose-transforms --binary=false $dir/dct.mat $dir/hamm.mat $transf 2>$dir/hamm_dct.log || exit 1
#convert transform to NNET format
{
  echo "<biasedlinearity> $((fea_dim*dct_basis)) $((fea_dim*(2*splice_lr+1)))"
  cat $transf
  echo -n ' [ '
  for i in $(seq $((fea_dim*dct_basis))); do echo -n '0 '; done
  echo ']'
} > $transf.net
#append transform to features
feats_tr="$feats_tr nnet-forward --print-args=false --silent=true $transf.net ark:- ark:- |"
feats_cv="$feats_cv nnet-forward --print-args=false --silent=true $transf.net ark:- ark:- |"

#renormalize the MLP input to zero mean and unit variance
echo "Renormalizing MLP input features"
cmvn_g="$dir/cmvn_glob.mat"
compute-cmvn-stats --binary=false "$feats_tr" $cmvn_g 2> $dir/cmvn_glob.log || exit 1
feats_tr="$feats_tr apply-cmvn --print-args=false --norm-vars=true $cmvn_g ark:- ark:- |"
feats_cv="$feats_cv apply-cmvn --print-args=false --norm-vars=true $cmvn_g ark:- ark:- |"


#get the DNN dimensions
num_fea=$((fea_dim*dct_basis))
num_hid=$nn_dimhid
num_tgt=$(hmm-info $alidir/final.mdl 2>$dir/gmm-info.log | grep pdfs | awk '{print $NF}')


###### PERFORM THE PRE-TRAINING ######
for depth in $(seq -f '%02g' 1 $nn_depth); do
  echo "%%%%%%% PRE-TRAINING DEPTH $depth"
  RBM=$dir/nnet/hid${depth}a_rbm.d/nnet/hid${depth}a_rbm
  mkdir -p $(dirname $RBM); mkdir -p $(dirname $RBM)/../log
  echo "Pre-training RBM $RBM "
  #The first RBM needs special treatment, because of Gussian input nodes
  if [ "$depth" == "01" ]; then
    #initialize the RBM with gaussian input
    scripts/gen_rbm_init.py --dim=${num_fea}:${num_hid} --gauss --negbias --vistype=gauss --hidtype=bern > $RBM.init
    #pre-train with reduced lrate and more iters
    #a)low momentum
    scripts/pretrain_rbm.sh --iters $iters_rbm_init --lrate $rbm_lrate_low --momentum $rbm_momentum --l2-penalty $rbm_l2penalty $RBM.init "$feats_tr" ${RBM}_mmt$rbm_momentum
    #b)high momentum
    scripts/pretrain_rbm.sh --iters $iters_rbm_low_lrate --lrate $rbm_lrate_low --momentum $rbm_momentum_high --l2-penalty $rbm_l2penalty ${RBM}_mmt$rbm_momentum "$feats_tr" ${RBM}_mmt${rbm_momentum_high}
  else
    #initialize the RBM
    scripts/gen_rbm_init.py --dim=${num_hid}:${num_hid} --gauss --negbias --vistype=bern --hidtype=bern > $RBM.init
    #pre-train (with higher learning rate)
    #a)low momentum
    scripts/pretrain_rbm.sh --feature-transform $TRANSF --iters $iters_rbm_init --lrate $rbm_lrate --momentum $rbm_momentum --l2-penalty $rbm_l2penalty $RBM.init "$feats_tr" ${RBM}_mmt$rbm_momentum
    #b)high momentum
    scripts/pretrain_rbm.sh --feature-transform $TRANSF --iters $iters_rbm --lrate $rbm_lrate --momentum $rbm_momentum_high --l2-penalty $rbm_l2penalty ${RBM}_mmt$rbm_momentum "$feats_tr" ${RBM}_mmt${rbm_momentum_high}
  fi

  #Compose trasform + RBM + multiclass logistic regression
  echo "Compsing the nnet for discriminative supervised trainng"
  NNET=$dir/nnet/hid${depth}b_nnet
  [ ! -r $TRANSF ] && rm $NNET.init 2>/dev/null
  [ -r $TRANSF ] && cat $TRANSF > $NNET.init
  rbm-convert-to-nnet --binary=false ${RBM}_mmt${rbm_momentum_high} - >> $NNET.init
  scripts/gen_mlp_init.py --dim=${num_hid}:${num_tgt} --gauss --negbias >> $NNET.init

  #Do single iteration of fine-tuning
  echo "Performing discriminative supervised trainng"
  scripts/pretrain_xent.sh --iters $iters_xent --lrate $lrate --l2-penalty $l2penalty $NNET.init "$feats_tr" "$feats_cv" "$labels" $NNET.xent

  #Cut the last layer (n=2:weights+softmax) in order 
  #to get the feature transform
  echo "Cutting the last layer"
  TRANSF=$dir/nnet/hid${depth}c_transf
  nnet-copy --remove-last-layers=2 --binary=false $NNET.xent $TRANSF
done


echo "Pre-training finished."

echo
echo "%%%% REPORT %%%%"
echo "% RBM pre-training progress"
grep -R progress $dir/nnet
echo "% Xent pre-training progress"
grep -R FRAME_ACCURACY $dir/nnet
echo 
echo "EOF"



#The final fine-tuning will be run from the outer level (the run.sh script),
#this will be done for all tne $NNET.xent networks...

