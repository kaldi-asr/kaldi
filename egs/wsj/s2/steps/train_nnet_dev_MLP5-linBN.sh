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
# Neural network training, using fbank features, cepstral mean normalization 
# and hamming-dct transform
#
# The network is 5-layer MLP with 3 hidden layers, the middle layer (bottleneck) has linear activation
#
# Two datasets are used: trainset and devset (for early stopping/model selection)



while [ 1 ]; do
  case $1 in
    --model-size)
      shift; modelsize=$1; shift;
      ;;
    --lrate)
      shift; lrate=$1; shift;
      ;;
    --bunchsize)
      shift; bunchsize=$1; shift;
      ;;
    --l2-penalty)
      shift; l2penalty=$1; shift;
      ;;
    --bn-size)
      shift; bnsize=$1; shift;
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
    *)
      break;
      ;;
  esac
done


if [ $# != 6 ]; then
   echo "Usage: steps/train_nnet.sh <data-dir> <data-dev> <lang-dir> <ali-dir> <ali-dev> <exp-dir>"
   echo " e.g.: steps/train_nnet.sh data/train data/cv data/lang exp/mono_ali exp/mono_ali_cv exp/mono_nnet"
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
TRAIN_TOOL="nnet-train-xent-hardlab-frmshuff"

#feature config
echo norm_vars ${norm_vars:=false} #false:CMN, true:CMVN on fbanks
echo fea_dim: ${fea_dim:=23}     #FBANK dimension
echo splice_lr: ${splice_lr:=15}   #left- and right-splice value
echo dct_basis: ${dct_basis:=16}   #number of DCT basis computed from temporal trajectory of single band

#mlp size
echo modelsize: ${modelsize:=1000000} #number of free parameters in the MLP
echo bnsize: ${bnsize:=30} #dimensionality of the bottleneck


#global config for trainig
max_iters=20
start_halving_inc=0.5
end_halving_inc=0.1
halving_factor=0.5
echo lrate: ${lrate:=0.015} #learning rate
echo bunchsize: ${bunchsize:=256} #size of the Stochastic-GD update block
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
scripts/count_class_frames.awk $dir/train.pdf $dir/cur.counts
#copy the old transition model, will be needed by decoder
copy-transition-model --binary=false $alidir/final.mdl $dir/transition.mdl
cp $alidir/tree $dir/tree

###### PREPARE FEATURES ######
# shuffle the list
echo "Preparing train/cv lists"
cat $data/feats.scp.fbank | scripts/shuffle_list.pl ${seed:-777} > $dir/train.scp
cp $data_cv/feats.scp.fbank $dir/cv.scp
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


###### INITIALIZE THE NNET ######
echo -n "Initializng MLP: "
num_fea=$((fea_dim*dct_basis))
num_tgt=$(gmm-copy --print-args=false --binary=false $alidir/final.mdl - 2>$dir/gmm-copy.log | grep NUMPDFS | awk '{ print $4 }')
num_bn=$bnsize
num_hid=$((modelsize / (num_fea+num_tgt+2*num_bn)))
mlp_init=$dir/nnet_${num_fea}_${num_hid}_${num_bn}lin_${num_hid}_${num_tgt}.init
echo " $mlp_init"
scripts/gen_mlp_init.py --dim=${num_fea}:${num_hid}:${num_bn}:${num_hid}:${num_tgt} --gauss --negbias --seed=777 --linBNdim=$num_bn > $mlp_init



###### TRAIN ######
echo "Starting training:"
source scripts/train_nnet_scheduler.sh
echo "Training finished."
if [ "" == "$mlp_final" ]; then
  echo "No final network returned!"
else
  cp $mlp_final $dir/final.nnet
  echo "Final network $mlp_final"
fi

