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
# and hamming-dct transform.
#
# The network is simple 3-layer MLP with 1 hidden layer.
#
# Two datasets are used: trainset and devset (for early stopping/model selection)

while [ 1 ]; do
  case $1 in
    --model-size)
      shift; modelsize=$1; shift;
      ;;
    --hid-layers)
      shift; hid_layers=$1; shift;
      ;;
    --bn-dim)
      shift; bn_dim=$1; shift;
      ;;
    --hid-dim)
      shift; hid_dim=$1; shift;
      ;;
    --mlp-init)
      shift; mlp_init=$1; shift;
      ;;
    --learn-rate)
      shift; lrate=$1; shift;
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
    --norm-vars)
      shift; norm_vars=$1; shift;
      ;;
    --feat-type)
      shift; feat_type=$1; shift;
      ;;
    --splice-lr)
      shift; splice_lr=$1; shift;
      ;;
    --dct-basis)
      shift; dct_basis=$1; shift;
      ;;
    --min-iters)
      shift; min_iters=$1; shift;
      ;;
    --*)
      echo "ERROR : Unknown argument $1"; exit 1;
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
echo norm_vars: ${norm_vars:=false} #false:CMN, true:CMVN on fbanks
echo feat_type: ${feat_type:=traps}  #default features are traps
echo splice_lr: ${splice_lr:=15}   #left- and right-splice value
echo dct_basis: ${dct_basis:=16}   #number of DCT basis computed from temporal trajectory of single band

#mlp size
echo modelsize: ${modelsize:=1000000} #number of free parameters in the MLP
echo hid_layers: ${hid_layers:=2} #number of hidden layers
${bn_dim+: echo bn_dim: $bn_dim}

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
ali-to-pdf $alidir/final.mdl "ark:gunzip -c $alidir/ali.gz |" $labels_tr 2>$dir/train.pdf_log || exit 1
if [[ "$alidir" == "$alidir_cv" ]]; then
  labels=$labels_tr
else
  #convert ali to pdf (cv set)
  labels_cv="ark:$dir/cv.pdf"
  ali-to-pdf $alidir_cv/final.mdl "ark:gunzip -c $alidir_cv/ali.gz |" $labels_cv 2>$dir/cv.pdf_log || exit 1
  #merge the two parts (scheduler expects one file in $labels)
  labels="ark:$dir/train_and_cv.pdf"
  cat $dir/train.pdf $dir/cv.pdf > $dir/train_and_cv.pdf || exit 1
fi

#get the priors, count the class examples from alignments
analyze-counts ark:$dir/train.pdf $dir/train.counts 2>$dir/train.counts_log || exit 1
#copy the old transition model, will be needed by decoder
copy-transition-model --binary=false $alidir/final.mdl $dir/final.mdl 2>$dir/final.mdl_log || exit 1
cp $alidir/tree $dir/tree || exit 1

#analyze the train/cv alignments
scripts/analyze_alignments.sh "TRAINING SET" "ark:gunzip -c $alidir/ali.gz |" $dir/final.mdl $lang > $dir/__ali_stats_train
scripts/analyze_alignments.sh "VALIDATION SET" "ark:gunzip -c $alidir_cv/ali.gz |" $dir/final.mdl $lang > $dir/__ali_stats_cv


###### PREPARE FEATURES ######
# shuffle the list
echo "Preparing train/cv lists"
cat $data/feats.scp | scripts/shuffle_list.pl ${seed:-777} > $dir/train.scp
cp $data_cv/feats.scp $dir/cv.scp
# print the list sizes
wc -l $dir/train.scp $dir/cv.scp

#get feature dim
echo -n "Getting feature dim"
feat_dim=$(feat-to-dim scp:$dir/train.scp -)
echo $feat_dim

#compute per-speaker CMVN
echo "Computing cepstral mean and variance statistics"
cmvn="ark:$dir/cmvn.ark"
cmvn_cv="ark:$dir/cmvn_cv.ark"
compute-cmvn-stats --spk2utt=ark:$data/spk2utt scp:$dir/train.scp $cmvn 2>$dir/cmvn.ark_log || exit 1
compute-cmvn-stats --spk2utt=ark:$data_cv/spk2utt scp:$dir/cv.scp $cmvn_cv 2>$dir/cmvn_cv.ark_log || exit 1
feats_tr="ark:apply-cmvn --print-args=false --norm-vars=$norm_vars --utt2spk=ark:$data/utt2spk $cmvn scp:$dir/train.scp ark:- |"
feats_cv="ark:apply-cmvn --print-args=false --norm-vars=$norm_vars --utt2spk=ark:$data_cv/utt2spk $cmvn_cv scp:$dir/cv.scp ark:- |"
echo $norm_vars > $dir/norm_vars

#add splicing
feats_tr="$feats_tr splice-feats --print-args=false --left-context=$splice_lr --right-context=$splice_lr ark:- ark:- |"
feats_cv="$feats_cv splice-feats --print-args=false --left-context=$splice_lr --right-context=$splice_lr ark:- ark:- |"
echo $splice_lr > $dir/splice_lr

#choose further processing of spliced features
echo "Feature type : $feat_type"
case $feat_type in
  plain)
  ;;
  traps)
    #generate hamming+dct transform
    echo "Preparing Hamming DCT transform"
    transf=$dir/hamm_dct.mat
    scripts/gen_hamm_mat.py --fea-dim=$feat_dim --splice=$splice_lr > $dir/hamm.mat
    scripts/gen_dct_mat.py --fea-dim=$feat_dim --splice=$splice_lr --dct-basis=$dct_basis > $dir/dct.mat
    compose-transforms --binary=false $dir/dct.mat $dir/hamm.mat $transf 2>$dir/hamm_dct.mat_log || exit 1
    #convert transform to NNET format
    {
      echo "<biasedlinearity> $((feat_dim*dct_basis)) $((feat_dim*(2*splice_lr+1)))"
      cat $transf
      echo -n ' [ '
      for i in $(seq $((feat_dim*dct_basis))); do echo -n '0 '; done
      echo ']'
    } > $transf.net
    #append transform to features
    feats_tr="$feats_tr nnet-forward --print-args=false --silent=true $transf.net ark:- ark:- |"
    feats_cv="$feats_cv nnet-forward --print-args=false --silent=true $transf.net ark:- ark:- |"
  ;;
  transf)
    transf=$dir/final.mat
    [ ! -f $alidir/final.mat ] && echo "Missing transform $alidir/final.mat" && exit 1;
    cp $alidir/final.mat $transf
    echo "Copied transform $transf"
    feats_tr="$feats_tr transform-feats $transf ark:- ark:- |"
    feats_cv="$feats_cv transform-feats $transf ark:- ark:- |"
  ;;
  transf-sat)
    echo yet unimplemented...
    exit 1;
  ;;
  *)
    echo "Unknown feature type $feat_type"
    exit 1;
esac
echo $feat_type > $dir/feat_type #remember the type

#renormalize the MLP input to zero mean and unit variance
echo "Renormalizing MLP input features"
cmvn_g="$dir/cmvn_glob.mat"
compute-cmvn-stats --binary=false "$feats_tr" $cmvn_g 2> $dir/cmvn_glob.mat_log || exit 1
feats_tr="$feats_tr apply-cmvn --print-args=false --norm-vars=true $cmvn_g ark:- ark:- |"
feats_cv="$feats_cv apply-cmvn --print-args=false --norm-vars=true $cmvn_g ark:- ark:- |"


###### INITIALIZE THE NNET ######

if [ "" != "$mlp_init" ]; then
  echo "Using pre-initalized netwk $mlp_init";
else
  echo -n "Initializng MLP: "
  num_fea=$((feat_dim*dct_basis))
  num_tgt=$(hmm-info --print-args=false $alidir/final.mdl | grep pdfs | awk '{ print $NF }')
  # What is the topology?
  if [ "" == "$bn_dim" ]; then #MLP w/o bottleneck
    case "$hid_layers" in
      1) #3-layer MLP
        if [ "" != "$hid_dim" ]; then
          num_hid=$hid_dim
        else
          num_hid=$((modelsize/(num_fea+num_tgt)))
        fi
        mlp_init=$dir/nnet_${num_fea}_${num_hid}_${num_tgt}.init
        echo " $mlp_init"
        scripts/gen_mlp_init.py --dim=${num_fea}:${num_hid}:${num_tgt} \
          --gauss --negbias --seed=777 > $mlp_init
        ;;
      2|3|4|5|6|7|8|9|10) #(>3)-layer MLP
        if [ "" != "$hid_dim" ]; then
          num_hid=$hid_dim
        else
          a=$((hid_layers-1))
          b=$((num_fea+num_tgt))
          c=$((-modelsize))
          num_hid=$(awk "BEGIN{ num_hid= -$b/(2*$a) + sqrt($b^2 -4*$a*$c)/(2*$a); print int(num_hid) }") 
        fi
        mlp_init=$dir/nnet_${num_fea}
        dim_arg=${num_fea}
        for i in $(seq $hid_layers); do
          mlp_init=${mlp_init}_$num_hid
          dim_arg=${dim_arg}:${num_hid}
        done
        mlp_init=${mlp_init}_${num_tgt}.init
        dim_arg=${dim_arg}:${num_tgt}
        echo " $mlp_init"
        scripts/gen_mlp_init.py --dim=${dim_arg} --gauss --negbias --seed=777 > $mlp_init
        ;;
      *)
        echo "Unsupported number of hidden layers $hid_layers"
        exit 1;
    esac
  else #bn-syatem
    num_bn=$bn_dim
    case "$hid_layers" in # ie. number of layers in front of bottleneck
      1) #5-layer MLP
        if [ "" != "$hid_dim" ]; then
          num_hid=$hid_dim
        else
          num_hid=$((modelsize/(num_fea+num_tgt+(2*num_bn))))
        fi
        mlp_init=$dir/nnet_${num_fea}_${num_hid}_${num_bn}_${num_hid}_${num_tgt}.init
        echo " $mlp_init"
        scripts/gen_mlp_init.py --dim=${num_fea}:${num_hid}:${num_bn}:${num_hid}:${num_tgt} --gauss --negbias --seed=777 --linBNdim=$num_bn > $mlp_init
        ;;
      2|3|4|5|6|7|8|9|10) #(>5)-layer MLP
        if [ "" != "$hid_dim" ]; then
          num_hid=$hid_dim
        else
          a=$((hid_layers-1))
          b=$((num_fea+2*num_bn+num_tgt))
          c=$((-modelsize))
          num_hid=$(awk "BEGIN{ num_hid= -$b/(2*$a) + sqrt($b^2 -4*$a*$c)/(2*$a); print int(num_hid) }") 
        fi
        mlp_init=$dir/nnet_${num_fea}
        dim_arg=${num_fea}
        for i in $(seq $hid_layers); do
          mlp_init=${mlp_init}_$num_hid
          dim_arg=${dim_arg}:${num_hid}
        done
        mlp_init=${mlp_init}_${num_bn}lin_${num_hid}_${num_tgt}.init
        dim_arg=${dim_arg}:${num_bn}:${num_hid}:${num_tgt}
        echo " $mlp_init"
        scripts/gen_mlp_init.py --dim=${dim_arg} --gauss --negbias --seed=777 --linBNdim=$num_bn > $mlp_init
        ;;
      *)
        echo "Unsupported number of hidden layers $hid_layers"
        exit 1;
    esac
  fi
fi



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

