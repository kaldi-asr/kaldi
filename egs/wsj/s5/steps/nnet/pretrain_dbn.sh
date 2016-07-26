#!/bin/bash
# Copyright 2013-2015 Brno University of Technology (author: Karel Vesely)

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

# To be run from ../../
#
# Restricted Boltzman Machine (RBM) pre-training by Contrastive Divergence
# algorithm (CD-1). A stack of RBMs forms a Deep Belief Neetwork (DBN).
#
# This script by default pre-trains on plain features (ie. saved fMLLR features),
# building a 'feature_transform' containing +/-5 frame splice and global CMVN.
#
# There is also a support for adding speaker-based CMVN, deltas, i-vectors,
# or passing custom 'feature_transform' or its prototype.
#

# Begin configuration.

# topology, initialization,
nn_depth=6             # number of hidden layers,
hid_dim=2048           # number of neurons per layer,
param_stddev_first=0.1 # init parameters in 1st RBM
param_stddev=0.1 # init parameters in other RBMs
input_vis_type=gauss # type of visible nodes on DBN input

# number of iterations,
rbm_iter=1            # number of pre-training epochs (Gaussian-Bernoulli RBM has 2x more)

# pre-training opts,
rbm_lrate=0.4         # RBM learning rate
rbm_lrate_low=0.01    # lower RBM learning rate (for Gaussian units)
rbm_l2penalty=0.0002  # L2 penalty (increases RBM-mixing rate)
rbm_extra_opts=

# data processing,
copy_feats=true     # resave the features to tmpdir,
copy_feats_tmproot=/tmp/kaldi.XXXX # sets tmproot for 'copy-feats',
copy_feats_compress=true # compress feats while resaving

# feature processing,
splice=5            # (default) splice features both-ways along time axis,
cmvn_opts=          # (optional) adds 'apply-cmvn' to input feature pipeline, see opts,
delta_opts=         # (optional) adds 'add-deltas' to input feature pipeline, see opts,
ivector=            # (optional) adds 'append-vector-to-feats', the option is rx-filename for the 2nd stream,
ivector_append_tool=append-vector-to-feats # (optional) the tool for appending ivectors,

feature_transform_proto= # (optional) use this prototype for 'feature_transform',
feature_transform=  # (optional) directly use this 'feature_transform',

# misc.
verbose=1 # enable per-cache reports
skip_cuda_check=false

# End configuration.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

set -euo pipefail

if [ $# != 2 ]; then
   echo "Usage: $0 <data> <exp-dir>"
   echo " e.g.: $0 data/train exp/rbm_pretrain"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>           # config containing options"
   echo ""
   echo "  --nn-depth <N>                   # number of RBM layers"
   echo "  --hid-dim <N>                    # number of hidden units per layer"
   echo "  --rbm-iter <N>                   # number of CD-1 iterations per layer"
   echo "                                   # can be used to subsample large datasets"
   echo "  --rbm-lrate <float>              # learning-rate for Bernoulli-Bernoulli RBMs"
   echo "  --rbm-lrate-low <float>          # learning-rate for Gaussian-Bernoulli RBM"
   echo ""
   echo "  --cmvn-opts  <string>            # add 'apply-cmvn' to input feature pipeline"
   echo "  --delta-opts <string>            # add 'add-deltas' to input feature pipeline"
   echo "  --splice <N>                     # splice +/-N frames of input features"
   echo "  --copy-feats <bool>              # copy features to /tmp, lowers storage stress"
   echo ""
   echo "  --feature_transform_proto <file> # use this prototype for 'feature_transform'"
   echo "  --feature-transform <file>       # directly use this 'feature_transform'"
   exit 1;
fi

data=$1
dir=$2

for f in $data/feats.scp; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

echo "# INFO"
echo "$0 : Pre-training Deep Belief Network as a stack of RBMs"
printf "\t dir       : $dir \n"
printf "\t Train-set : $data '$(cat $data/feats.scp | wc -l)'\n"
echo

[ -e $dir/${nn_depth}.dbn ] && echo "$0 Skipping, already have $dir/${nn_depth}.dbn" && exit 0

# check if CUDA compiled in and GPU is available,
if ! $skip_cuda_check; then cuda-gpu-available || exit 1; fi

mkdir -p $dir/log

###### PREPARE FEATURES ######
echo
echo "# PREPARING FEATURES"
if [ "$copy_feats" == "true" ]; then
  # re-save the features to local disk into /tmp/,
  tmpdir=$(mktemp -d $copy_feats_tmproot)
  trap "echo \"# Removing features tmpdir $tmpdir @ $(hostname)\"; ls $tmpdir; rm -r $tmpdir" INT QUIT TERM EXIT
  copy-feats --compress=$copy_feats_compress scp:$data/feats.scp ark,scp:$tmpdir/train.ark,$dir/train_sorted.scp || exit 1
else
  # or copy the list,
  cp $data/feats.scp $dir/train_sorted.scp
fi
# shuffle the list,
utils/shuffle_list.pl --srand 777 <$dir/train_sorted.scp >$dir/train.scp

# create a 10k utt subset for global cmvn estimates,
head -n 10000 $dir/train.scp > $dir/train.scp.10k

# for debugging, add list with non-local features,
utils/shuffle_list.pl --srand 777 <$data/feats.scp >$dir/train.scp_non_local

###### OPTIONALLY IMPORT FEATURE SETTINGS ######
ivector_dim= # no ivectors,
if [ ! -z $feature_transform ]; then
  D=$(dirname $feature_transform)
  echo "# importing feature settings from dir '$D'"
  [ -e $D/cmvn_opts ] && cmvn_opts=$(cat $D/cmvn_opts)
  [ -e $D/delta_opts ] && delta_opts=$(cat $D/delta_opts)
  [ -e $D/ivector_dim ] && ivector_dim=$(cat $D/ivector_dim)
  [ -e $D/ivector_append_tool ] && ivector_append_tool=$(cat $D/ivector_append_tool)
  echo "# cmvn_opts='$cmvn_opts' delta_opts='$delta_opts' ivector_dim='$ivector_dim'"
fi

###### PREPARE FEATURE PIPELINE ######
# read the features
feats_tr="ark:copy-feats scp:$dir/train.scp ark:- |"

# optionally add per-speaker CMVN
if [ ! -z "$cmvn_opts" ]; then
  echo "+ 'apply-cmvn' with '$cmvn_opts' using statistics : $data/cmvn.scp"
  [ ! -r $data/cmvn.scp ] && echo "Missing $data/cmvn.scp" && exit 1;
  [ ! -r $data/utt2spk ] && echo "Missing $data/utt2spk" && exit 1;
  feats_tr="$feats_tr apply-cmvn $cmvn_opts --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp ark:- ark:- |"
else
  echo "# 'apply-cmvn' not used,"
fi

# optionally add deltas
if [ ! -z "$delta_opts" ]; then
  feats_tr="$feats_tr add-deltas $delta_opts ark:- ark:- |"
  echo "# + 'add-deltas' with '$delta_opts'"
fi

# keep track of the config,
[ ! -z "$cmvn_opts" ] && echo "$cmvn_opts" >$dir/cmvn_opts
[ ! -z "$delta_opts" ] && echo "$delta_opts" >$dir/delta_opts
#

# get feature dim,
feat_dim=$(feat-to-dim "$feats_tr" -)
echo "# feature dim : $feat_dim (input of 'feature_transform')"

# Now we start building 'feature_transform' which goes right in front of a NN.
# The forwarding is computed on a GPU before the frame shuffling is applied.
#
# Same GPU is used both for 'feature_transform' and the NN training.
# So it has to be done by a single process (we are using exclusive mode).
# This also reduces the CPU-GPU uploads/downloads to minimum.

if [ ! -z "$feature_transform" ]; then
  echo "# importing 'feature_transform' from '$feature_transform'"
  tmp=$dir/imported_$(basename $feature_transform)
  cp $feature_transform $tmp; feature_transform=$tmp
else
  # Make default proto with splice,
  if [ ! -z $feature_transform_proto ]; then
    echo "# importing custom 'feature_transform_proto' from : $feature_transform_proto"
  else
    echo "+ default 'feature_transform_proto' with splice +/-$splice frames"
    feature_transform_proto=$dir/splice${splice}.proto
    echo "<Splice> <InputDim> $feat_dim <OutputDim> $(((2*splice+1)*feat_dim)) <BuildVector> -$splice:$splice </BuildVector>" >$feature_transform_proto
  fi

  # Initialize 'feature-transform' from a prototype,
  feature_transform=$dir/tr_$(basename $feature_transform_proto .proto).nnet
  nnet-initialize --binary=false $feature_transform_proto $feature_transform

  # Renormalize the MLP input to zero mean and unit variance,
  feature_transform_old=$feature_transform
  feature_transform=${feature_transform%.nnet}_cmvn-g.nnet
  echo "# compute normalization stats from 10k sentences"
  nnet-forward --print-args=true --use-gpu=yes $feature_transform_old \
    "$(echo $feats_tr | sed 's|train.scp|train.scp.10k|')" ark:- |\
    compute-cmvn-stats ark:- $dir/cmvn-g.stats
  echo "# + normalization of NN-input at '$feature_transform'"
  nnet-concat --print-args=false --binary=false $feature_transform_old \
    "cmvn-to-nnet $dir/cmvn-g.stats -|" $feature_transform
fi

if [ ! -z $ivector ]; then
  echo
  echo "# ADDING IVECTOR FEATURES"
  # The iVectors are concatenated 'as they are' directly to the input of the neural network,
  # To do this, we paste the features, and use <ParallelComponent> where the 1st component
  # contains the transform and 2nd network contains <Copy> component.

  echo "# getting dims,"
  dim_raw=$(feat-to-dim "$feats_tr" -)
  dim_raw_and_ivec=$(feat-to-dim "$feats_tr $ivector_append_tool ark:- '$ivector' ark:- |" -)
  dim_ivec=$((dim_raw_and_ivec - dim_raw))
  echo "# dims, feats-raw $dim_raw, ivectors $dim_ivec,"

  # Should we do something with 'feature_transform'?
  if [ ! -z $ivector_dim ]; then
    # No, the 'ivector_dim' comes from dir with 'feature_transform' with iVec forwarding,
    echo "# assuming we got '$feature_transform' with ivector forwarding,"
    [ $ivector_dim != $dim_ivec ] && \
    echo -n "Error, i-vector dimensionality mismatch!" && \
    echo " (expected $ivector_dim, got $dim_ivec in $ivector)" && exit 1
  else
    # Yes, adjust the transform to do ``iVec forwarding'',
    feature_transform_old=$feature_transform
    feature_transform=${feature_transform%.nnet}_ivec_copy.nnet
    echo "# setting up ivector forwarding into '$feature_transform',"
    dim_transformed=$(feat-to-dim "$feats_tr nnet-forward $feature_transform_old ark:- ark:- |" -)
    nnet-initialize --print-args=false <(echo "<Copy> <InputDim> $dim_ivec <OutputDim> $dim_ivec <BuildVector> 1:$dim_ivec </BuildVector>") $dir/tr_ivec_copy.nnet
    nnet-initialize --print-args=false <(echo "<ParallelComponent> <InputDim> $((dim_raw+dim_ivec)) <OutputDim> $((dim_transformed+dim_ivec)) <NestedNnetFilename> $feature_transform_old $dir/tr_ivec_copy.nnet </NestedNnetFilename>") $feature_transform
  fi
  echo $dim_ivec >$dir/ivector_dim # mark down the iVec dim!
  echo $ivector_append_tool >$dir/ivector_append_tool

  # pasting the iVecs to the feaures,
  echo "# + ivector input '$ivector'"
  feats_tr="$feats_tr $ivector_append_tool ark:- '$ivector' ark:- |"
fi

###### Show the final 'feature_transform' in the log,
echo
echo "### Showing the final 'feature_transform':"
nnet-info $feature_transform
echo "###"

###### MAKE LINK TO THE FINAL feature_transform, so the other scripts will find it ######
[ -f $dir/final.feature_transform ] && unlink $dir/final.feature_transform
(cd $dir; ln -s $(basename $feature_transform) final.feature_transform )
feature_transform=$dir/final.feature_transform


###### GET THE DIMENSIONS ######
num_fea=$(feat-to-dim --print-args=false "$feats_tr nnet-forward --use-gpu=no $feature_transform ark:- ark:- |" - 2>/dev/null)
num_hid=$hid_dim


###### PERFORM THE PRE-TRAINING ######
for depth in $(seq 1 $nn_depth); do
  echo
  echo "# PRE-TRAINING RBM LAYER $depth"
  RBM=$dir/$depth.rbm
  [ -f $RBM ] && echo "RBM '$RBM' already trained, skipping." && continue

  # The first RBM needs special treatment, because of Gussian input nodes,
  if [ "$depth" == "1" ]; then
    # This is usually Gaussian-Bernoulli RBM (not if CNN layers are part of input transform)
    # initialize,
    echo "# initializing '$RBM.init'"
    echo "<Rbm> <InputDim> $num_fea <OutputDim> $num_hid <VisibleType> $input_vis_type <HiddenType> bern <ParamStddev> $param_stddev_first" > $RBM.proto
    nnet-initialize $RBM.proto $RBM.init 2>$dir/log/nnet-initialize.$depth.log || exit 1
    # pre-train,
    num_iter=$rbm_iter; [ $input_vis_type == "gauss" ] && num_iter=$((2*rbm_iter)) # 2x more epochs for Gaussian input
    [ $input_vis_type == "bern" ] && rbm_lrate_low=$rbm_lrate # original lrate for Bernoulli input
    echo "# pretraining '$RBM' (input $input_vis_type, lrate $rbm_lrate_low, iters $num_iter)"
    rbm-train-cd1-frmshuff --learn-rate=$rbm_lrate_low --l2-penalty=$rbm_l2penalty \
      --num-iters=$num_iter --verbose=$verbose \
      --feature-transform=$feature_transform \
      $rbm_extra_opts \
      $RBM.init "$feats_tr" $RBM 2>$dir/log/rbm.$depth.log || exit 1
  else
    # This is Bernoulli-Bernoulli RBM,
    # cmvn stats for init,
    echo "# computing cmvn stats '$dir/$depth.cmvn' for RBM initialization"
    if [ ! -f $dir/$depth.cmvn ]; then
      nnet-forward --print-args=false --use-gpu=yes \
        "nnet-concat $feature_transform $dir/$((depth-1)).dbn - |" \
        "$(echo $feats_tr | sed 's|train.scp|train.scp.10k|')" ark:- | \
      compute-cmvn-stats --print-args=false ark:- - | \
      cmvn-to-nnet --print-args=false - $dir/$depth.cmvn || exit 1
    else
      echo "# compute-cmvn-stats already done, skipping."
    fi
    # initialize,
    echo "initializing '$RBM.init'"
    echo "<Rbm> <InputDim> $num_hid <OutputDim> $num_hid <VisibleType> bern <HiddenType> bern <ParamStddev> $param_stddev <VisibleBiasCmvnFilename> $dir/$depth.cmvn" > $RBM.proto
    nnet-initialize $RBM.proto $RBM.init 2>$dir/log/nnet-initialize.$depth.log || exit 1
    # pre-train,
    echo "pretraining '$RBM' (lrate $rbm_lrate, iters $rbm_iter)"
    rbm-train-cd1-frmshuff --learn-rate=$rbm_lrate --l2-penalty=$rbm_l2penalty \
      --num-iters=$rbm_iter --verbose=$verbose \
      --feature-transform="nnet-concat $feature_transform $dir/$((depth-1)).dbn - |" \
      $rbm_extra_opts \
      $RBM.init "$feats_tr" $RBM 2>$dir/log/rbm.$depth.log || exit 1
  fi

  # Create DBN stack,
  if [ "$depth" == "1" ]; then
    echo "# converting RBM to $dir/$depth.dbn"
    rbm-convert-to-nnet $RBM $dir/$depth.dbn
  else
    echo "# appending RBM to $dir/$depth.dbn"
    nnet-concat $dir/$((depth-1)).dbn "rbm-convert-to-nnet $RBM - |"  $dir/$depth.dbn
  fi

done

echo
echo "# REPORT"
echo "# RBM pre-training progress (line per-layer)"
grep progress $dir/log/rbm.*.log
echo

echo "Pre-training finished."

sleep 3
exit 0
