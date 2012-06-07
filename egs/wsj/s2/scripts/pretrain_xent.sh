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
# Rbm pre-training script
#

while [ 1 ]; do
  case $1 in
    --feature-transform)
      shift; transf=$1; shift;
      ;;
    --iters)
      shift; iters=$1; shift;
      ;;
    --lrate)
      shift; lrate=$1; shift;
      ;;
    --momentum)
      shift; momentum=$1; shift;
      ;;
    --l2-penalty)
      shift; l2penalty=$1; shift;
      ;;
    --bunchsize)
      shift; bunchsize=$1; shift;
      ;;
    *)
      break;
      ;;
  esac
done


if [ $# != 5 ]; then
   echo "Usage: steps/pretrain_xent.sh <nnet-in> <features-train> <features-cv> <labels> <nnet-out>"
   echo " e.g.: steps/pretrain_xent.sh nnet.init \"scp:train.scp\" \"scp:cv.scp\" \"ark:labels.pdf\" nnet.trained"
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

nnet_in=$1
feats_tr=$2
feats_cv=$3
labels=$4
nnet_out=$5

if [ ! -f $nnet_in ]; then
  echo "Error: initial rbm '$nnet_in' does not exist"
  exit 1;
fi



######## CONFIGURATION
TRAIN_TOOL="nnet-train-xent-hardlab-frmshuff"

#global parameters
echo iters ${iters:=1}   #by default one iteration
echo lrate ${lrate:=0.001} 
echo momentum ${momentum:=0.5}
echo l2penalty ${l2penalty:=0.0002}
echo bunchsize: ${bunchsize:=256} #size of the Stochastic-GD update block


dir=$nnet_out.d
mkdir -p $dir/{nnet,log}

mlp_init=$nnet_in
max_iters=$iters
start_halving_inc=0.5
end_halving_inc=0.5
halving_factor=0.5


###### TRAIN ######
echo "Starting training:"
source scripts/train_nnet_scheduler.sh
echo "Training finished."
if [ "" == "$mlp_final" ]; then
  echo "No final network returned!"
else
  [[ ${mlp_final:0:1} != "/" && ${mlp_final:0:1} != "~" ]] && mlp_final=$PWD/$mlp_final
  ln -s $mlp_final $nnet_out
  echo "Final network $mlp_final"
fi


