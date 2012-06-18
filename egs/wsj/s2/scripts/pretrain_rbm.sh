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
    *)
      break;
      ;;
  esac
done


if [ $# != 3 ]; then
   echo "Usage: steps/pretrain_rbm.sh <rbm-in> <feature-pipeline> <rbm-out>"
   echo " e.g.: steps/pretrain_rbm.sh rbm.init \"scp:features.scp\" rbm.trained"
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

rbm_in=$1
features=$2
rbm_out=$3

if [ ! -f $rbm_in ]; then
  echo "Error: initial rbm '$rbm_in' does not exist"
  exit 1;
fi



######## CONFIGURATION
TRAIN_TOOL="rbm-train-cd1-frmshuff --bunchsize=128 "

#global parameters
echo iters ${iters:=1}   #by default one iteration
echo lrate ${lrate:=0.001} 
echo momentum ${momentum:=0.5}
echo l2penalty ${l2penalty:=0.0002}


dir=$(dirname $rbm_out)
logdir=$dir/../log
mkdir -p $dir $logdir

base=$(basename $rbm_out)

#PRE-TRAIN
for i in $(seq 1 $iters); do
  $TRAIN_TOOL --learn-rate=$lrate --momentum=$momentum --l2-penalty=$l2penalty \
    ${transf:+ --feature-transform=$transf} \
    $rbm_in "$features" $rbm_out.iter$i 2>$logdir/$base.iter$i.log || exit 1
  rbm_in=$rbm_out.iter$i
done

#make full path
[[ ${rbm_out:0:1} != "/" && ${rbm_out:0:1} != "~" ]] && rbm_out=$PWD/$rbm_out

#link the final rbm
ln -s $rbm_out.iter$i $rbm_out

echo "$0 finished ok"

