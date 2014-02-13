#!/bin/bash

# Copyright 2013-2014  Johns Hopkins University (authors: Jan Trmal, Guoguo Chen, Dan Povey)

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


# Script for system combination using minimum Bayes risk decoding.
# This calls lattice-combine to create a union of lattices that have been 
# normalized by removing the total forward cost from them. The resulting lattice
# is used as input to lattice-mbr-decode. This should not be put in steps/ or 
# utils/ since the scores on the combined lattice must not be scaled.

# begin configuration section.
cmd=run.pl
stage=0
# Combination power
power=0.4
# Ntrue-scale
Ntrue_scale=1.1
min_lmw=8
max_lmw=12
skip_scoring=false

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

datadir=$1
lang=$2
odir=${@: -1}  # last argument to the script
shift 2;
decode_dirs=( $@ )  # read the remaining arguments into an array
unset decode_dirs[${#decode_dirs[@]}-1]  # 'pop' the last argument which is odir
num_sys=${#decode_dirs[@]}  # number of systems to combine


for f in $datadir/kws/ecf.xml $datadir/kws/kwlist.xml ; do
  [ ! -f $f ] && echo "$0: file $f does not exist" && exit 1;
done
ecf=$datadir/kws/ecf.xml
kwlist=$datadir/kws/kwlist.xml

# Duration
duration=`head -1 $ecf |\
    grep -o -E "duration=\"[0-9]*[    \.]*[0-9]*\"" |\
    perl -e 'while($m=<>) {$m=~s/.*\"([0-9.]+)\".*/\1/; print $m/2;}'`

mkdir -p $odir/log

weight=$(perl -e "print (1.0/$num_sys);")
systems=""
for i in `seq 0 $[num_sys-1]`; do
  if [ -f ${decode_dirs[$i]} ] ; then
    systems+="$weight ${decode_dirs[$i]} "
  else
    kwsfile=${decode_dirs[$i]}/kwslist.unnormalized.xml
    [ ! -f ${kwsfile} ] && echo "The file ${kwsfile} does not exist!" && exit 1
    systems+="$weight ${kwsfile} "
  fi
done

# Combination of the weighted sum and power rule
$cmd PWR=1:9 $odir/log/combine_kws.PWR.log \
  mkdir -p $odir/kws_PWR/ '&&' \
  local/naive_comb.pl --method=2 --power=0.PWR \
    $systems $odir/kws_PWR/kwslist.unnormalized.xml || exit 1

$cmd PWR=1:9 $odir/log/postprocess_kws.PWR.log \
  utils/kwslist_post_process.pl --duration=${duration} --digits=3 \
    --normalize=true --Ntrue-scale=${Ntrue_scale} \
    $odir/kws_PWR/kwslist.unnormalized.xml \
    $odir/kws_PWR/kwslist.xml || exit 1

if ! $skip_scoring ; then
$cmd PWR=1:9 $odir/log/score_kws.PWR.log \
  local/kws_score.sh $datadir $odir/kws_PWR || exit 1
fi
