#!/usr/bin/env bash

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


# Script for system combination of the KWS posting lists

# begin configuration section.
cmd=run.pl
stage=0
# Ntrue-scale
ntrue_scale=1.1
min_lmw=8
max_lmw=12
extraid=
skip_scoring=false
optimize_weights=false
#end of configuration section

help_message="Usage: $(basename $0) [options] <data-dir> <lang-dir|graph-dir> <decode-dir1>[:lmwt-bias] <decode-dir2>[:lmwt-bias] [<decode-dir3>[:lmwt-bias] ... ] <out-dir>
E.g.: $(basename $0) data/dev10h.pem data/lang exp/tri6_nnet/decode_dev10h.pem/kws_10/  exp/tri6_nnet/decode_dev10h.pem/oov_kws_10/    exp/combine/dev10hx.pem
"
[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -lt 5 ]; then
  printf "$help_message\n";
  exit 1;
fi

datadir=$1
lang=$2
odir=${@: -1}  # last argument to the script
shift 2;
decode_dirs=( $@ )  # read the remaining arguments into an array
unset decode_dirs[${#decode_dirs[@]}-1]  # 'pop' the last argument which is odir
num_sys=${#decode_dirs[@]}  # number of systems to combine


if [ -z "$extraid" ] ; then
  kwsdatadir=$datadir/kws
  kwsoutputdir="$odir/kws"
else
  kwsdatadir=$datadir/${extraid}_kws
  kwsoutputdir="$odir/${extraid}_kws"
fi

for f in $kwsdatadir/ecf.xml $kwsdatadir/kwlist.xml ; do
  [ ! -f $f ] && echo "$0: file $f does not exist" && exit 1;
done
ecf=$kwsdatadir/ecf.xml
kwlist=$kwsdatadir/kwlist.xml

# Duration
duration=`head -1 $ecf |\
    grep -o -E "duration=\"[0-9]*[    \.]*[0-9]*\"" |\
    perl -e 'while($m=<>) {$m=~s/.*\"([0-9.]+)\".*/\1/; print $m/2;}'`

mkdir -p $kwsoutputdir/log

total_sum=0
for i in `seq 0 $[num_sys-1]`; do
  decode_dir=${decode_dirs[$i]}
  offset=`echo $decode_dir | cut -d: -s -f2` # add this to the lm-weight.
  [ -z "$offset" ] && offset=1
  total_sum=$(($total_sum+$offset))
done

systems=""
for i in `seq 0 $[num_sys-1]`; do
  decode_dir=${decode_dirs[$i]}
  offset=`echo $decode_dir | cut -d: -s -f2` # add this to the lm-weight.
  decode_dir=`echo $decode_dir | cut -d: -f1`
  [ -z "$offset" ] && offset=1

  weight=$(perl -e "print ($offset/$total_sum);")
  if [ -f $decode_dir ] ; then
    systems+="$weight $decode_dir "
  else
    kwsfile=$decode_dir/kwslist.unnormalized.xml
    [ ! -f ${kwsfile} ] && echo "The file ${kwsfile} does not exist!" && exit 1
    systems+="$weight ${kwsfile} "
  fi
done

echo $systems

# Combination of the weighted sum and power rule
$cmd PWR=1:9 $kwsoutputdir/log/combine_kws.PWR.log \
  mkdir -p ${kwsoutputdir}_PWR '&&' \
  local/naive_comb.pl --method=2 --power=0.PWR \
    $systems ${kwsoutputdir}_PWR/kwslist.unnormalized.xml || exit 1

$cmd PWR=1:9 $kwsoutputdir/log/postprocess_kws.PWR.log \
  utils/kwslist_post_process.pl --duration=${duration} --digits=3 \
    --normalize=true --Ntrue-scale=${ntrue_scale} \
    ${kwsoutputdir}_PWR/kwslist.unnormalized.xml \
    ${kwsoutputdir}_PWR/kwslist.xml || exit 1

echo "Scoring..."
if ! $skip_scoring ; then
$cmd PWR=1:9 $kwsoutputdir/log/score_kws.PWR.log \
  local/kws_score.sh --extraid "${extraid}" $datadir ${kwsoutputdir}_PWR || exit 1
fi


