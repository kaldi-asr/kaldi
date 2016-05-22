#!/bin/bash 

# Copyright 2015 Sri Harish Mallidi
# Apache 2.0
# To be run from .. (one directory up from here)
# see ../run.sh for example

# Begin configuration section.
bin_dir=         # binary dir of nnet-score-sent , 
                 # should end with / at the end

only_mse=true
nj=1
cmd=run.pl
# End configuration section.

# LABELS
labels=

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
   echo "usage: $0 [options] <tgt-data-dir> <src-data-dir> <labels> <nnet-dir> <log-dir> <abs-path-to-mse-dir>";
   echo "options: "
   echo "  --labels <labels>                                # use these labels as targerts to nnet"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

data=$1
srcdata=$2
nndir=$3
logdir=$4
msedir=$5

######## CONFIGURATION

# make $msedir an absolute pathname.
msedir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $msedir ${PWD}`

# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $data   || exit 1;
mkdir -p $msedir || exit 1;
mkdir -p $logdir || exit 1;

utils/copy_data_dir.sh $srcdata $data; rm $data/{feats,cmvn}.scp 2>/dev/null

srcscp=$srcdata/feats.scp

if [ ! -d $srcdata/split$nj -o $srcdata/split$nj -ot $srcdata/feats.scp ]; then
  utils/split_data.sh $srcdata $nj
fi
sdata=$srcdata/split$nj

required="$srcscp $nndir/final.nnet $sdata/1/feats.scp"
for f in $required; do
  if [ ! -f $f ]; then
    echo "$0: no such file $f"
    exit 1;
  fi
done


# PREPARE FEATURE EXTRACTION PIPELINE
# import config,
cmvn_opts=
delta_opts=
D=$nndir
[ -e $D/norm_vars ] && cmvn_opts="--norm-means=true --norm-vars=$(cat $D/norm_vars)" # Bwd-compatibility,
[ -e $D/cmvn_opts ] && cmvn_opts=$(cat $D/cmvn_opts)
[ -e $D/delta_order ] && delta_opts="--delta-order=$(cat $D/delta_order)" # Bwd-compatibility,
[ -e $D/delta_opts ] && delta_opts=$(cat $D/delta_opts)
#
# Create the feature stream,
feats="ark,s,cs:copy-feats scp:$sdata/JOB/feats.scp ark:- |"
# apply-cmvn (optional),
[ ! -z "$cmvn_opts" -a ! -f $sdata/1/cmvn.scp ] && echo "$0: Missing $sdata/1/cmvn.scp" && exit 1
[ ! -z "$cmvn_opts" ] && feats="$feats apply-cmvn $cmvn_opts --utt2spk=ark:$srcdata/utt2spk scp:$srcdata/cmvn.scp ark:- ark:- |"
# add-deltas (optional),
[ ! -z "$delta_opts" ] && feats="$feats add-deltas $delta_opts ark:- ark:- |"
#

###
### Prepare labels
echo 
echo "# PREPARING LABELS"
if [ ! -z "$labels" ]; then
  echo "Using targets '$labels' (by force)"
else
  feature_transform=$nndir/final.feature_transform
  [ ! -f $f ] && echo "$0: missing file $feature_transform" 
  labels="$feats nnet-forward $feature_transform ark:- ark:- | feat-to-post ark:- ark:- |"
  
fi
###
###

nnet=$nndir/final.nnet
# Run neural net MSE computation
if [ $only_mse == "true" ]; then
$cmd JOB=1:$nj $logdir/compute_mse.JOB.log \
  ${bin_dir}nnet-score-sent --use-gpu=no --objective-function=mse \
    --feature-transform=$feature_transform $nnet "$feats" "$labels" ark:- \| \
    select-feats 0 ark:- ark,scp:$msedir/mse_$name.JOB.ark,$msedir/mse_$name.JOB.scp || exit 1

else
$cmd JOB=1:$nj $logdir/compute_mse.JOB.log \
  ${bin_dir}nnet-score-sent --use-gpu=no --objective-function=mse \
    --feature-transform=$feature_transform $nnet "$feats" "$labels" \
    ark,scp:$msedir/mse_$name.JOB.ark,$msedir/mse_$name.JOB.scp || exit 1;
fi

N0=$(cat $srcdata/feats.scp | wc -l) 
N1=$(cat $msedir/mse_$name.*.scp | wc -l)
if [[ "$N0" != "$N1" ]]; then
  echo "Error producing mse for $name:"
  echo "Original feats : $N0  number of mse : $N1"
  exit 1;
fi

# concatenate the .scp files together.
for ((n=1; n<=nj; n++)); do
  cat $msedir/mse_$name.$n.scp >> $data/feats.scp
done


echo "Succeeded creating MSE for $name ($data)"

