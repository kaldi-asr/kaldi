#!/usr/bin/env bash

# Copyright 2012-2015  Brno University of Technology (author: Karel Vesely),
#
# Apache 2.0
#
# This script dumps fMMI features in a new data directory, 
# which is later used for neural network training/testing.

# Begin configuration section.  
iter=final
nj=4
cmd=run.pl
ngselect=2; # Just use the 2 top Gaussians for fMMI/fMPE.  Should match train.
transform_dir=
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

set -euo pipefail

if [ $# != 5 ]; then
   echo "Usage: $0 [options] <tgt-data-dir> <src-data-dir> <gmm-dir> <log-dir> <fea-dir>"
   echo "e.g.: $0 data-fmmi/train data/train exp/tri5a_fmmi_b0.1 data-fmmi/train/_log data-fmmi/train/_data "
   echo ""
   echo "This script works on CMN + (delta+delta-delta | LDA+MLLT) features; it works out"
   echo "what type of features you used (assuming it's one of these two)"
   echo "You can also use fMLLR features-- you have to supply --transform-dir option."
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --iter <iter>                                    # Iteration of model to test."
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --transform-dir <transform-dir>                  # where to find fMLLR transforms."
   exit 1;
fi

data=$1
srcdata=$2
gmmdir=$3
logdir=$4
feadir=$5

sdata=$srcdata/split$nj;

# Get the config,
D=$gmmdir
[ -f $D/cmvn_opts ] && cmvn_opts=$(cat $D/cmvn_opts) || cmvn_opts=
[ -f $D/splice_opts ] && splice_opts=$(cat $D/splice_opts) || splice_opts=

mkdir -p $data $logdir $feadir
[[ -d $sdata && $srcdata/feats.scp -ot $sdata ]] || split_data.sh $srcdata $nj || exit 1;

for f in $sdata/1/feats.scp $sdata/1/cmvn.scp $gmmdir/$iter.fmpe; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

if [ -f $gmmdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type";

case $feat_type in
  delta) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $gmmdir/final.mat ark:- ark:- |";;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac

if [ ! -z "$transform_dir" ]; then # add transforms to features...
  echo "Using fMLLR transforms from $transform_dir"
  [ ! -f $transform_dir/trans.1 ] && echo "Expected $transform_dir/trans.1 to exist."
  [ "`cat $transform_dir/num_jobs`" -ne $nj ] && \
     echo "Mismatch in number of jobs with $transform_dir";
  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$transform_dir/trans.JOB ark:- ark:- |"
fi

# Get Gaussian selection info.
$cmd JOB=1:$nj $logdir/gselect.JOB.log \
  gmm-gselect --n=$ngselect $gmmdir/$iter.fmpe "$feats" \
  "ark:|gzip -c >$feadir/gselect.JOB.gz" || exit 1;

# prepare the dir
cp $srcdata/* $data 2>/dev/null; rm $data/{feats,cmvn}.scp;

# make $bnfeadir an absolute pathname.
feadir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $feadir ${PWD}`

# forward the feats
$cmd JOB=1:$nj $logdir/make_fmmi_feats.JOB.log \
  fmpe-apply-transform $gmmdir/$iter.fmpe "$feats" "ark,s,cs:gunzip -c $feadir/gselect.JOB.gz|"  \
  ark,scp:$feadir/feats_fmmi.JOB.ark,$feadir/feats_fmmi.JOB.scp || exit 1;
   
# merge the feats to single SCP
for n in $(seq 1 $nj); do
  cat $feadir/feats_fmmi.$n.scp 
done > $data/feats.scp

echo "$0 finished... $srcdata -> $data ($gmmdir)"

exit 0;
