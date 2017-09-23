#!/bin/bash
#
# Apache 2.0.

# This script applies sliding window cmvn and removes silence frames.  This
# is performed on the raw features prior to generating examples for training
# the xvector system.

nj=40
cmd="run.pl"
stage=0
norm_vars=false
center=true
compress=true
cmn_window=300

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;
if [ $# != 3 ]; then
  echo "Usage: $0 <in-data-dir> <out-data-dir> <feat-dir>"
  echo "e.g.: $0 data/train data/train_no_sil exp/make_xvector_features"
  echo "Options: "
  echo "  --nj <nj>                                        # number of parallel jobs"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --norm-vars <true|false>                         # If true, normalize variances in the sliding window cmvn"
  exit 1;
fi

data_in=$1
data_out=$2
dir=$3

name=`basename $data_in`

for f in $data_in/feats.scp $data_in/vad.scp ; do
  [ ! -f $f ] && echo "$0: No such file $f" && exit 1;
done

# Set various variables.
mkdir -p $dir/log
mkdir -p $data_out
featdir=${PWD}/$dir

cp $data_in/utt2spk $data_out/utt2spk
cp $data_in/spk2utt $data_out/spk2utt
cp $data_in/wav.scp $data_out/wav.scp

sdata_in=$data_in/split$nj;
utils/split_data.sh $data_in $nj || exit 1;

$cmd JOB=1:$nj $dir/log/create_xvector_feats_${name}.JOB.log \
  apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=$cmn_window \
  scp:${sdata_in}/JOB/feats.scp ark:- \| \
  select-voiced-frames ark:- scp,s,cs:${sdata_in}/JOB/vad.scp ark:- \| \
  copy-feats --compress=$compress ark:- \
  ark,scp:$featdir/xvector_feats_${name}.JOB.ark,$featdir/xvector_feats_${name}.JOB.scp || exit 1;

for n in $(seq $nj); do
  cat $featdir/xvector_feats_${name}.$n.scp || exit 1;
done > ${data_out}/feats.scp || exit 1

echo "$0: Succeeded creating xvector features for $name"
