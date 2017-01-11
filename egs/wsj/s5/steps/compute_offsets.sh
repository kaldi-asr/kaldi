#!/bin/bash

# Copyright 2016 Pegah Ghahremani
# Apache 2.0

# Computes random offsets per speaker w.r.t between-speakers's covariance matrix.
# It first computes CMN stats per speaker if not available and then computes
# between-speaker's covariance matrix.

# Begin configuration section.
cmd=run.pl
cmn_offset_scale=0.5 # offset scale used to scale the covariance matrix
num_cmn_offsets=4    # for chain model:4 , for xent model:3
compress=true
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -lt 1 ] || [ $# -gt 3 ];then
  echo "Usage: $0 [opts] <data-dir> [<log-dir> [<offset-dir>] ]"
  echo " e.g.: $0 data/train exp/make_offset/train offsets"
  echo "Note: <log-dir> defaults to <data-dir>/log, and <offset-dir> defaults to <data-dir>/data"
  exit 1;
fi

data=$1

if [ $# -ge 2 ]; then
  logdir=$2
else
  logdir=$data/log
fi

if [ $# -ge 3 ]; then
  offsetdir=$3
else
  offsetdir=$data/data
fi


# make $offsetdir an absolute pathname.
offsetdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $offsetdir ${PWD}`

# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $offsetdir || exit 1;
mkdir -p $logdir || exit 1;


required_files="$data/feats.scp $data/spk2utt $offsets_config"

for f in $required_files; do
  if [ ! -f $f ]; then
    echo "$0: no such file $f"
    exit 1;
  fi
done

if [ ! -f $data/cmvn.scp ]; then
  echo "$0: Compute cmvn stats per speaker for $data."
  steps/compute_cmvn_stats.sh $data || exit 1;
fi

if [ ! -f $data/offsets_per_spk.scp ]; then
  echo "$0: Computes per-spk offsets using precomputed cmvn stats as cmvn.scp."
  $cmd $logdir/compute_rand_offset.log \
    generate-random-cmn-offsets --cmn-offset-scale=$cmn_offset_scale \
    --num-cmn-offsets=$num_cmn_offsets \
    ark:$data/spk2utt scp:$data/cmvn.scp ark:- \| \
    copy-feats --compress=$compress \
      ark:- ark,scp:$offsetdir/offsets_per_spk.ark,$data/offsets_per_spk.scp || exit 1;
fi

if [ ! -f $data/offsets.scp ]; then
  echo "$0: Generates per-utterance offsets scp file offsets.scp by copying spk's offsets."
  utils/apply_map.pl -f 2 $data/offsets_per_spk.scp \
    <$data/utt2spk > $data/offsets.scp || exit 1;
fi

echo "Succeeded creating random offsets for $name"
