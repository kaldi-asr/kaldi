#!/bin/bash 

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0
# To be run from .. (one directory up from here)
# see ../run.sh for example

# Compute cepstral mean and variance statistics per speaker.  
# This version of the script takes a directory produced by
# "train_speechsil_dubms.sh" that contains certain information
# that gives us some per-frame probabilities of speech and silence,
# which we can use to reweight to a particular balance of
# speech vs. silence before doing CMVN.

# Begin configuration section.
nj=4
cmd=run.pl
max_silence_proportion=0.2
stage=0
cleanup=true
# End configuration section

echo "$0 $@"  # Print the command line for logging

. utils/parse_options.sh

if [ $# != 4 ]; then
   echo "Usage: compute_cmvn_stats_balanced.sh <data-dir> <speechsil-dir> <cmvn-data-dir> <cmvn-log-dir> ";
   echo "Note: <speechsil-dir> would be produced by train_speechsil_dubms.sh"
   echo "E.g.: compute_cmvn_stats.sh data/train_bal exp/tri2b_speechsil /mnt/data/swbd/cmvn exp/tri2b_cmvn"
   echo "  See script for options"
   exit 1;
fi

if [ -f path.sh ]; then . ./path.sh; fi

data=$1
speechsildir=$2
cmvndir=$3
dir=$4 # for logs.


# make $cmvndir an absolute pathname.
cmvndir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $cmvndir ${PWD}`

[ ! -d $cmvndir ] && echo "No such directory $cmvndir" && exit 1;

# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $cmvndir || exit 1;
mkdir -p $dir/log || exit 1;

echo $nj > $dir/num_jobs
sdata=$data/split$nj;
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
splice_opts=`cat $speechsildir/splice_opts` || exit 1;

required="$data/feats.scp $speechsildir/silence.ubm $speechsildir/nonsilence.ubm $speechsildir/silence.cmvn $speechsildir/nonsilence.cmvn"

for f in $required; do
  if [ ! -f $f ]; then
    echo "make_cmvn.sh: no such file $f"
    exit 1;
  fi
done

feats="ark,s,cs:splice-feats $splice_opts scp:$sdata/JOB/feats.scp ark:- | transform-feats $speechsildir/final.mat ark:- ark:- |"


if [ $stage -le 0 ]; then
# First get the silence probabilities for each frame.
  $cmd JOB=1:$nj $dir/log/nonsil_probs.JOB.log \
    get-silence-probs --write-nonsil-probs=true \
    "$feats gmm-global-get-frame-likes $speechsildir/silence.ubm ark:- ark:- |" \
    "$feats gmm-global-get-frame-likes $speechsildir/nonsilence.ubm ark:- ark:- |" \
    "ark,t:|gzip -c >$dir/weights.JOB.gz" || exit 1;
fi  


if [ $stage -le 1 ]; then
  $cmd JOB=1:$nj $dir/log/cmvn.JOB.log \
    compute-cmvn-stats-balanced --spk2utt=ark:$sdata/JOB/spk2utt \
     $speechsildir/silence.cmvn $speechsildir/nonsilence.cmvn \
     scp:$sdata/JOB/feats.scp "ark,s,cs:gunzip -c $dir/weights.JOB.gz|" \
     ark,scp:$cmvndir/cmvn.JOB.$name.ark,$cmvndir/cmvn.JOB.$name.scp  || exit 1;
fi

for j in `seq $nj`; do
  cat $cmvndir/cmvn.$j.$name.scp
done > $data/cmvn.scp

$cleanup && rm $dir/weights.*.gz

nc=`cat $data/cmvn.scp | wc -l` 
nu=`cat $data/spk2utt | wc -l` 
if [ $nc -ne $nu ]; then
  echo "Error: it seems not all of the speakers got cmvn stats ($nc != $nu);"
  exit 1;
fi

echo "Succeeded creating CMVN stats for $name"
