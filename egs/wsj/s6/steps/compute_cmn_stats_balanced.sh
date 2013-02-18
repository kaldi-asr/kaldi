#!/bin/bash 

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0
# To be run from .. (one directory up from here)
# see ../run.sh for example

# Compute cepstral mean and variance statistics per speaker.  
# This version of the script takes a directory produced by
# "train_cmn_models.sh" that contains certain information
# that gives us some per-frame probabilities of speech and silence,
# which we can use to reweight to a particular balance of
# speech vs. silence before doing CMVN.

# Begin configuration section.
nj=4
cmd=run.pl
silence_proportion=0.15
count_cutoff=200  # this is kind of a minor option, so I don't mention it in the usage
                  # message.
stage=0
cleanup=true
# End configuration section

echo "$0 $@"  # Print the command line for logging

. utils/parse_options.sh

if [ $# != 4 ]; then
   echo "Usage: compute_cmn_stats_balanced.sh <data-dir> <cmn-model-dir> <cmn-data-dir> <cmn-log-dir> ";
   echo "Note: <cmn-model-dir> would be produced by train_cmn_models.sh"
   echo "E.g.: compute_cmn_stats.sh data/train exp/tri2a_cmn mfcc exp/tri2a_cmn_train"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file|none>                      # config containing options"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --silence-proportion <proportion|0.15>           # proportion of silence that we try to reweight stats to."
   echo "  --cleanup <true or false|true>                   # if true, clean up temporary files."
   exit 1;
fi

if [ -f path.sh ]; then . ./path.sh; fi

data=$1
speechsildir=$2
cmndir=$3
dir=$4 # for logs.

# make $cmndir an absolute pathname.
cmndir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $cmndir ${PWD}`

[ ! -d $cmndir ] && echo "No such directory $cmndir" && exit 1;

# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $cmndir || exit 1;
mkdir -p $dir/log || exit 1;

echo $nj > $dir/num_jobs
sdata=$data/split$nj;
split_data.sh $data $nj || exit 1;
splice_opts=`cat $speechsildir/splice_opts` || exit 1;

required="$data/feats.scp $speechsildir/silence.ubm $speechsildir/nonsilence.ubm $speechsildir/silence.cmvn $speechsildir/nonsilence.cmvn"

for f in $required; do
  if [ ! -f $f ]; then
    echo "make_cmn.sh: no such file $f"
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
  $cmd JOB=1:$nj $dir/log/cmn.JOB.log \
    compute-cmn-stats-balanced --verbose=2 --count-cutoff=$count_cutoff \
     --silence-proportion=$silence_proportion  --spk2utt=ark:$sdata/JOB/spk2utt \
     $speechsildir/silence.cmvn $speechsildir/nonsilence.cmvn \
     scp:$sdata/JOB/feats.scp "ark,s,cs:gunzip -c $dir/weights.JOB.gz|" \
     ark,scp:$cmndir/cmn.JOB.$name.ark,$cmndir/cmn.JOB.$name.scp  || exit 1;
fi

for j in `seq $nj`; do
  cat $cmndir/cmn.$j.$name.scp
done > $data/cmvn.scp

$cleanup && rm $dir/weights.*.gz

nc=`cat $data/cmvn.scp | wc -l` 
nu=`cat $data/spk2utt | wc -l` 
if [ $nc -ne $nu ]; then
  echo "Error: it seems not all of the speakers got cmn stats ($nc != $nu);"
  exit 1;
fi

echo "Succeeded creating CMN stats for $name"
