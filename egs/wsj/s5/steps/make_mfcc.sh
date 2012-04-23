#!/bin/bash 

# Copyright 2012  Daniel Povey
# Apache 2.0
# To be run from .. (one directory up from here)
# see ../run.sh for example

nj=4
cmd=utils/run.pl
config=conf/mfcc.conf

for x in 1 2; do
  if [ $1 == "--num-jobs" ]; then
     nj=$2
     shift 2
  fi
  if [ $1 == "--cmd" ]; then
     cmd=$2
     shift 2
  fi  
  if [ $1 == "--config" ]; then
     config=$2
     shift 2
  fi  
done

if [ $# != 3 ]; then
   echo "usage: make_mfcc.sh [options] <data-dir> <log-dir> <path-to-mfccdir>";
   echo "options: [--config <config-file>] [--num-jobs <num-jobs>] [--cmd utils/run.pl|utils/queue.pl]"
   exit 1;
fi

if [ -f path.sh ]; then . ./path.sh; fi

data=$1
logdir=$2
mfccdir=$3

# make $mfccdir an absolute pathname.
mfccdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $mfccdir ${PWD}`

# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $mfccdir || exit 1;
mkdir -p $logdir || exit 1;

scp=$data/wav.scp

required="$scp $config"

for f in $required; do
  if [ ! -f $f ]; then
    echo "make_mfcc.sh: no such file $f"
    exit 1;
  fi
done

# note: in general, the double-parenthesis construct in bash "((" is "C-style
# syntax" where we can get rid of the $ for variable names, and omit spaces.
# The "for" loop in this style is a special construct.


if [ -f $data/segments ]; then
  echo "Segments file exists: using that."
  split_segments=""
  for ((n=1; n<=nj; n++)); do
    split_segments="$split_segments $logdir/segments.$n"
  done

  utils/split_scp.pl $segments $split_segments || exit 1;
  rm $logdir/.error 2>/dev/null

  $cmd JOB=1:$nj $logdir/make_mfcc.JOB.log \
    extract-segments scp:$scp $logdir/segments.JOB ark:- \| \
    compute-mfcc-feats --verbose=2 --config=$config ark:- \
    ark,scp:$mfccdir/raw_mfcc_$name.JOB.ark,$mfccdir/raw_mfcc_$name.JOB.scp \
     || exit 1;

else
  echo "make_mfcc.sh: no segments file exists: assuming wav.scp indexed by utterance."
  split_scps=""
  for ((n=1; n<=nj; n++)); do
    split_scps="$split_scps $logdir/wav.$n.scp"
  done

  utils/split_scp.pl $scp $split_scps || exit 1;
 
  $cmd JOB=1:$nj $logdir/make_mfcc.JOB.log \
    compute-mfcc-feats  --verbose=2 --config=$config scp:$logdir/wav.JOB.scp \
      ark,scp:$mfccdir/raw_mfcc_$name.JOB.ark,$mfccdir/raw_mfcc_$name.JOB.scp \
      || exit 1;

fi


if [ -f $logdir/.error.$name ]; then
  echo "Error producing mfcc features for $name:"
  tail $logdir/make_mfcc.*.log
  exit 1;
fi

# concatenate the .scp files together.
for ((n=1; n<=nj; n++)); do
  cat $mfccdir/raw_mfcc_$name.$n.scp >> $data/feats.scp || exit 1;
done > $data/feats.scp

rm $logdir/wav.*.scp  $logdir/segments.* 2>/dev/null

nf=`cat $data/feats.scp | wc -l` 
nu=`cat $data/utt2spk | wc -l` 
if [ $nf -ne $nu ]; then
  echo "It seems not all of the feature files were successfully ($nf != $nu);"
  echo "consider using utils/fix_data_dir.sh $data"
fi

echo "Succeeded creating MFCC features for $name"
