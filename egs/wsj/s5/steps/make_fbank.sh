#!/bin/bash 

# Copyright 2012  Karel Vesely, Daniel Povey
# Apache 2.0
# To be run from .. (one directory up from here)
# see ../run.sh for example

nj=4
cmd=run.pl
fbank_config=conf/fbank.conf

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "usage: make_fbank.sh [options] <tgt-data-dir> <src-data-dir> <log-dir> <path-to-fbankdir>";
   echo "options: "
   echo "  --fbank-config <config-file>                      # config passed to compute-fbank-feats "
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
srcdata=$2
logdir=$3
fbankdir=$4

# copy the dataset metadata from srcdata.
mkdir -p $data || exit 1;
cp $srcdata/* $data 2>/dev/null; rm $data/feats.scp;

# make $fbankdir an absolute pathname.
fbankdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $fbankdir ${PWD}`

# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $fbankdir || exit 1;
mkdir -p $logdir || exit 1;


srcscp=$srcdata/wav.scp
scp=$data/wav.scp

required="$srcscp $scp $fbank_config"

for f in $required; do
  if [ ! -f $f ]; then
    echo "make_fbank.sh: no such file $f"
    exit 1;
  fi
done

# note: in general, the double-parenthesis construct in bash "((" is "C-style
# syntax" where we can get rid of the $ for variable names, and omit spaces.
# The "for" loop in this style is a special construct.


if [ -f $data/segments ]; then
  echo "$0 [info]: segments file exists: using that."
  split_segments=""
  for ((n=1; n<=nj; n++)); do
    split_segments="$split_segments $logdir/segments.$n"
  done

  utils/split_scp.pl $data/segments $split_segments || exit 1;
  rm $logdir/.error 2>/dev/null

  $cmd JOB=1:$nj $logdir/make_fbank.JOB.log \
    extract-segments scp:$scp $logdir/segments.JOB ark:- \| \
    compute-fbank-feats --verbose=2 --config=$fbank_config ark:- \
    ark,scp:$fbankdir/raw_fbank_$name.JOB.ark,$fbankdir/raw_fbank_$name.JOB.scp \
     || exit 1;

else
  echo "$0: [info]: no segments file exists: assuming wav.scp indexed by utterance."
  split_scps=""
  for ((n=1; n<=nj; n++)); do
    split_scps="$split_scps $logdir/wav.$n.scp"
  done

  utils/split_scp.pl $scp $split_scps || exit 1;
 
  $cmd JOB=1:$nj $logdir/make_fbank.JOB.log \
    compute-fbank-feats  --verbose=2 --config=$fbank_config scp:$logdir/wav.JOB.scp \
      ark,scp:$fbankdir/raw_fbank_$name.JOB.ark,$fbankdir/raw_fbank_$name.JOB.scp \
      || exit 1;

fi


if [ -f $logdir/.error.$name ]; then
  echo "Error producing fbank features for $name:"
  tail $logdir/make_fbank.*.log
  exit 1;
fi

# concatenate the .scp files together.
for ((n=1; n<=nj; n++)); do
  cat $fbankdir/raw_fbank_$name.$n.scp >> $data/feats.scp || exit 1;
done > $data/feats.scp

rm $logdir/wav.*.scp  $logdir/segments.* 2>/dev/null

nf=`cat $data/feats.scp | wc -l` 
nu=`cat $data/utt2spk | wc -l` 
if [ $nf -ne $nu ]; then
  echo "It seems not all of the feature files were successfully ($nf != $nu);"
  echo "consider using utils/fix_data_dir.sh $data"
fi

echo "Succeeded creating FBANK features for $name"
