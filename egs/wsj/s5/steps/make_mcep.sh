#!/bin/bash 

# Copyright 2016  CereProc LTD (Author: B. Potard)
# Apache 2.0
# To be run from .. (one directory up from here)
# see ../run.sh for example

# Begin configuration section.
nj=4
cmd=run.pl
mcep_config=conf/mcep.conf
frame_length=30
compress=false
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "usage: make_mcep.sh [options] <data-dir> <log-dir> <path-to-mfccdir>";
   echo "options: "
   echo "  --mcep-config <config-file>                     # config passed to compute-kaldi-mcep-feats "
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
logdir=$2
mcepdir=$3


# make $mcepdir an absolute pathname.
mcepdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $mcepdir ${PWD}`


# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $mcepdir || exit 1;
mkdir -p $logdir || exit 1;

scp=$data/wav.scp
pitch_scp=$data/pitch_feats.scp

required="$scp $pitch_scp $mcep_config"

for f in $required; do
  if [ ! -f $f ]; then
    echo "make_mcep.sh: no such file $f"
    exit 1;
  fi
done
utils/validate_data_dir.sh --no-text --no-feats $data || exit 1;

for n in $(seq $nj); do
  # the next command does nothing unless $mcepdir/storage/ exists, see
  # utils/create_data_link.pl for more info.
  utils/create_data_link.pl $mcepdir/raw_mcep_$name.$n.ark  
done

# note: in general, the double-parenthesis construct in bash "((" is "C-style
# syntax" where we can get rid of the $ for variable names, and omit spaces.
# The "for" loop in this style is a special construct.

if [ -f $data/segments ]; then
    echo "$0 [info]: segments file exists: using that."
    split_segments=""
    for n in $(seq $nj); do
	split_segments="$split_segments $logdir/segments.$n"
    done

    utils/split_scp.pl $data/segments $split_segments || exit 1;
    rm $logdir/.error 2>/dev/null

    in_feats="ark,s,cs:extract-segments scp,p:$scp $logdir/segments.JOB ark:- |"
else
    echo "$0: [info]: no segments file exists: assuming wav.scp indexed by utterance."
    split_scps=""
    for n in $(seq $nj); do
	split_scps="$split_scps $logdir/wav_${name}.$n.scp"
    done

    utils/split_scp.pl $scp $split_scps || exit 1;

    in_feats="scp,p:$logdir/wav_${name}.JOB.scp"
fi

mcep_feats="ark,s,cs:compute-mcep-feats.sh --config $mcep_config --frame_length $frame_length $in_feats ark:- |"


$cmd JOB=1:$nj $logdir/make_mcep.JOB.log \
    copy-feats --compress=$compress "$mcep_feats" \
      ark,scp:$mcepdir/raw_mcep_$name.JOB.ark,$mcepdir/raw_mcep_$name.JOB.scp \
     || exit 1;


if [ -f $logdir/.error.$name ]; then
  echo "Error producing mcep features for $name:"
  tail $logdir/make_mcep.*.log
  exit 1;
fi

# concatenate the .scp files together.
for n in $(seq $nj); do
  cat $mcepdir/raw_mcep_$name.$n.scp || exit 1;
done > $data/mcep_feats.scp

rm $logdir/wav_${name}.*.scp  $logdir/segments.* 2>/dev/null

nf=`cat $data/mcep_feats.scp | wc -l` 
nu=`cat $data/utt2spk | wc -l`
if [ $nf -ne $nu ]; then
  echo "It seems not all of the feature files were successfully processed ($nf != $nu);"
  echo "consider using utils/fix_data_dir.sh $data"
fi

if [ $nf -lt $[$nu - ($nu/20)] ]; then
  echo "Less than 95% of the features were successfully generated.  Probably a serious error."
  exit 1;
fi

echo "Succeeded creating Mcep features for $name"
