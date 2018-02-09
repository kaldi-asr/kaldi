#!/bin/bash 

# Copyright 2016  CereProc LTD (Author: B. Potard)
# Apache 2.0
# To be run from .. (one directory up from here)
# see ../run.sh for example

# Begin configuration section.
nj=4
cmd=run.pl
bndap_config=conf/bndap.conf
frame_length=40
#f0_frame_length=25
compress=false
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "usage: make_bndap.sh [options] <data-dir> <log-dir> <path-to-mfccdir>";
   echo "options: "
   echo "  --bndap-config <config-file>                     # config passed to compute-kaldi-bndap-feats "
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
logdir=$2
bndapdir=$3


# make $bndapdir an absolute pathname.
bndapdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $bndapdir ${PWD}`


# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $bndapdir || exit 1;
mkdir -p $logdir || exit 1;

scp=$data/wav.scp
pitch_scp=$data/pitch_feats.scp

required="$scp $pitch_scp $bndap_config"

for f in $required; do
  if [ ! -f $f ]; then
    echo "make_bndap.sh: no such file $f"
    exit 1;
  fi
done
utils/validate_data_dir.sh --no-text --no-feats $data || exit 1;

for n in $(seq $nj); do
  # the next command does nothing unless $bndapdir/storage/ exists, see
  # utils/create_data_link.pl for more info.
  utils/create_data_link.pl $bndapdir/raw_bndap_$name.$n.ark  
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

#output="ark,t:| awk '(\$2 == \"[\"){print}(\$2 != \"[\"){if (\$NF == \"]\") {end=1; NF=NF-1}; for (i = 1; i <= NF; i++) if (\$i >= 0.0) printf \"0.0 \" ; else printf \"%f \", 20 * (\$i) / log(10); if (end) printf \"]\n\"; else printf \"\n\"; end = 0}'"
bndap_feats="ark,s,cs:compute-aperiodic-feats --frame-length=$frame_length --config=$bndap_config $in_feats scp:$pitch_scp ark:- |"


$cmd JOB=1:$nj $logdir/make_bndap.JOB.log \
    copy-feats --compress=$compress "$bndap_feats" \
      ark,scp:$bndapdir/raw_bndap_$name.JOB.ark,$bndapdir/raw_bndap_$name.JOB.scp \
     || exit 1;


if [ -f $logdir/.error.$name ]; then
  echo "Error producing bndap features for $name:"
  tail $logdir/make_bndap.*.log
  exit 1;
fi

# concatenate the .scp files together.
for n in $(seq $nj); do
  cat $bndapdir/raw_bndap_$name.$n.scp || exit 1;
done > $data/bndap_feats.scp

rm $logdir/wav_${name}.*.scp  $logdir/segments.* 2>/dev/null

nf=`cat $data/bndap_feats.scp | wc -l` 
nu=`cat $data/utt2spk | wc -l`
if [ $nf -ne $nu ]; then
  echo "It seems not all of the feature files were successfully processed ($nf != $nu);"
  echo "consider using utils/fix_data_dir.sh $data"
fi

if [ $nf -lt $[$nu - ($nu/20)] ]; then
  echo "Less than 95% of the features were successfully generated.  Probably a serious error."
  exit 1;
fi

echo "Succeeded creating Bndap features for $name"
