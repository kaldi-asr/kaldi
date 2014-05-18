#!/bin/bash 

# Copyright 2013 The Shenzhen Key Laboratory of Intelligent Media and Speech,
#                PKU-HKUST Shenzhen Hong Kong Institution (Author: Wei Shi)
# Apache 2.0
# Combine PLP and pitch features together 
# Note: This file is based on make_plp.sh and make_pitch_kaldi.sh

# Begin configuration section.
nj=4
cmd=run.pl
plp_config=conf/plp.conf
pitch_config=conf/pitch.conf
pitch_postprocess_config=
paste_length_tolerance=2
compress=true
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "usage: make_plp_pitch.sh [options] <data-dir> <log-dir> <path-to-plp-pitch-dir>";
   echo "options: "
   echo "  --plp-config               <config-file>             # config passed to compute-plp-feats "
   echo "  --pitch_config             <pitch-config-file>       # config passed to compute-kaldi-pitch-feats "
   echo "  --pitch_postprocess_config <postprocess-config-file> # config passed to process-kaldi-pitch-feats "
   echo "  --paste_length_tolerance   <tolerance>               # length tolerance passed to paste-feats"
   echo "  --nj                       <nj>                      # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>)     # how to run jobs."
   exit 1;
fi

data=$1
logdir=$2
plp_pitch_dir=$3


# make $plp_pitch_dir an absolute pathname.
plp_pitch_dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $plp_pitch_dir ${PWD}`

# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $plp_pitch_dir || exit 1;
mkdir -p $logdir || exit 1;

scp=$data/wav.scp

required="$scp $plp_config $pitch_config"

for f in $required; do
  if [ ! -f $f ]; then
    echo "make_plp_pitch.sh: no such file $f"
    exit 1;
  fi
done
utils/validate_data_dir.sh --no-text --no-feats $data || exit 1;

if [ ! -z "$pitch_postprocess_config" ]; then
	postprocess_config_opt="--config=$pitch_postprocess_config";
else
	postprocess_config_opt=
fi

# note: in general, the double-parenthesis construct in bash "((" is "C-style
# syntax" where we can get rid of the $ for variable names, and omit spaces.
# The "for" loop in this style is a special construct.

if [ -f $data/spk2warp ]; then
  echo "$0 [info]: using VTLN warp factors from $data/spk2warp"
  vtln_opts="--vtln-map=ark:$data/spk2warp --utt2spk=ark:$data/utt2spk"
fi

if [ -f $data/segments ]; then
  echo "$0 [info]: segments file exists: using that."
  split_segments=""
  for ((n=1; n<=nj; n++)); do
    split_segments="$split_segments $logdir/segments.$n"
  done

  utils/split_scp.pl $data/segments $split_segments || exit 1;
  rm $logdir/.error 2>/dev/null
   
  plp_feats="ark:extract-segments scp:$scp $logdir/segments.JOB ark:- | compute-plp-feats $vtln_opts --verbose=2 --config=$plp_config ark:- ark:- |"
  pitch_feats="ark,s,cs:extract-segments scp:$scp $logdir/segments.JOB ark:- | compute-kaldi-pitch-feats --verbose=2 --config=$pitch_config ark:- ark:- | process-kaldi-pitch-feats $postprocess_config_opt ark:- ark:- |"

  $cmd JOB=1:$nj $logdir/make_plp_pitch_${name}.JOB.log \
    paste-feats --length-tolerance=$paste_length_tolerance "$plp_feats" "$pitch_feats" ark:- \| \
    copy-feats --compress=$compress ark:- \
      ark,scp:$plp_pitch_dir/raw_plp_pitch_$name.JOB.ark,$plp_pitch_dir/raw_plp_pitch_$name.JOB.scp \
     || exit 1;

else
  echo "$0: [info]: no segments file exists: assuming wav.scp indexed by utterance."
  split_scps=""
  for ((n=1; n<=nj; n++)); do
    split_scps="$split_scps $logdir/wav_${name}.$n.scp"
  done

  utils/split_scp.pl $scp $split_scps || exit 1;
  

  plp_feats="ark:compute-plp-feats $vtln_opts --verbose=2 --config=$plp_config scp:$logdir/wav_${name}.JOB.scp ark:- |"
  pitch_feats="ark,s,cs:compute-kaldi-pitch-feats --verbose=2 --config=$pitch_config scp:$logdir/wav_${name}.JOB.scp ark:- | process-kaldi-pitch-feats $postprocess_config_opt ark:- ark:- |"
 
  $cmd JOB=1:$nj $logdir/make_plp_pitch_${name}.JOB.log \
    paste-feats --length-tolerance=$paste_length_tolerance "$plp_feats" "$pitch_feats" ark:- \| \
    copy-feats --compress=$compress ark:- \
      ark,scp:$plp_pitch_dir/raw_plp_pitch_$name.JOB.ark,$plp_pitch_dir/raw_plp_pitch_$name.JOB.scp \
      || exit 1;

fi


if [ -f $logdir/.error.$name ]; then
  echo "Error producing plp & pitch features for $name:"
  tail $logdir/make_plp_pitch_${name}.1.log
  exit 1;
fi

# concatenate the .scp files together.
for ((n=1; n<=nj; n++)); do
  cat $plp_pitch_dir/raw_plp_pitch_$name.$n.scp || exit 1;
done > $data/feats.scp

rm $logdir/wav_${name}.*.scp  $logdir/segments.* 2>/dev/null

nf=`cat $data/feats.scp | wc -l` 
nu=`cat $data/utt2spk | wc -l` 
if [ $nf -ne $nu ]; then
  echo "It seems not all of the feature files were successfully processed ($nf != $nu);"
  echo "consider using utils/fix_data_dir.sh $data"
fi

if [ $nf -lt $[$nu - ($nu/20)] ]; then
  echo "Less than 95% the features were successfully generated.  Probably a serious error."
  exit 1;
fi

echo "Succeeded creating PLP & Pitch features for $name"
