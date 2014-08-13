#!/bin/bash 

# Copyright 2013 The Shenzhen Key Laboratory of Intelligent Media and Speech,
#                PKU-HKUST Shenzhen Hong Kong Institution (Author: Wei Shi)
# Apache 2.0
# Combine filterbank and pitch features together 
# Note: This file is based on make_fbank.sh and make_pitch_kaldi.sh

# Begin configuration section.
nj=4
cmd=run.pl
fbank_config=conf/fbank.conf
pitch_config=conf/pitch.conf
pitch_postprocess_config=
paste_length_tolerance=2
compress=true
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "usage: make_fbank_pitch.sh [options] <data-dir> <log-dir> <path-to-fbank-pitch-dir>";
   echo "options: "
   echo "  --fbank-config             <config-file>             # config passed to compute-fbank-feats "
   echo "  --pitch_config             <pitch-config-file>       # config passed to compute-kaldi-pitch-feats "
   echo "  --pitch_postprocess_config <postprocess-config-file> # config passed to process-kaldi-pitch-feats "
   echo "  --paste_length_tolerance   <tolerance>               # length tolerance passed to paste-feats"
   echo "  --nj                       <nj>                      # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>)     # how to run jobs."
   exit 1;
fi

data=$1
logdir=$2
fbank_pitch_dir=$3


# make $fbank_pitch_dir an absolute pathname.
fbank_pitch_dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $fbank_pitch_dir ${PWD}`

# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $fbank_pitch_dir || exit 1;
mkdir -p $logdir || exit 1;

if [ -f $data/feats.scp ]; then
  mkdir -p $data/.backup
  echo "$0: moving $data/feats.scp to $data/.backup"
  mv $data/feats.scp $data/.backup
fi

scp=$data/wav.scp

required="$scp $fbank_config $pitch_config"

for f in $required; do
  if [ ! -f $f ]; then
    echo "make_fbank_pitch.sh: no such file $f"
    exit 1;
  fi
done

if [ ! -z "$pitch_postprocess_config" ]; then
	postprocess_config_opt="--config=$pitch_postprocess_config";
else
	postprocess_config_opt=
fi

utils/validate_data_dir.sh --no-text --no-feats $data || exit 1;

if [ -f $data/spk2warp ]; then
  echo "$0 [info]: using VTLN warp factors from $data/spk2warp"
  vtln_opts="--vtln-map=ark:$data/spk2warp --utt2spk=ark:$data/utt2spk"
elif [ -f $data/utt2warp ]; then
  echo "$0 [info]: using VTLN warp factors from $data/utt2warp"
  vtln_opts="--vtln-map=ark:$data/utt2warp"
fi

for n in $(seq $nj); do
  # the next command does nothing unless $fbank_pitch_dir/storage/ exists, see
  # utils/create_data_link.pl for more info.
  utils/create_data_link.pl $fbank_pitch_dir/raw_fbank_pitch_$name.$n.ark  
done

if [ -f $data/segments ]; then
  echo "$0 [info]: segments file exists: using that."
  split_segments=""
  for n in $(seq $nj); do
    split_segments="$split_segments $logdir/segments.$n"
  done

  utils/split_scp.pl $data/segments $split_segments || exit 1;
  rm $logdir/.error 2>/dev/null
   
  fbank_feats="ark:extract-segments scp,p:$scp $logdir/segments.JOB ark:- | compute-fbank-feats $vtln_opts --verbose=2 --config=$fbank_config ark:- ark:- |"
  pitch_feats="ark,s,cs:extract-segments scp,p:$scp $logdir/segments.JOB ark:- | compute-kaldi-pitch-feats --verbose=2 --config=$pitch_config ark:- ark:- | process-kaldi-pitch-feats $postprocess_config_opt ark:- ark:- |"

  $cmd JOB=1:$nj $logdir/make_fbank_pitch_${name}.JOB.log \
    paste-feats --length-tolerance=$paste_length_tolerance "$fbank_feats" "$pitch_feats" ark:- \| \
    copy-feats --compress=$compress ark:- \
      ark,scp:$fbank_pitch_dir/raw_fbank_pitch_$name.JOB.ark,$fbank_pitch_dir/raw_fbank_pitch_$name.JOB.scp \
     || exit 1;

else
  echo "$0: [info]: no segments file exists: assuming wav.scp indexed by utterance."
  split_scps=""
  for n in $(seq $nj); do
    split_scps="$split_scps $logdir/wav.$n.scp"
  done

  utils/split_scp.pl $scp $split_scps || exit 1;
  
  fbank_feats="ark:compute-fbank-feats $vtln_opts --verbose=2 --config=$fbank_config scp,p:$logdir/wav.JOB.scp ark:- |"
  pitch_feats="ark,s,cs:compute-kaldi-pitch-feats --verbose=2 --config=$pitch_config scp,p:$logdir/wav.JOB.scp ark:- | process-kaldi-pitch-feats $postprocess_config_opt ark:- ark:- |"
 
  $cmd JOB=1:$nj $logdir/make_fbank_pitch_${name}.JOB.log \
    paste-feats --length-tolerance=$paste_length_tolerance "$fbank_feats" "$pitch_feats" ark:- \| \
    copy-feats --compress=$compress ark:- \
      ark,scp:$fbank_pitch_dir/raw_fbank_pitch_$name.JOB.ark,$fbank_pitch_dir/raw_fbank_pitch_$name.JOB.scp \
      || exit 1;

fi


if [ -f $logdir/.error.$name ]; then
  echo "Error producing fbank & pitch features for $name:"
  tail $logdir/make_fbank_pitch_${name}.1.log
  exit 1;
fi

# concatenate the .scp files together.
for n in $(seq $nj); do
  cat $fbank_pitch_dir/raw_fbank_pitch_$name.$n.scp || exit 1;
done > $data/feats.scp

rm $logdir/wav.*.scp  $logdir/segments.* 2>/dev/null

nf=`cat $data/feats.scp | wc -l` 
nu=`cat $data/utt2spk | wc -l` 
if [ $nf -ne $nu ]; then
  echo "It seems not all of the feature files were successfully processed ($nf != $nu);"
  echo "consider using utils/fix_data_dir.sh $data"
fi

echo "Succeeded creating filterbank & pitch features for $name"
