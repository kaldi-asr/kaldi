#!/bin/bash

# Copyright 2014  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# Begin configuration.
stage=0 # This allows restarting after partway, when something when wrong.
feature_type=mfcc
add_pitch=false
mfcc_config=conf/mfcc.conf # you can override any of these you need to override.
plp_config=conf/plp.conf
fbank_config=conf/fbank.conf 
pitch_config=conf/pitch.conf
pitch_process_config=conf/pitch_process.conf
per_utt_cmvn=false # If true, apply online CMVN normalization per utterance
                   # rather than per speaker.

# Below are some options that affect the iVectors, and should probably
# match those used in extract_ivectors_online.sh.
num_gselect=5 # Gaussian-selection using diagonal model: number of Gaussians to select
posterior_scale=0.1 # Scale on the acoustic posteriors, intended to account for
                    # inter-frame correlations.
min_post=0.025 # Minimum posterior to use (posteriors below this are pruned out)
               # caution: you should use the same value in the online-estimation
               # code.
# End configuration.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

if [ $# -ne 4 ] && [ $# -ne 3 ]; then
   echo "Usage: $0 [options] <lang-dir> [<ivector-extractor-dir>] <nnet-dir> <output-dir>"
   echo "e.g.: $0 data/lang exp/nnet2_online/extractor exp/nnet2_online/nnet exp/nnet2_online/nnet_online"
   echo "main options (for others, see top of script file)"
   echo "  --feature-type <mfcc|plp>                        # Type of the base features; "
   echo "                                                   # important to generate the correct"
   echo "                                                   # configs in <output-dir>/conf/"
   echo "  --add-pitch <true|false>                         # Append pitch features to cmvn"
   echo "                                                   # (default: false)"
   echo "  --per-utt-cmvn <true|false>                      # Apply online CMVN per utt, not"
   echo "                                                   # per speaker (default: false)"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --config <config-file>                           # config containing options"
   echo "  --stage <stage>                                  # stage to do partial re-run from."
   exit 1;
fi


if [ $# -eq 4 ]; then
  lang=$1
  iedir=$2
  srcdir=$3
  dir=$4
else
  [ $# -eq 3 ] || exit 1;
  lang=$1
  iedir=
  srcdir=$2
  dir=$3
fi

for f in $lang/phones.txt $srcdir/final.mdl; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done
if [ ! -z "$iedir" ]; then
  for f in final.{mat,ie,dubm} splice_opts global_cmvn.stats online_cmvn.conf; do
    [ ! -f $iedir/$f ] && echo "$0: no such file $iedir/$f" && exit 1;
  done
fi

mkdir -p $dir/conf


cp $srcdir/final.mdl $dir/ || exit 1;
if [ ! -z "$iedir" ]; then
  mkdir -p $dir/ivector_extractor/
  cp $iedir/final.{mat,ie,dubm} $iedir/global_cmvn.stats $dir/ivector_extractor/ || exit 1;
fi


mkdir -p $dir/conf
rm $dir/{plp,mfcc,fbank}.conf 2>/dev/null
echo "$0: preparing configuration files in $dir/conf"

if [ -f $dir/conf/online_nnet2_decoding.conf ]; then
  echo "$0: moving $dir/conf/online_nnet2_decoding.conf to $dir/conf/online_nnet2_decoding.conf.bak"
  mv $dir/conf/online_nnet2_decoding.conf $dir/conf/online_nnet2_decoding.conf.bak
fi

conf=$dir/conf/online_nnet2_decoding.conf
echo -n >$conf

echo "--feature-type=$feature_type" >>$conf

case "$feature_type" in
  mfcc)
    echo "--mfcc-config=$dir/conf/mfcc.conf" >>$conf
    cp $mfcc_config $dir/conf/mfcc.conf || exit 1;;
  plp)
    echo "--plp-config=$dir/conf/plp.conf" >>$conf
    cp $plp_config $dir/conf/plp.conf || exit 1;;
  fbank)
    echo "--fbank-config=$dir/conf/fbank.conf" >>$conf
    cp $fbank_config $dir/conf/fbank.conf || exit 1;;
  *)
    echo "Unknown feature type $feature_type"
esac



if [ ! -z "$iedir" ]; then
  echo -n >$ieconf
  echo "--ivector-extraction-config=$ieconf" >>$conf
  ieconf=$dir/conf/ivector_extractor.conf
  cp $iedir/online_cmvn.conf $dir/conf/online_cmvn.conf || exit 1;
  for x in $(cat $iedir/splice_opts); do echo "$x"; done > $dir/conf/splice.conf
  echo "--splice-config=$dir/conf/splice.conf" >>$ieconf
  echo "--cmvn-config=$dir/conf/online_cmvn.conf" >>$ieconf
  echo "--lda-matrix=$dir/ivector_extractor/final.mat" >>$ieconf
  echo "--global-cmvn-stats=$dir/ivector_extractor/global_cmvn.stats" >>$ieconf
  echo "--diag-ubm=$dir/ivector_extractor/final.dubm" >>$ieconf
  echo "--ivector-extractor=$dir/ivector_extractor/final.ie" >>$ieconf
  echo "--num-gselect=5"  >>$ieconf
  echo "--min-post=0.025" >>$ieconf
  echo "--posterior-scale=0.1" >>$ieconf # this is currently the default in the scripts.
  echo "--use-most-recent-ivector=true" >>$ieconf # probably makes very little difference.
  echo "--max-remembered-frames=1000" >>$ieconf # the default
fi

if $add_pitch; then
  echo "$0: enabling pitch features (note: this has not been tested)"
  echo "--add-pitch=true" >>$conf
  echo "$0: creating $dir/conf/pitch.conf"
  echo "--pitch-config=$dir/conf/pitch.conf" >>$conf
  cp $pitch_config $dir/conf/pitch.conf || exit 1;
  echo "--pitch-process-config=$dir/conf/pitch_process.conf" >>$conf
  cp $pitch_process_config $dir/conf/pitch_process.conf || exit 1;
fi
silphonelist=`cat $lang/phones/silence.csl` || exit 1;
echo "--endpoint.silence-phones=$silphonelist" >>$conf
echo "$0: created config file $conf"


