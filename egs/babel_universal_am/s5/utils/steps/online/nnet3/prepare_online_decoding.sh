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
# online_pitch_config is the config file for both pitch extraction and
# post-processing; we combine them into one because during training this
# is given to the program compute-and-process-kaldi-pitch-feats.
online_pitch_config=conf/online_pitch.conf

# Below are some options that affect the iVectors, and should probably
# match those used in extract_ivectors_online.sh.
num_gselect=5 # Gaussian-selection using diagonal model: number of Gaussians to select
posterior_scale=0.1 # Scale on the acoustic posteriors, intended to account for
                    # inter-frame correlations.
min_post=0.025 # Minimum posterior to use (posteriors below this are pruned out)
               # caution: you should use the same value in the online-estimation
               # code.
max_count=100   # This max-count of 100 can make iVectors more consistent for
                # different lengths of utterance, by scaling up the prior term
                # when the data-count exceeds this value.  The data-count is
                # after posterior-scaling, so assuming the posterior-scale is
                # 0.1, --max-count 100 starts having effect after 1000 frames,
                # or 10 seconds of data.
iter=final
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
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --config <config-file>                           # config containing options"
   echo "  --iter <model-iteration|final>                   # iteration of model to take."
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

for f in $lang/phones/silence.csl $srcdir/${iter}.mdl $srcdir/tree; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done
if [ ! -z "$iedir" ]; then
  for f in final.{mat,ie,dubm} splice_opts global_cmvn.stats online_cmvn.conf; do
    [ ! -f $iedir/$f ] && echo "$0: no such file $iedir/$f" && exit 1;
  done
fi


dir=$(utils/make_absolute.sh $dir) # Convert $dir to an absolute pathname, so that the
                        # configuration files we write will contain absolute
                        # pathnames.
mkdir -p $dir/conf

utils/lang/check_phones_compatible.sh $lang/phones.txt $srcdir/phones.txt || exit 1;
cp $lang/phones.txt $dir || exit 1;

cp $srcdir/${iter}.mdl $dir/final.mdl || exit 1;
cp $srcdir/tree $dir/ || exit 1;
if [ -f $srcdir/frame_subsampling_factor ]; then
	cp $srcdir/frame_subsampling_factor $dir/
fi

if [ ! -z "$iedir" ]; then
  mkdir -p $dir/ivector_extractor/
  cp $iedir/final.{mat,ie,dubm} $iedir/global_cmvn.stats $dir/ivector_extractor/ || exit 1;

  # The following things won't be needed directly by the online decoding, but
  # will allow us to run prepare_online_decoding.sh again with
  # $dir/ivector_extractor/ as the input directory (useful in certain
  # cross-system training scenarios).
  cp $iedir/splice_opts $iedir/online_cmvn.conf $dir/ivector_extractor/ || exit 1;
fi


mkdir -p $dir/conf
rm $dir/{plp,mfcc,fbank}.conf 2>/dev/null
echo "$0: preparing configuration files in $dir/conf"

if [ -f $dir/conf/online.conf ]; then
  echo "$0: moving $dir/conf/online.conf to $dir/conf/online.conf.bak"
  mv $dir/conf/online.conf $dir/conf/online.conf.bak
fi

conf=$dir/conf/online.conf
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
  ieconf=$dir/conf/ivector_extractor.conf
  echo -n >$ieconf
  echo "--ivector-extraction-config=$ieconf" >>$conf
  cp $iedir/online_cmvn.conf $dir/conf/online_cmvn.conf || exit 1;
  # the next line puts each option from splice_opts on its own line in the config.
  for x in $(cat $iedir/splice_opts); do echo "$x"; done > $dir/conf/splice.conf
  echo "--splice-config=$dir/conf/splice.conf" >>$ieconf
  echo "--cmvn-config=$dir/conf/online_cmvn.conf" >>$ieconf
  echo "--lda-matrix=$dir/ivector_extractor/final.mat" >>$ieconf
  echo "--global-cmvn-stats=$dir/ivector_extractor/global_cmvn.stats" >>$ieconf
  echo "--diag-ubm=$dir/ivector_extractor/final.dubm" >>$ieconf
  echo "--ivector-extractor=$dir/ivector_extractor/final.ie" >>$ieconf
  echo "--num-gselect=$num_gselect"  >>$ieconf
  echo "--min-post=$min_post" >>$ieconf
  echo "--posterior-scale=$posterior_scale" >>$ieconf # this is currently the default in the scripts.
  echo "--max-remembered-frames=1000" >>$ieconf # the default
  echo "--max-count=$max_count" >>$ieconf
fi

if $add_pitch; then
  echo "$0: enabling pitch features"
  echo "--add-pitch=true" >>$conf
  echo "$0: creating $dir/conf/online_pitch.conf"
  if [ ! -f $online_pitch_config ]; then
    echo "$0: expected file '$online_pitch_config' to exist.";
    exit 1;
  fi
  cp $online_pitch_config $dir/conf/online_pitch.conf || exit 1;
  echo "--online-pitch-config=$dir/conf/online_pitch.conf" >>$conf
fi

silphonelist=`cat $lang/phones/silence.csl` || exit 1;
echo "--endpoint.silence-phones=$silphonelist" >>$conf
echo "$0: created config file $conf"
