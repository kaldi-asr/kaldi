#!/bin/bash
# Copyright 2015  Vimal Manohar
# Apache 2.0.

set -e 
set -o pipefail

# Begin configuration section.
cmd=run.pl
nj=32 
stage=-10
add_pitch=false
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: diarization/prepare_data.sh <data> <tmpdir> <featdir>"
  echo " e.g.: diarization/prepare_data.sh data/dev exp/vad_dev mfcc"
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  exit 1;
fi

data=$1
tmpdir=$2
featdir=$3

if [ $stage -le 1 ]; then
  if $add_pitch; then
    steps/make_mfcc_pitch.sh --mfcc-config conf/mfcc_vad.conf --nj $nj --cmd $cmd \
      $data $tmpdir/make_mfcc_vad $featdir || exit 1
  else
    steps/make_mfcc.sh --mfcc-config conf/mfcc_vad.conf --nj $nj --cmd $cmd \
      $data $tmpdir/make_mfcc_vad $featdir || exit 1
  fi
fi

if [ $stage -le 3 ]; then
  steps/compute_cmvn_stats.sh $data $tmpdir/make_mfcc_vad $featdir || exit 1
fi

if [ $stage -le 2 ]; then
  sid/compute_vad_decision.sh --vad-config conf/vad.conf --cmd $cmd --nj $nj \
    $data $tmpdir/make_vad $featdir || exit 1
fi
