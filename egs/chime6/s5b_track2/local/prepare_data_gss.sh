#!/bin/bash
#
# Copyright  2017  Johns Hopkins University (Author: Shinji Watanabe, Yenda Trmal)
#            2021  Johns Hopkins University (Author: Desh Raj)
# Apache 2.0

# This scripts prepares the data directory from the enhanced wav files and the
# RTTM file.

. ./utils/parse_options.sh  # accept options.. you can run this run.sh with the

. ./path.sh

echo >&2 "$0" "$@"
if [ $# -ne 2 ] ; then
  echo >&2 "$0" "$@"
  echo >&2 "$0: Error: wrong number of arguments"
  echo -e >&2 "Usage:\n  $0 [opts] <audio-dir> <out-data-dir>"
  echo -e >&2 "eg:\n  $0 enhanced/gss/dev exp/dev_diarization/rttm data/dev_gss"
  exit 1
fi

set -e -o pipefail

adir=$(utils/make_absolute.sh $1)
data_dir=$2

mkdir -p $data_dir

find -L $adir -name  "S[0-9]*.wav" | \
  perl -ne '{
    chomp;
    $path = $_;
    next unless $path;
    @F = split "/", $path;
    ($f = $F[@F-1]) =~ s/.wav//;
    print "$f $path\n";
  }' | sort > $data_dir/wav.scp

paste <(cut -d' ' -f1 $data_dir/wav.scp) <(cut -d'-' -f1,2 $data_dir/wav.scp) \
  >$data_dir/utt2spk

utils/utt2spk_to_spk2utt.pl $data_dir/utt2spk > $data_dir/spk2utt

# Some wav files may not have been generated due to GSS issues
utils/fix_data_dir.sh ${data_dir}
# Check that data dirs are okay!
utils/validate_data_dir.sh --no-text --no-feats $data_dir || exit 1
