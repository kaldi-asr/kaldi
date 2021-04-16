#!/bin/bash
#
# Copyright  2017  Johns Hopkins University (Author: Shinji Watanabe, Yenda Trmal)
#            2021  Johns Hopkins University (Author: Desh Raj)
# Apache 2.0

# This scripts prepares the data directory from the enhanced wav files and the
# RTTM file.

# Begin configuration section.
cleanup=true
# End configuration section
. ./utils/parse_options.sh  # accept options.. you can run this run.sh with the

. ./path.sh

echo >&2 "$0" "$@"
if [ $# -ne 3 ] ; then
  echo >&2 "$0" "$@"
  echo >&2 "$0: Error: wrong number of arguments"
  echo -e >&2 "Usage:\n  $0 [opts] <audio-dir> <rttm-file> <out-data-dir>"
  echo -e >&2 "eg:\n  $0 enhanced/gss/dev exp/dev_diarization/rttm data/dev_gss"
  exit 1
fi

set -e -o pipefail

adir=$(utils/make_absolute.sh $1)
rttm_file=$2
data_dir=$3

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

$cleanup && rm -f $data_dir/text.* $data_dir/wav.scp.* $data_dir/wav.flist

cat ${rttm_file} |\
  sed 's/'.ENH'//g' |\
  local/truncate_rttm.py --min-segment-length 0.2 - local/uem_file - |\
  sed 's/_U06/_U06.ENH/g' |\
  awk '($8=="5"){$8="1"}{print $0}' > ${rttm_file}_1

local/convert_rttm_to_utt2spk_and_segments.py --append-reco-id-to-spkr=true ${rttm_file}_1 \
  <(awk '{print $2,$2,$3}' ${rttm_file}_1 | sort -u) \
  ${data_dir}/utt2spk ${data_dir}/segments_all

utils/utt2spk_to_spk2utt.pl $data_dir/utt2spk > $data_dir/spk2utt

# Check that data dirs are okay!
utils/validate_data_dir.sh --no-text --no-feats $data_dir || exit 1
