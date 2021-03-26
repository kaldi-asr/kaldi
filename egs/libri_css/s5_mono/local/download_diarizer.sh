#!/usr/bin/env bash
#
# Copyright  2020  University of Stuttgart (Author: Pavel Denisov)
# Apache 2.0

# Begin configuration section.
# End configuration section
. ./utils/parse_options.sh  # accept options

. ./path.sh

echo >&2 "$0" "$@"
if [ $# -ne 0 ] ; then
  echo >&2 "$0" "$@"
  echo >&2 "$0: Error: wrong number of arguments"
  echo -e >&2 "Usage:\n  $0"
  exit 1
fi


set -e -o pipefail

mkdir -p downloads
dir=$(mktemp -d ./downloads/lcss.XXXXXXXXX)
trap "rm -rf ${dir}" EXIT

cd ${dir}

# Download x-vector extractor trained on VocxCeleb2 data
wget http://kaldi-asr.org/models/12/0012_diarization_v1.tar.gz
tar -xvzf 0012_diarization_v1.tar.gz
rm -f 0012_diarization_v1.tar.gz

# Download PLDA model trained on augmented Librispeech data
rm 0012_diarization_v1/exp/xvector_nnet_1a/plda
wget https://desh2608.github.io/static/files/jsalt/plda -P 0012_diarization_v1/exp/xvector_nnet_1a/
cd ../..
cp -r ${dir}/0012_diarization_v1/exp .
