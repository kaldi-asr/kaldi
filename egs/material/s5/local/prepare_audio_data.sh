#!/usr/bin/env bash
# Copyright (c) 2017, Johns Hopkins University (Jan "Yenda" Trmal<jtrmal@gmail.com>)
# License: Apache 2.0

# Begin configuration section.
# End configuration section
set -e -o pipefail
set -o nounset                              # Treat unset variables as an error
echo "$0 " "$@"

if [ $# -ne 1 ] ; then
  echo "Invalid number of script parameters. "
  echo "  $0 <path-to-material-corpus>"
  echo "e.g."
  echo "  $0 /export/corpora5/MATERIAL/IARPA_MATERIAL_BASE-1A-BUILD_v1.0/"
  exit
fi
data=$1

conversational_train=$data/conversational/training/
audio=$conversational_train/audio/
[ ! -d $audio ] && \
  echo "The directory $audio does not exist!" && exit 1

find $audio -type f \( -name "*.wav" -o -name "*.sph" \) | \
  local/audio2wav_scp.pl > data/train/wav.scp


conversational_dev=$data/conversational/dev
audio=$conversational_dev/audio/
[ ! -d $audio ] && \
  echo "The directory $audio does not exist!" && exit 1

find $audio -type f \( -name "*.wav" -o -name "*.sph" \) | \
  local/audio2wav_scp.pl > data/dev/wav.scp

