#!/usr/bin/env bash
# Author: Ashish Arora
# Apache 2.0

. ./cmd.sh
. ./path.sh

enhancement=gss
. utils/parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Wrong #arguments ($#, expected 3)"
   echo "Usage: local/add_location_to_uttid.sh [options] <json-transcription-in-dir>"
   echo "                        <perutt-in-dir> <uttid-location-mapping-out-file>"
   echo "main options (for others, see top of script file)"
   echo "  --enhancement                    # enhancement type (gss or beamformit)"
   exit 1;
fi

jdir=$1
puttdir=$2
utt_loc_file=$3

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

if [[ ${enhancement} == *gss* ]]; then
  local/get_location.py $jdir > $utt_loc_file
  local/replace_uttid.py $utt_loc_file $puttdir/per_utt > $puttdir/per_utt_loc
fi

if [[ ${enhancement} == *beamformit* ]]; then
  cat $puttdir/per_utt > $puttdir/per_utt_loc
fi
