#!/bin/sh
set -euo pipefail
set -e -o pipefail                                                              
set -o nounset                              # Treat unset variables as an error 
echo "$0 $@"

data=$1
graph_dir=$2
decode_dir=$3

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh


# get recording-level CTMs from the lattice by resolving the overlapping
# regions

steps/get_ctm_fast.sh --frame-shift 0.03 \
  data/${data}_hires/ ${graph_dir} \
  ${decode_dir} ${decode_dir}/score_10/

cat ${decode_dir}/score_10/ctm.* \
  > ${decode_dir}/score_10/ctm

awk '{print $2" "$2" 1"}' data/${data}_hires/segments > \
  data/${data}_hires/reco2file_and_channel

utils/ctm/resolve_ctm_overlaps.py data/${data}_hires/segments \
  ${decode_dir}/score_10/ctm \
  - | utils/convert_ctm.pl data/${data}_hires/segments \
  data/${data}_hires/reco2file_and_channel > \
  ${decode_dir}/score_10/ctm_out

# compute WER              
local/score_segments.sh data/${data}_hires/ ${decode_dir}

