#!/bin/sh
set -euo pipefail
set -e -o pipefail                                                              
set -o nounset                              # Treat unset variables as an error 
echo "$0 $@"

test_sets=$1
tree_dir=$2
dir=$3
language=$4

./cmd.sh                                                                        
./path.sh                                                                       
./utils/parse_options.sh

# get recording-level CTMs from the lattice by resolving the overlapping
# regions

for data in $test_sets; do
  steps/cleanup/internal/get_ctm.sh --frame-shift 0.03 \
    data/$language/${data}_hires/ $tree_dir/graph/ \
    $dir/decode_$data/

  cat $dir/decode_$data/score_10/${data}_hires.ctm.* \
    > $dir/decode_$data/score_10/${data}_hires.ctm

  awk '{print $2" "$2" 1"}' data/$language/${data}_hires/segments > \
    data/$language/${data}_hires/reco2file_and_channel

  utils/ctm/resolve_ctm_overlaps.py data/$language/${data}_hires/segments\
    $dir/decode_$data/score_10/${data}_hires.ctm \
    - | utils/convert_ctm.pl data/$language/${data}_hires/segments \
    data/$language/${data}_hires/reco2file_and_channel > \
   $dir/decode_$data/score_10/ctm_out

  # compute WER              
  local/score_segments.sh data/$language/${data}_hires/ $dir/decode_$data
done

exit 0;
