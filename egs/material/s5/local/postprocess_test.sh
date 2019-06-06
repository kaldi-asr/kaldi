#!/bin/sh
set -euo pipefail
echo "$0 $@"

stage=0

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if [ $# -ne 3 ]; then
  echo "Usage: $0 <data-id> <graph-dir> <decode-dir>"
  echo " e.g.: $0 analysis1 exp/chain/tdnn/graph exp/chain/tdnn/decode_analysis1_segmented"
  exit 1
fi

data=$1
graph_dir=$2
decode_dir=$3

# get recording-level CTMs from the lattice by resolving the overlapping
# regions

if [ $stage -le 0 ]; then
  steps/get_ctm_fast.sh --cmd "$decode_cmd" --frame-shift 0.03 \
    data/${data}_hires/ ${graph_dir} \
    ${decode_dir} ${decode_dir}/score_10_0.0
fi

if [ $stage -le 1 ]; then
  utils/ctm/resolve_ctm_overlaps.py data/${data}_hires/segments \
    ${decode_dir}/score_10_0.0/ctm \
    - | utils/convert_ctm.pl data/${data}_hires/segments data/${data}_hires/reco2file_and_channel > \
    ${decode_dir}/score_10_0.0/${data}_hires.ctm
fi

if [ $stage -le 2 ]; then
  # extract n-best lists from archive.* files
  if [[ ${decode_dir} == *_rescore_nbest ]]; then
    hyp_filtering_cmd="cat"
    [ -x local/wer_output_filter ] && hyp_filtering_cmd="local/wer_output_filter"
    [ -x local/wer_hyp_filter ] && hyp_filtering_cmd="local/wer_hyp_filter"
    mkdir -p ${decode_dir}/output_nbest
    for f in ${decode_dir}/archives.*; do
      docid=$(head -1 $f/words_text | awk '{print $1}' | cut -f1,2 -d'-')
      $hyp_filtering_cmd $f/words_text  > \
        ${decode_dir}/output_nbest/$docid".n.txt" || exit 1;
    done
  fi

  # compute WER              
  local/score_stm.sh --min-lmwt 10 --max-lmwt 10 --word-ins-penalty 0.0 \
    --cmd "$decode_cmd" data/${data}_hires $graph_dir ${decode_dir}

  grep -H Sum ${decode_dir}/score*/*.sys | utils/best_wer.sh
fi
