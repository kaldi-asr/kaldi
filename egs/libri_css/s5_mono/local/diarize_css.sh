#!/bin/bash
# Copyright   2019   David Snyder
#             2020   Desh Raj

# Apache 2.0.
#
# This script is exactly the same as local/diarize.sh until
# stage 2 (x-vector extraction), but after that, it is slightly
# different. The key difference is that since we have multiple
# streams of audio (and subsequently multiple streams of subsegments)
# from the same recording, we want to perform cosine scoring across 
# all of these streams. 

stage=0
nj=10
cmd="run.pl"
ref_rttm=
window=1.5
period=0.75
min_segment=0.5
post_process_rttm=false # set to true to remove same speaker segments in different
                       # streams at the same time
score_overlaps_only=true

echo "$0 $@"  # Print the command line for logging

set -e

. ./path.sh
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 <model-dir> <in-data-dir> <out-dir>"
  echo "e.g.: $0 exp/xvector_nnet_1a  data/dev exp/dev_diarization"
  echo "Options: "
  echo "  --nj <nj>                                        # number of parallel jobs."
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --ref_rttm ./local/dev_rttm                      # the location of the reference RTTM file"
  exit 1;
fi

model_dir=$1
data_in=$2
out_dir=$3

name=$(basename "$data_in")

for f in $data_in/feats.scp $data_in/segments \
  $model_dir/final.raw $model_dir/extract.config; do
  [ ! -f $f ] && echo "$0: No such file $f" && exit 1;
done

if [ $stage -le 1 ]; then
  echo "$0: computing features for x-vector extractor"
  utils/fix_data_dir.sh data/${name}
  rm -rf data/${name}_cmn
  local/nnet3/xvector/prepare_feats.sh --nj $nj --cmd "$cmd" \
    data/$name data/${name}_cmn exp/${name}_cmn
  cp data/$name/segments exp/${name}_cmn/
  utils/fix_data_dir.sh data/${name}_cmn
fi

if [ $stage -le 2 ]; then
  echo "$0: extracting x-vectors for all segments"
  diarization/nnet3/xvector/extract_xvectors.sh --cmd "$cmd" \
    --nj $nj --window $window --period $period --apply-cmn false \
    --min-segment $min_segment $model_dir \
    data/${name}_cmn $out_dir/xvectors_${name}
fi

# Perform cosine scoring. The following stage is the key difference.
# We change the segments and utt2spk files in the xvector directory
# to reflect that the subsegments are from the same recording. 
# But we also keep the original segments file since that will
# be required in subsequent stages for ASR decoding.
if [ $stage -le 3 ]; then
  # The if condition is just to ensure that we don't accidentally
  # make this modification more than once (which would mess up the
  # segments file)
  if [ ! -f ${out_dir}/xvectors_${name}/segments.bak ]; then
    mv ${out_dir}/xvectors_${name}/segments ${out_dir}/xvectors_${name}/segments.bak
    mv ${out_dir}/xvectors_${name}/utt2spk ${out_dir}/xvectors_${name}/utt2spk.bak
    awk '{$2=$2;sub(/_[0-9]*$/, "", $2); print}' ${out_dir}/xvectors_${name}/segments.bak \
      > ${out_dir}/xvectors_${name}/segments
    awk '{$2=$2;sub(/_[0-9]*$/, "", $2); print}' ${out_dir}/xvectors_${name}/utt2spk.bak \
      > ${out_dir}/xvectors_${name}/utt2spk
    utils/utt2spk_to_spk2utt.pl ${out_dir}/xvectors_${name}/utt2spk > ${out_dir}/xvectors_${name}/spk2utt
  fi
fi

# nj needs to be changed since we now have #wav/#streams number
# of recordings. Just get it from the segments file
new_nj=$(cat ${out_dir}/xvectors_${name}/segments | cut -d' ' -f2 | uniq | wc -l)
nj=$(echo $((nj>new_nj ? new_nj : nj)))

if [ $stage -le 4 ]; then
  # Perform cosine similarity scoring on all pairs of segments for each recording.
  echo "$0: performing cosine similarity scoring between all pairs of x-vectors"
  diarization/score_cossim.sh --cmd "$cmd" \
    --nj $nj $out_dir/xvectors_${name} \
    $out_dir/xvectors_${name}/cossim_scores
fi

if [ $stage -le 5 ]; then
  echo "$0: performing spectral clustering using cosine similarity scores"
  local/diarization/scluster.sh --cmd "$cmd" --nj $nj \
    --rttm-channel 1 \
    $out_dir/xvectors_${name} $out_dir
  echo "$0: wrote RTTM to output directory ${out_dir}"

  # The above clustering generates RTTM with reco separated into streams,
  # so we have to remove the stream name for evaluation.
  awk '{$2=$2;sub(/_[0-9]*$/, "", $2); print}' ${out_dir}/rttm \
    > ${out_dir}/rttm.comb
fi

if [ $stage -le 6 ] && [ $post_process_rttm == "true" ]; then
  echo "$0: applying post-processing to remove simultaneous same-speaker segments"
  local/diarization/post_process_css_rttm.py ${out_dir}/rttm ${out_dir}/rttm.post

  awk '{$2=$2;sub(/_[0-9]*$/, "", $2); print}' ${out_dir}/rttm.post \
    > ${out_dir}/rttm.comb
fi

hyp_rttm=${out_dir}/rttm.comb

if [ $stage -le 7 ]; then
  echo "Diarization results for "${name}
  local/dscore.sh --score-overlaps-only $score_overlaps_only \
    $ref_rttm $hyp_rttm
fi
