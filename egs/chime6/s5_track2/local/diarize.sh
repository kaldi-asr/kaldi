#!/bin/bash
# Copyright   2019   David Snder
# Apache 2.0.
#
# This script takes an input directory that has a segments file (and
# a feats.scp file), and performs diarization on it.  The output directory
# contains an RTTM file which can be used to resegment the input data.

stage=0
nj=10
cmd="run.pl"
ref_rttm=

echo "$0 $@"  # Print the command line for logging
if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;
if [ $# != 3 ]; then
  echo "Usage: $0 <model-dir>  <in-data-dir> <out-dir>"
  echo "e.g.: $0 exp/xvector_nnet_1a  data/dev exp/dev_diarization"
  echo "Options: "
  echo "  --nj <nj>                                        # number of parallel jobs."
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --ref-rttm <path to reference RTTM>              # if present, used to score output RTTM."
  exit 1;
fi

model_dir=$1
data_in=$2
out_dir=$3

name=`basename $data_in`

for f in $data_in/feats.scp $data_in/segments $model_dir/plda \
  $model_dir/final.raw $model_dir/extract.config; do
  [ ! -f $f ] && echo "$0: No such file $f" && exit 1;
done

if [ $stage -le 0 ]; then
  echo "$0: keeping only data corresponding to array U06 "
  echo "$0: we can skip this stage, to perform diarization on all arrays "
  cp -r data/$name data/${name}.bak
  mv data/$name/wav.scp data/$name/wav.scp.bak
  grep 'U06' data/$name/wav.scp.bak > data/$name/wav.scp
  utils/fix_data_dir.sh data/$name
  nj=2 # since we have reduced number of "speakers" now
fi

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
    --nj $nj --window 1.5 --period 0.75 --apply-cmn false \
    --min-segment 0.5 $model_dir \
    data/${name}_cmn $out_dir/xvectors_${name}
fi

# Perform PLDA scoring
if [ $stage -le 3 ]; then
  # Perform PLDA scoring on all pairs of segments for each recording.
  echo "$0: performing PLDA scoring between all pairs of x-vectors"
  diarization/nnet3/xvector/score_plda.sh --cmd "$cmd" \
    --target-energy 0.5 \
    --nj $nj $model_dir/ $out_dir/xvectors_${name} \
    $out_dir/xvectors_${name}/plda_scores
fi

if [ $stage -le 4 ]; then
  echo "$0: performing clustering using PLDA scores (we assume 4 speakers per recording)"
  awk '{print $1, "4"}' data/$name/wav.scp > data/$name/reco2num_spk
  diarization/cluster.sh --cmd "$cmd" --nj $nj \
    --reco2num-spk data/$name/reco2num_spk \
    --rttm-channel 1 \
    $out_dir/xvectors_${name}/plda_scores $out_dir
  echo "$0: wrote RTTM to output directory ${out_dir}"
fi

if [ $stage -le 5 ]; then
  if [ -f $ref_rttm ]; then
    echo "$0: computing diariztion error rate (DER) using reference ${ref_rttm}"
    mkdir -p $out_dir/tuning/
    md-eval.pl -c 0.25 -1 -r $ref_rttm -s $out_dir/rttm 2> $out_dir/log/der.log > $out_dir/der
    der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' ${out_dir}/der)
    echo "DER: $der%"
  fi
fi

