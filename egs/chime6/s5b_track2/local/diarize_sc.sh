#!/bin/bash
# Copyright   2019   David Snyder
#             2020   Maxim Korenevsky, Ivan Medennikov (STC-innovations Ltd)
# Apache 2.0.
#
# This script takes an input directory that has a segments file (and
# a feats.scp file), and performs spectral clustering based diarization on it. 
# The output directory contains an RTTM file which can be used to resegment the input data.

stage=0
nj=10
cmd="run.pl"
ref_rttm=

echo "$0 $@"  # Print the command line for logging
if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;
if [ $# != 3 ]; then
  echo "Usage: $0 <model-dir> <in-data-dir> <out-dir>"
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
  # to perform diarization ond scoring on all array please skip this step and
  # pass all_array = true in local/multispeaker_score.sh
  cp -r data/$name data/${name}.bak
  mv data/$name/wav.scp data/$name/wav.scp.bak
  grep 'U06' data/$name/wav.scp.bak > data/$name/wav.scp
  utils/fix_data_dir.sh data/$name
fi
nj=2 # since we have reduced number of "speakers" now

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

# Perform cosine similarity scoring
if [ $stage -le 3 ]; then
  # Perform cosine similarity scoring on all pairs of segments for each recording.
  echo "$0: performing cosine similarity scoring between all pairs of x-vectors"
  diarization/score_cossim.sh --cmd "$cmd" \
    --nj $nj $out_dir/xvectors_${name} \
    $out_dir/xvectors_${name}/cossim_scores
fi


if [ $stage -le 4 ]; then
  echo "$0: performing spectral clustering using cosine similarity scores (we assume 4 speakers per recording)"
  awk '{print $1, "4"}' data/$name/wav.scp > data/$name/reco2num_spk
  diarization/scluster.sh --cmd "$cmd" --nj $nj \
    --reco2num-spk data/$name/reco2num_spk \
    --rttm-channel 1 \
    $out_dir/xvectors_${name}/cossim_scores $out_dir
  echo "$0: wrote RTTM to output directory ${out_dir}"
fi

hyp_rttm=${out_dir}/rttm

# For scoring the diarization system, we use the same tool that was
# used in the DIHARD II challenge. This is available at:
# https://github.com/nryant/dscore
# Note that the scoring takes a single reference RTTM and a single
# hypothesis RTTM.
if [ $stage -le 5 ]; then
  # If a reference RTTM file is not provided, we create one using the backed up
  # segments and utt2spk files in the original data directory.
  if [ -z "$ref_rttm" ]; then
    steps/segmentation/convert_utt2spk_and_segments_to_rttm.py data/$name/utt2spk.bak \
      data/$name/segments.bak data/$name/rttm
    ref_rttm=data/$name/rttm
  fi
  echo "Diarization results for "${name}
  if ! [ -d dscore ]; then
    git clone https://github.com/nryant/dscore.git || exit 1;
    cd dscore
    python -m pip install --user -r requirements.txt
    cd ..
  fi
  sed 's/_U0[1-6]\.ENH//g' $ref_rttm > $ref_rttm.scoring
  sed 's/_U0[1-6]\.ENH//g' $hyp_rttm > $hyp_rttm.scoring
  ref_rttm_path=$(readlink -f ${ref_rttm}.scoring)
  hyp_rttm_path=$(readlink -f ${hyp_rttm}.scoring)
  cat ./local/uem_file | grep 'U06' | sed 's/_U0[1-6]//g' > ./local/uem_file.scoring
  cd dscore && python score.py -u ../local/uem_file.scoring -r $ref_rttm_path \
    -s $hyp_rttm_path 2>&1 | tee -a ../${out_dir}/DER && cd .. || exit 1;
fi
