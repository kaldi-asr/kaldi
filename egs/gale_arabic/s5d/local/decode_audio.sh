#!/usr/bin/env bash

set -e

stage=0
nj=10

data_dir=data/test
model_dir=exp
conf_dir=conf
dir=${model_dir}/chain/tdnn_1a_sp
rnnlm_dir=${model_dir}/rnnlm_lstm_1e
lang_dir=data/lang_test

preprocess=true

min_lmwt=9
max_lmwt=9

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

audio_list=$1
output_file=$2

if [ $# != 2 ]; then
  echo "Usage: ./decode_audio.sh <audio-list-file> <output-file>"
  exit 1;
fi

if ! cuda-compiled; then
   cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

mkdir -p ${dir}
mkdir -p ${data_dir}
mkdir -p ${rnnlm_dir}

if ! command -v "sox" > /dev/null 2>&1; then
  echo "It seems sox it not installed. Can not process flac file."
fi

if [ ${stage} -le 0 ]; then
  echo "$0: Preprocessing test data"
  # generate wav.scp (*.wav, *.flac, *.sph)
  relative_path=false

  if [ ! -f ${audio_list} ]; then
    echo "$0: Audio list file ${audio_list} does not exit" && exit 1;
  fi

  echo $KALDI_ROOT
  local/get_wavscp.py ${relative_path} ${audio_list} ${data_dir}/wav.scp $KALDI_ROOT
  awk '{print $1" "$1}' ${data_dir}/wav.scp > ${data_dir}/utt2spk

  if ${prepocess} ; then
    echo "$0: Doing segmentation on test data"
    local/preprocess_test.sh ${data_dir}
    data_dir=${data_dir}_segmented
    utils/fix_data_dir.sh ${data_dir}
  fi
fi

[ -d ${data_dir}_segmented ] && data_dir=${data_dir}_segmented
num_segs=$(wc -l < ${data_dir}/segments)
[ $nj -gt ${num_segs} ] && nj=${num_segs}

if [ $stage -le 1 ]; then
  echo "$0: Creating high-resolution MFCC features"
  utils/copy_data_dir.sh ${data_dir} ${data_dir}_hires
  steps/make_mfcc.sh --nj ${nj} --mfcc-config ${conf_dir}/mfcc_hires.conf \
    --cmd "${train_cmd}" ${data_dir}_hires exp/make_mfcc exp/mfcc
  steps/compute_cmvn_stats.sh ${data_dir}_hires
  utils/fix_data_dir.sh ${data_dir}_hires
fi

num_spks=$(wc -l < ${data_dir}_hires/spk2utt)
[ $nj -gt ${num_spks} ] && nj=${num_spks}

if [ $stage -le 2 ]; then
  echo "$0: Extracting i-vector for test data"
  steps/online/nnet2/extract_ivectors_online.sh --cmd "${train_cmd}" --nj ${nj} \
    ${data_dir}_hires ${model_dir}/nnet3/extractor \
    exp/nnet3/ivectors_test_hires
fi

tree_dir=${model_dir}/chain/tree_a_sp
chunk_width=150,110,100

if [ $stage -le 3 ]; then
  echo "$0: Decoding with chain model"
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)

  steps/nnet3/decode.sh --skip-scoring true \
    --skip-diagnostics true \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --extra-left-context 0 --extra-right-context 0 \
    --extra-left-context-initial 0 \
    --extra-right-context-final 0 \
    --frames-per-chunk $frames_per_chunk \
    --nj $nj --cmd "${decode_cmd}"  --num-threads 1 \
    --online-ivector-dir exp/nnet3/ivectors_test_hires \
    $tree_dir/graph ${data_dir}_hires ${dir}/decode_asr || exit 1
fi

weight=0.5
ngram_order=4
ac_model_dir=${dir}
decode_dir=${ac_model_dir}/decode_asr

if [ $stage -le 4 ]; then
  echo "$0: Performing lattice-rescoring"
  pruned=_pruned

  rnnlm/lmrescore${pruned}.sh --skip-scoring true \
    --cmd "${decode_cmd} --mem 8G" \
    --weight ${weight} --max-ngram-order ${ngram_order} \
    ${lang_dir} ${rnnlm_dir} \
    ${data_dir}_hires ${decode_dir} \
    ${decode_dir}_rnnlm_${weight}
fi

cmd=${decode_cmd}
wip="0.0"

if [ $stage -le 5 ]; then
  mkdir -p ${data_dir}/tmp
  mkdir -p ${decode_dir}_rnnlm_${weight}/scoring_kaldi/penalty_${wip}_rnnlm
  echo "$0: Getting the best path"

  local/get_1best.sh --cmd "${cmd}" --wip ${wip} --min-lmwt ${min_lmwt} --max-lmwt ${max_lmwt} \
    ${data_dir} ${lang_dir} ${decode_dir} ${dir} ${output_file}
fi

echo "$0: Successfully finish decoding, results are at ${output_file}"
exit 0;
