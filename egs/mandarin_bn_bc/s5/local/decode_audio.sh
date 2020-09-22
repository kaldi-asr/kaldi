#!/usr/bin/env bash


stage=-1
nj=10
fixed_lmwt=9
preprocess=true # Segment audio data and generate required files for decoding

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

if [ $# -ne 2 ]; then
  echo "Usage: $0 <input-audio-list> <asr-output-text-filename>"
  echo "e.g.: $0 test_audio_list.txt asr_output.txt"
  echo "  --nj <nj>                    # number of parallel jobs"
  echo "  --fixed-lmwt <int>             # use this LM-weight for lattice rescoring"
  exit 1;
fi

audio_list=$1
output_file=$2

##### The paths below should be provided by the user ######
lang_dir=data/lang_large_test
model_dir=exp
ac_model_dir=exp/chain_cleanup/tdnn_1d_sp
graph_dir=exp/chain_cleanup/tdnn_1d_sp/graph_large_test
rnnlm_dir=exp/rnnlm_lstm_1a
ivec_extractor=exp/nnet3_cleanup/extractor
##### The paths above should be provided by the user ######

datadir=data/test
output_dir=exp/asr_tmp_output
mkdir -p $datadir
mkdir -p $output_dir

rnn_weight=0.45 # For RNNLM rescoring
ngram_order=4 # For RNNLM rescroing
wip=0.0 # word insertion penalty
min_lmwt=$fixed_lmwt # Usually between 7-10
max_lmwt=$fixed_lmwt # Usually between 15-20

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

set -e -o pipefail

if [ $stage -le 0 ]; then
  relative_path=false
  echo "$(date -u): Creating $datadir: wav.scp, utt2spk"
  [ ! -f $audio_list ] && echo "$audio_list does not exist " && exit 1;
  bash local/get_wavscp.sh $audio_list $datadir
  awk '{print $1" "$1}' ${datadir}/wav.scp > $datadir/utt2spk

  if $preprocess; then
    echo "$(date -u): Performing uniform segmentation for the test data"
    local/preprocess_test.sh --nj $nj $datadir ${datadir}_segmented
  fi
fi
[ -d ${datadir}_segmented ] && datadir=${datadir}_segmented
utils/fix_data_dir.sh ${datadir}

num_segs=$(wc -l < ${datadir}/segments)
if [ $nj -gt $num_segs ]; then
  nj=$num_segs
fi
if [ $stage -le 1 ]; then
  echo "$(date -u): Getting high-resolution MFCC features for test data"
  mfccdir=mfcc_hires
  utils/copy_data_dir.sh $datadir ${datadir}_hires

  steps/make_mfcc_pitch_online.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
    --cmd "$train_cmd" ${datadir}_hires $output_dir/make_mfcc_hires $output_dir/$mfccdir || exit 1
  steps/compute_cmvn_stats.sh ${datadir}_hires $output_dir/make_mfcc_hires $output_dir/$mfccdir
  utils/fix_data_dir.sh ${datadir}_hires

  # Remove pitch to extract ivectors
  utils/data/limit_feature_dim.sh 0:39 ${datadir}_hires ${datadir}_hires_nopitch || exit 1;
  steps/compute_cmvn_stats.sh ${datadir}_hires_nopitch \
    $output_dir/make_mfcc_hires_no_pitch \
    $output_dir/${mfccdir}_no_pitch || exit 1;
fi

ivector_dir=$output_dir/ivectors_test_hires_nopitch
num_spks=$(wc -l < ${datadir}_hires_nopitch/spk2utt)
if [ $nj -gt $num_spks ]; then
  nj=$num_spks
fi
if [ $stage -le 2 ]; then
  echo "$(date -u): Getting i-vector for test data"
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    ${datadir}_hires_nopitch $ivec_extractor $ivector_dir
fi

decode_dir=${ac_model_dir}/decode_test
if [ $stage -le 3 ]; then
  echo "$(date -u): Decoding test dataset, results in $decode_dir"
  steps/nnet3/decode.sh --skip-scoring true --skip-diagnostics true \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --nj $nj --cmd "$decode_cmd" \
    --online-ivector-dir "$ivector_dir" \
    $graph_dir ${datadir}_hires $decode_dir || exit 1
fi

decode_dir_rnnlm=${ac_model_dir}/decode_test_rnnlm_${rnn_weight}
if [ $stage -le 4 ]; then
  echo "$(date -u): Perform lattice-rescoring on test data, results in $decode_dir_rnnlm"
  pruned=_pruned
  rnnlm/lmrescore$pruned.sh --skip-scoring true \
    --cmd "$decode_cmd --mem 8G" \
    --weight $rnn_weight --max-ngram-order $ngram_order \
    $lang_dir $rnnlm_dir \
    ${datadir}_hires $decode_dir $decode_dir_rnnlm
  bash local/get_1best.sh --cmd "$decode_cmd" \
    --wip ${wip} --min-lmwt ${min_lmwt} --max-lmwt ${max_lmwt} \
    ${datadir}_hires $graph_dir $decode_dir_rnnlm $output_dir
  cp -r $output_dir/scoring_kaldi/penalty_${wip}/$fixed_lmwt.txt $output_file
fi

echo "$(date -u) $0: Successfully finish all steps. ASR output is in $output_file"

exit 0;
