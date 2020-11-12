#!/usr/bin/env bash
# This script is used for performing single stage decoding
# for the development data for which we have the transcripts

stage=0
cmd=queue.pl
dataset=safe_t_dev1
. ./utils/parse_options.sh
. ./cmd.sh
. ./path.sh
set -e -o pipefail
set -o nounset

dir=exp/chain_all/tdnn_all
extractor=exp/nnet3_all
nj=$(cat data/${dataset}/spk2utt | wc -l)

if [ $stage -le 0 ] ; then
  utils/copy_data_dir.sh data/$dataset data/${dataset}_hires
  steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
    --cmd "$train_cmd" data/${dataset}_hires
  steps/compute_cmvn_stats.sh data/${dataset}_hires
  utils/fix_data_dir.sh data/${dataset}_hires

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd --max-jobs-run 64" --nj $nj \
    data/${dataset}_hires $extractor/extractor \
    $extractor/ivectors_${dataset}_hires
fi

if [ $stage -le 1 ] ; then
  # decode
  steps/nnet3/decode.sh --num-threads 4 --nj $nj --cmd "$decode_cmd --max-jobs-run 64" \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --online-ivector-dir $extractor/ivectors_${dataset}_hires \
    $dir/graph_3 data/${dataset}_hires $dir/decode_${dataset} || exit 1;
fi
echo "Done"
