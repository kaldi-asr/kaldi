#!/bin/bash -u

. ./cmd.sh
. ./path.sh

# DNN training, on FBANK features based on egs/ami/s5/local/run_dnn.sh

# Config:
nj=80
nj_decode=30
stage=0 # resume training with --stage=N
. utils/parse_options.sh || exit 1;
#

if [ $# -ne 1 ]; then
  printf "\nUSAGE: %s [opts] <mic condition(ihm|sdm|mdm)>\n\n" `basename $0`
  exit 1;
fi
mic=$1

gmmdir=exp/$mic/tri4a
data_fbank=data-fbank/$mic

final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7
graph_dir=$gmmdir/graph_${LM}

# Set bash to 'debug' mode, it will exit on : 
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail
set -x

# Extract FBANK features
if [ $stage -le 0 ]; then
  # eval
  mkdir -p $data_fbank/eval; 
  cp data/$mic/eval/{stm,glm,text,utt2spk,segments,wav.scp,reco2file_and_channel,spk2utt} $data_fbank/eval/;
  steps/make_fbank.sh --cmd "$train_cmd" --nj 15 --compress false \
  $data_fbank/eval $data_fbank/eval/log $data_fbank/eval/data 
  steps/compute_cmvn_stats.sh $data_fbank/eval $data_fbank/eval/log $data_fbank/eval/data

  # dev
  mkdir -p $data_fbank/dev; 
  cp data/$mic/dev/{stm,glm,text,utt2spk,segments,wav.scp,reco2file_and_channel,spk2utt} $data_fbank/dev/;
  steps/make_fbank.sh --cmd "$train_cmd" --nj 15 --compress false \
  $data_fbank/dev $data_fbank/dev/log $data_fbank/dev/data 
  steps/compute_cmvn_stats.sh $data_fbank/dev $data_fbank/dev/log $data_fbank/dev/data

  # train
  mkdir -p $data_fbank/train; 
  cp data/$mic/train/{text,utt2spk,segments,wav.scp,reco2file_and_channel,spk2utt} $data_fbank/train/;
  steps/make_fbank.sh --cmd "$train_cmd" --nj 30 --compress false \
  $data_fbank/train $data_fbank/train/log $data_fbank/train/data 
  steps/compute_cmvn_stats.sh $data_fbank/train $data_fbank/train/log $data_fbank/train/data

fi

# Train the DNN optimizing per-frame cross-entropy,
if [ $stage -le 1 ]; then
  dir=exp/$mic/dnn4_fbank_dnn
  ali=${gmmdir}_ali

  utils/subset_data_dir_tr_cv.sh $data_fbank/train $data_fbank/train_tr90 $data_fbank/train_cv10 
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh \
    --cmvn-opts "--norm-means=true --norm-vars=true" --delta-opts "--delta-order=2" \
    $data_fbank/train_tr90 $data_fbank/train_cv10 data/lang $ali $ali $dir
  # Decode (reuse HCLG graph)
  steps/nnet/decode.sh --nj $nj_decode --cmd "$decode_cmd" --config conf/decode_dnn.conf --acwt 0.1 \
    --num-threads 3 \
    $graph_dir $data_fbank/dev $dir/decode_dev_${LM}
  steps/nnet/decode.sh --nj $nj_decode --cmd "$decode_cmd" --config conf/decode_dnn.conf --acwt 0.1 \
    --num-threads 3 \
    $graph_dir $data_fbank/eval $dir/decode_eval_${LM}
fi

# Getting results [see RESULTS file]
# for x in exp/$mic/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done

