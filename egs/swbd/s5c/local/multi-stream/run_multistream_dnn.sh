#!/bin/bash

# Copyright 2012-2014 Sri Harish Mallidi 

# This example script trains multistream ASR system
# The training is done as follows:
# 
# 1) Train a multistream bottleneck feature extractor
#    streams as sub-band streams
# 2) Train performance monitor
#    get the best performing bottleneck features
# All the following steps are motivated from Karel Vesely
# 3) Train a GMM on top of bottleneck features
# 4) fMMLR transform on bottleneck features
# 5) RBM pre-training:
#    in this unsupervised stage we train stack of RBMs, 
#    a good starting point for frame cross-entropy trainig.
# 6) frame cross-entropy training:
#    the objective is to classify frames to correct pdfs.
# 7) sequence-training optimizing sMBR: 
#    the objective is to emphasize state-sequences with better 
#    frame accuracy w.r.t. reference alignment.

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

### Config:
njdec=60


train_id=train_nodup
test_id=eval2000

gmmdir=exp/tri4
stage=0 # resume training with --stage=N
has_fisher=true

# multistream opts
strm_indices="0:30:60:90:120:150:180:210:246:276"
### End of config.
. utils/parse_options.sh || exit 1;
#

set -euxo pipefail

num_streams=`echo $strm_indices | awk -F ":" '{print NF-1}'`
all_stream_combn=`echo 2^$num_streams -1|bc`


# Extract multistream filterbank features
if [ $stage -le 0 ]; then
  ####
  # create multistream-fbank-config
  mkdir -p data-multistream-fbank/conf
  echo "--window-type=hamming" >data-multistream-fbank/conf/fbank_multistream.conf
  echo "--use-energy=false" >>data-multistream-fbank/conf/fbank_multistream.conf
  echo "--sample-frequency=8000" >>data-multistream-fbank/conf/fbank_multistream.conf

  echo "--dither=1" >>data-multistream-fbank/conf/fbank_multistream.conf

  echo "--num-mel-bins=46" >>data-multistream-fbank/conf/fbank_multistream.conf
  echo "--htk-compat=true" >>data-multistream-fbank/conf/fbank_multistream.conf
  ####

  c="$test_id"
  mkdir -p data-multistream-fbank/${c}; 
  cp data/${c}/{glm,reco2file_and_channel,segments,spk2utt,stm,stm.filt,text,utt2spk,wav.scp} data-multistream-fbank/${c}/
  steps/make_fbank.sh --fbank-config data-multistream-fbank/conf/fbank_multistream.conf \
    data-multistream-fbank/$c data-multistream-fbank/$c/log data-multistream-fbank/$c/data || exit 1
  steps/compute_cmvn_stats.sh \
    data-multistream-fbank/$c data-multistream-fbank/$c/log data-multistream-fbank/$c/data || exit 1

  c="$train_id"
  mkdir -p data-multistream-fbank/${c}; 
  cp data/${c}/{reco2file_and_channel,segments,spk2utt,text,utt2spk,wav.scp} data-multistream-fbank/${c}/
  steps/make_fbank.sh --fbank-config data-multistream-fbank/conf/fbank_multistream.conf \
    data-multistream-fbank/$c data-multistream-fbank/$c/log data-multistream-fbank/$c/data || exit 1
  steps/compute_cmvn_stats.sh \
    data-multistream-fbank/$c data-multistream-fbank/$c/log data-multistream-fbank/$c/data || exit 1
fi

## Train multistream bottleneck feature extractor
# This can take a lot of time
if [ $stage -le 1 ]; then
  local/multi-stream/multi-stream_bnfeats_train.sh \
    --ali ${gmmdir}_ali_nodup --train data-multistream-fbank/${train_id} \
    --strm-indices $strm_indices --iters-per-epoch 5 --dir exp/dnn5b_multistream_bottleneck_featureXtractor || exit 1;
fi


if [ $stage -le 2 ]; then
  # Train Performance monitor
  local/multi-stream/multi-stream_autoencoder_perf-monitor_train.sh --stage 1 \
    --train data-multistream-fbank/${train_id} --strm-indices $strm_indices \
    --nnet-dir exp/dnn5b_multistream_bottleneck_featureXtractor \
    --aann-dir exp/aann_tandem_dnn5b_multistream_bottleneck_featureXtractor || exit 1;
fi

# Get multi-stream masks
mask_dir=strm-mask/dnn5b_multistream_bottleneck_featureXtractor
train_mask_dir=$mask_dir/Comb${all_stream_combn}/${train_id}
test_mask_dir=$mask_dir/autoencoder_pm/${test_id}

if [ $stage -le 3 ]; then
  # train, no PM
  local/multi-stream/get-CombX_strm-mask.sh \
    --strm-indices "$strm_indices" --comb-num $all_stream_combn \
    --mask-dir $train_mask_dir --test data-multistream-fbank/${train_id} || exit 1;

  # test
  local/multi-stream/get_autoencoder-pm_strm-mask.sh --njdec $njdec \
    --test data-multistream-fbank/${test_id} \
    --strm-indices $strm_indices --tandem-transf-dir tandem_feats/dnn5b_multistream_bottleneck_featureXtractor_tandem_dim120/pca_transf \
    --aann-dir exp/aann_tandem_dnn5b_multistream_bottleneck_featureXtractor/aann \
    --mask-dir $test_mask_dir || exit 1;

fi

exit 0;

# Extract multistream bottleneck features
if [ $stage -le 3 ]; then
# train, 

# test
fi

# train GMM BNF

# train GMM BNF fMLLR

# RBM

# DNN_XE

# DNN_XE_sMBR


if [ $stage -le 0 ]; then
  # Store fMLLR features, so we can train on them easily,
  # eval2000
  dir=$data_fmllr/eval2000
  steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
     --transform-dir $gmmdir/decode_eval2000_sw1_tg \
     $dir data/eval2000 $gmmdir $dir/log $dir/data
  # train
  dir=$data_fmllr/train_nodup
  steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
     --transform-dir ${gmmdir}_ali_nodup \
     $dir data/train_nodup $gmmdir $dir/log $dir/data
  # split the data : 90% train 10% cross-validation (held-out)
  utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10
fi

if [ $stage -le 1 ]; then
  # Pre-train DBN, i.e. a stack of RBMs
  dir=exp/dnn5b_pretrain-dbn
  $cuda_cmd $dir/log/pretrain_dbn.log \
    steps/nnet/pretrain_dbn.sh --rbm-iter 1 $data_fmllr/train_nodup $dir
fi

if [ $stage -le 2 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  dir=exp/dnn5b_pretrain-dbn_dnn
  ali=${gmmdir}_ali_nodup
  feature_transform=exp/dnn5b_pretrain-dbn/final.feature_transform
  dbn=exp/dnn5b_pretrain-dbn/6.dbn
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
    $data_fmllr/train_nodup_tr90 $data_fmllr/train_nodup_cv10 data/lang $ali $ali $dir
  # Decode with the trigram swbd language model.
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" \
    --config conf/decode_dnn.config --acwt 0.08333 \
    $gmmdir/graph_sw1_tg $data_fmllr/eval2000 \
    $dir/decode_eval2000_sw1_tg
  if $has_fisher; then
    # Rescore with the 4gram swbd+fisher language model.
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
      data/lang_sw1_{tg,fsh_fg} data/eval2000 \
      $dir/decode_eval2000_sw1_{tg,fsh_fg}
  fi
fi


# Sequence training using sMBR criterion, we do Stochastic-GD 
# with per-utterance updates. The typical acwt value is around 0.1
dir=exp/dnn5b_pretrain-dbn_dnn_smbr
srcdir=exp/dnn5b_pretrain-dbn_dnn
acwt=0.0909

if [ $stage -le 3 ]; then
  # First we generate lattices and alignments:
  steps/nnet/align.sh --nj 250 --cmd "$train_cmd" \
    $data_fmllr/train_nodup data/lang $srcdir ${srcdir}_ali
  steps/nnet/make_denlats.sh --nj 10 --sub-split 100 --cmd "$decode_cmd" --config conf/decode_dnn.config \
    --acwt $acwt $data_fmllr/train_nodup data/lang $srcdir ${srcdir}_denlats
fi

if [ $stage -le 4 ]; then
  # Re-train the DNN by 1 iteration of sMBR 
  steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 4 --acwt $acwt --do-smbr true \
    $data_fmllr/train_nodup data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir
  # Decode (reuse HCLG graph)
  for ITER in 4 3 2 1; do
    # Decode with the trigram swbd language model.
    steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" \
      --config conf/decode_dnn.config \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $gmmdir/graph_sw1_tg $data_fmllr/eval2000 \
      $dir/decode_eval2000_sw1_tg_it$ITER
    if $has_fisher; then
      # Rescore with the 4gram swbd+fisher language model.
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
        data/lang_sw1_{tg,fsh_fg} data/eval2000 \
        $dir/decode_eval2000_sw1_{tg,fsh_fg}_it$ITER
    fi
  done 
fi

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
