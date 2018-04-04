#!/bin/bash

# Copyright 2014 (author: Hainan Xu, Ahmed Ali)
# Apache 2.0

. ./path.sh
. ./cmd.sh

#train DNN
mfcc_fmllr_dir=mfcc_fmllr
baseDir=exp/tri3b
alignDir=exp/tri3b_ali
dnnDir=exp/tri3b_dnn_2048x5
align_dnnDir=exp/tri3b_dnn_2048x5_ali
dnnLatDir=exp/tri3b_dnn_2048x5_denlats
dnnMPEDir=exp/tri3b_dnn_2048x5_smb

trainTr90=data/train_tr90
trainCV=data/train_cv10 

steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
  --transform-dir $baseDir/decode data/dev_fmllr data/dev \
  $baseDir $mfcc_fmllr_dir/log_dev $mfcc_fmllr_dir || exit 1;

steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
  --transform-dir $alignDir data/train_fmllr data/train \
  $baseDir $mfcc_fmllr_dir/log_train $mfcc_fmllr_dir || exit 1;
                            
utils/subset_data_dir_tr_cv.sh  data/train_fmllr $trainTr90 $trainCV || exit 1;

(tail --pid=$$ -F $dnnDir/train_nnet.log 2>/dev/null)& 
$cuda_cmd $dnnDir/train_nnet.log \
steps/train_nnet.sh --use-gpu-id 0 --hid-dim 2048 --hid-layers 5 --learn-rate 0.008 \
$trainTr90 $trainCV data/lang $alignDir $alignDir $dnnDir || exit 1;

steps/decode_nnet.sh --nj $nDecodeJobs --cmd $decode_cmd --config conf/decode_dnn.config \
  --nnet $dnnDir/final.nnet --acwt 0.08 $baseDir/graph data/dev_fmllr $dnnDir/decode || exit 1;

steps/nnet/align.sh --nj $nJobs --cmd $train_cmd data/train_fmllr data/lang \
  $dnnDir $align_dnnDir || exit 1;

steps/nnet/make_denlats.sh --nj $nJobs --cmd $train_cmd --config conf/decode_dnn.config --acwt 0.1 \
  data/train_fmllr data/lang $dnnDir $dnnLatDir || exit 1;

steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 6 --acwt 0.1 --do-smbr true \
  data/train_fmllr data/lang $dnnDir $align_dnnDir $dnnLatDir $dnnMPEDir || exit 1;
  
#decode
for n in 1 2 3 4 5 6; do
  steps/decode_nnet.sh --nj $nDecodeJobs --cmd "$train_cmd" --config conf/decode_dnn.config \
  --nnet $dnnMPEDir/$n.nnet --acwt 0.08 \
  $baseDir/graph data/dev_fmllr $dnnMPEDir/decode_dev_it$n || exit 1;

# End of DNN
