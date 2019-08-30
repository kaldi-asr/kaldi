#!/bin/bash 

# Copyright 2014 QCRI (author: Ahmed Ali)
# Apache 2.0

. ./path.sh
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
nDecodeJobs=80

mer=$1

#train DNN
mfcc_fmllr_dir=mfcc_fmllr_mer$mer
baseDir=exp/mer$mer/tri4
alignDir=exp/mer$mer/tri4_ali
dnnDir=exp/mer$mer/tri4_dnn_2048x5
align_dnnDir=exp/mer$mer/tri4_dnn_2048x5_ali
dnnLatDir=exp/mer$mer/tri4_dnn_2048x5_denlats
dnnMPEDir=exp/mer$mer/tri4_dnn_2048x5_smb

trainTr90=data/train_mer${mer}_tr90
trainCV=data/train_mer${mer}_cv10 

for dev in overlap non_overlap; do
  steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
    --transform-dir $baseDir/decode_$dev data/dev_${dev}_fmllr data/dev_${dev} \
    $baseDir $mfcc_fmllr_dir/log_dev_${dev} $mfcc_fmllr_dir || exit 1;
    cp data/dev_${dev}/reco2file_channel data/dev_${dev}_fmllr/reco2file_channel
    cp data/dev_${dev}/test.stm data/dev_${dev}_fmllr/test.stm
done

steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
  --transform-dir $alignDir data/train_mer${mer}_fmllr data/train_mer$mer \
  $baseDir $mfcc_fmllr_dir/log_train_mer$mer $mfcc_fmllr_dir || exit 1;
                            
utils/subset_data_dir_tr_cv.sh  data/train_mer${mer}_fmllr $trainTr90 $trainCV || exit 1;

(tail --pid=$$ -F $dnnDir/train_mer${mer}_nnet.log 2>/dev/null)& 
$train_cmd --gpu 1 $dnnDir/train_mer${mer}_nnet.log \
steps/train_nnet.sh  --hid-dim 2048 --hid-layers 5 --learn-rate 0.008 \
  $trainTr90 $trainCV data/lang $alignDir $alignDir $dnnDir || exit 1;

for dev in overlap non_overlap; do
  steps/decode_nnet.sh --nj $nDecodeJobs --cmd "$decode_cmd" \
    --config conf/decode_dnn.config --nnet $dnnDir/final.nnet \
    --acwt 0.08 $baseDir/graph data/dev_${dev}_fmllr $dnnDir/decode_$dev &
done

#
steps/nnet/align.sh --nj $nDecodeJobs --cmd "$train_cmd" data/train_mer${mer}_fmllr data/lang \
  $dnnDir $align_dnnDir || exit 1;

steps/nnet/make_denlats.sh --nj $nDecodeJobs --cmd "$train_cmd" --config conf/decode_dnn.config --acwt 0.1 \
  data/train_mer${mer}_fmllr data/lang $dnnDir $dnnLatDir || exit 1;

steps/nnet/train_mpe.sh --cmd "$train_cmd --gpu 1" --num-iters 6 --acwt 0.1 --do-smbr true \
  data/train_mer${mer}_fmllr data/lang $dnnDir $align_dnnDir $dnnLatDir $dnnMPEDir || exit 1;

#decode
for dev in overlap non_overlap; do
  for n in 1 2 3 4 5 6; do
    steps/decode_nnet.sh --nj $nDecodeJobs --cmd "$train_cmd" --config conf/decode_dnn.config \
    --nnet $dnnMPEDir/$n.nnet --acwt 0.08 \
    $baseDir/graph data/dev_${dev}_fmllr $dnnMPEDir/decode_${dev}_it$n || exit 1;
  done
done

echo DNN success
# End of DNN

