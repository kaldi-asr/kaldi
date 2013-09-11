#!/bin/bash

# Karel Vesely (Brno University of Technology) 's style DNN training 
#
# Prepared by Ricky Chan Ho Yin (Hong Kong University of Science and Technology)
#
# Apache License, 2.0

. cmd.sh

. path.sh

ulimit -u 10000

gmmdir=exp/tri5a

{
# test data preparation
dir=data-fmllr-tri5a/test
  steps/make_fmllr_feats.sh --nj 2 --cmd "$train_cmd" \
     --transform-dir exp/tri5a/decode_eval \
     $dir data/eval $gmmdir $dir/_log $dir/_data || exit 1

# train data preparation
dir=data-fmllr-tri5a/train
  steps/make_fmllr_feats.sh --nj 25 --cmd "$train_cmd" \
    --transform-dir exp/tri5a_ali_dt100k \
    $dir data/train $gmmdir $dir/_log $dir/_data || exit 1

# split the data : 90% train 10% cross-validation (held-out)
  utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10 || exit 1
}


{ # Pre-train the DBN with Restricted Boltzmann Machine
dir=exp/tri5a_pretrain-dbn
 (tail --pid=$$ -F $dir/_pretrain_dbn.log 2>/dev/null)&
 $cuda_cmd $dir/_pretrain_dbn.log \
   steps/pretrain_dbn.sh --hid-dim 1024 --rbm-iter 20 data-fmllr-tri5a/train $dir || exit 1;
}


# Frame-level cross-entropy DNN training
{ # Train the MLP
dir=exp/tri5a_pretrain-dbn_dnn
ali=exp/tri5a_ali_dt100k
feature_transform=exp/tri5a_pretrain-dbn/final.feature_transform
dbn=exp/tri5a_pretrain-dbn/6.dbn
(tail --pid=$$ -F $dir/_train_nnet.log 2>/dev/null)& 
$cuda_cmd $dir/_train_nnet.log \
   steps/train_nnet.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
   data-fmllr-tri5a/train_tr90 data-fmllr-tri5a/train_cv10 data/lang $ali $ali $dir || exit 1;
}


