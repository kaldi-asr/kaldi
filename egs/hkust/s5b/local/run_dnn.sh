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


# Sequence discriminative training of DNN with sMBR criterion
dir=exp/tri5a_pretrain-dbn_dnn_smbr
srcdir=exp/tri5a_pretrain-dbn_dnn
acwt=0.1
# Create alignment and denominator lattices
{
 steps/align_nnet.sh --nj 20 --cmd "$train_cmd" \
   data-fmllr-tri5a/train data/lang $srcdir ${srcdir}_ali || exit 1;
 steps/make_denlats_nnet.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
   data-fmllr-tri5a/train data/lang $srcdir ${srcdir}_denlats  || exit 1;
}

# DNN training with several iterations of sMBR criterion
{
 steps/train_nnet_mpe.sh --cmd "$cuda_cmd" --num-iters 6 --acwt $acwt --do-smbr true \
  data-fmllr-tri5a/train data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1;
}

