#!/usr/bin/env bash

# Copyright 2012-2013  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# In this recipe we build DNN in four stages:
# 1) Data preparations : the fMLLR features are stored to disk
# 2) RBM pre-training : in this unsupervised stage we train stack of RBMs, a good starting point for Cross-entropy trainig
# 3) Frame-level cross-entropy training : in this stage the objective is to classify frames correctly.
# 4) Sequence-criterion training : in this stage the objective is to classify the whole sequence correctly,
#     the idea is similar to the 'Discriminative training' in context of GMM-HMMs.


. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)



#false && \
{
gmmdir=exp/tri4b

###
### Generate the alignments of dev93 
### (held-out set for Cross-entropy training)
###
steps/align_fmllr.sh --nj 10 --cmd "$train_cmd" \
  data/test_dev93 data/lang $gmmdir exp/tri4b_ali_dev93 || exit 1

###
### Store the fMLLR features, so we can train on them easily
###

# train si284
# generate the features
dir=data-fmllr-tri4b/train_si284
steps/make_fmllr_feats.sh --nj 20 --cmd "$train_cmd" \
   --transform-dir exp/tri4b_ali_si284 \
   $dir data/train_si284 $gmmdir $dir/_log $dir/_data || exit 1

# eval92
dir=data-fmllr-tri4b/test_eval92
steps/make_fmllr_feats.sh --nj 8 --cmd "$train_cmd" \
   --transform-dir exp/tri4b/decode_tgpr_eval92 \
   $dir data/test_eval92 $gmmdir $dir/_log $dir/_data || exit 1

# dev93 (unsupervised fMLLR)
# held-out set of Cross-entropy training
dir=data-fmllr-tri4b/test_dev93
steps/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
   --transform-dir exp/tri4b/decode_tgpr_dev93 \
   $dir data/test_dev93 $gmmdir $dir/_log $dir/_data || exit 1
}



###
### Now we can pre-train stack of RBMs
###
#false && \
{ # Pre-train the DBN
dir=exp/tri4b_pretrain-dbn
(tail --pid=$$ -F $dir/_pretrain_dbn.log 2>/dev/null)&
$cuda_cmd $dir/_pretrain_dbn.log \
  steps/pretrain_dbn.sh --rbm-iter 3 data-fmllr-tri4b/train_si284 $dir
}



###
### Now we train the DNN optimizing cross-entropy.
### This will take quite some time.
###

#false && \
{ # Train the MLP
dir=exp/tri4b_pretrain-dbn_dnn
ali=exp/tri4b_ali
feature_transform=exp/tri4b_pretrain-dbn/final.feature_transform
dbn=exp/tri4b_pretrain-dbn/6.dbn
(tail --pid=$$ -F $dir/_train_nnet.log 2>/dev/null)& 
$cuda_cmd $dir/_train_nnet.log \
  steps/train_nnet.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
  data-fmllr-tri4b/train_si284 data-fmllr-tri4b/test_dev93 data/lang ${ali}_si284 ${ali}_dev93 $dir || exit 1;
# decode with 'big-dictionary' (reuse HCLG graph)
steps/decode_nnet.sh --nj 10 --cmd "$decode_cmd" --acwt 0.10 --config conf/decode_dnn.config \
  exp/tri4b/graph_bd_tgpr data-fmllr-tri4b/test_dev93 $dir/decode_bd_tgpr_dev93 || exit 1;
steps/decode_nnet.sh --nj 8 --cmd "$decode_cmd" --acwt 0.10 --config conf/decode_dnn.config \
  exp/tri4b/graph_bd_tgpr data-fmllr-tri4b/test_eval92 $dir/decode_bd_tgpr_eval92 || exit 1;
}



###
### Finally we train using sMBR criterion.
### We do Stochastic-GD with per-utterance updates. 
###
### To get faster convergence, we will re-generate 
### the lattices after 1st epoch of sMBR.
###

dir=exp/tri4b_pretrain-dbn_dnn_smbr
srcdir=exp/tri4b_pretrain-dbn_dnn
acwt=0.10

# First we need to generate lattices and alignments:
#false && \
{
steps/align_nnet.sh --nj 100 --cmd "$train_cmd" \
  data-fmllr-tri4b/train_si284 data/lang $srcdir ${srcdir}_ali_si284 || exit 1;
steps/make_denlats_nnet.sh --nj 100 --cmd "$decode_cmd" \
  --config conf/decode_dnn.config --acwt $acwt \
  data-fmllr-tri4b/train_si284 data/lang $srcdir ${srcdir}_denlats_si284  || exit 1;
}
# Now we re-train the hybrid by single iteration of sMBR 
#false && \
{
steps/train_nnet_mpe.sh --cmd "$cuda_cmd" --num-iters 1 --acwt $acwt --do-smbr true \
  data-fmllr-tri4b/train_si284 data/lang $srcdir \
  ${srcdir}_ali_si284 ${srcdir}_denlats_si284 $dir || exit 1
}
# Decode
#false && \
{
for ITER in 1; do
  # decode dev93 with big dict graph_bd_tgpr
  steps/decode_nnet.sh --nj 10 --cmd "$decode_cmd" --config conf/decode_dnn.config \
    --nnet $dir/${ITER}.nnet --acwt $acwt \
    exp/tri4b/graph_bd_tgpr data-fmllr-tri4b/test_dev93 $dir/decode_dev93_bd_tgpr_it${ITER} || exit 1
  # decode eval92 with big dict graph_bd_tgpr
  steps/decode_nnet.sh --nj 8 --cmd "$decode_cmd" --config conf/decode_dnn.config \
    --nnet $dir/${ITER}.nnet --acwt $acwt \
    exp/tri4b/graph_bd_tgpr data-fmllr-tri4b/test_eval92 $dir/decode_eval92_bd_tgpr_it${ITER} || exit 1
done 
}


###
### Re-generate lattices and run several more iterations of sMBR
###

dir=exp/tri4b_pretrain-dbn_dnn_smbr_iter1-lats
srcdir=exp/tri4b_pretrain-dbn_dnn_smbr
acwt=0.10

# First we need to generate lattices and alignments:
#false && \
{
steps/align_nnet.sh --nj 100 --cmd "$train_cmd" \
  data-fmllr-tri4b/train_si284 data/lang $srcdir ${srcdir}_ali_si284 || exit 1;
steps/make_denlats_nnet.sh --nj 100 --cmd "$decode_cmd" \
  --config conf/decode_dnn.config --acwt $acwt \
  data-fmllr-tri4b/train_si284 data/lang $srcdir ${srcdir}_denlats_si284  || exit 1;
}
# Now we re-train the hybrid by several iterations of sMBR 
#false && \
{
steps/train_nnet_mpe.sh --cmd "$cuda_cmd" --num-iters 4 --acwt $acwt --do-smbr true \
  data-fmllr-tri4b/train_si284 data/lang $srcdir \
  ${srcdir}_ali_si284 ${srcdir}_denlats_si284 $dir 
}
# Decode
#false && \
{
for ITER in 1 2 3 4; do
  # decode dev93 with big dict graph_bd_tgpr
  steps/decode_nnet.sh --nj 10 --cmd "$decode_cmd" --config conf/decode_dnn.config \
    --nnet $dir/${ITER}.nnet --acwt $acwt \
    exp/tri4b/graph_bd_tgpr data-fmllr-tri4b/test_dev93 $dir/decode_dev93_bd_tgpr_it${ITER} || exit 1
  # decode eval92 with big dict graph_bd_tgpr
  steps/decode_nnet.sh --nj 8 --cmd "$decode_cmd" --config conf/decode_dnn.config \
    --nnet $dir/${ITER}.nnet --acwt $acwt \
    exp/tri4b/graph_bd_tgpr data-fmllr-tri4b/test_eval92 $dir/decode_eval92_bd_tgpr_it${ITER} || exit 1
done 
}


# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
