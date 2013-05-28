#!/bin/bash


. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)



###
### First we need to dump the fMLLR features, so we can train on them easily
###

#false && \
{
gmmdir=exp/tri3b

# test
dir=data-fmllr-tri3b/test
steps/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
   --transform-dir exp/tri3b/decode \
   $dir data/test $gmmdir $dir/_log $dir/_data || exit 1

# train
dir=data-fmllr-tri3b/train
steps/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
   --transform-dir exp/tri4b_ali \
   $dir data/train $gmmdir $dir/_log $dir/_data || exit 1

# split the data : 90% train 10% cross-validation (held-out)
utils/subset_data_dir_tr90_cv10.sh $dir ${dir}_tr90 ${dir}_cv10 || exit 1
}



###
### Now we can pre-train stack of RBMs
### (small database, smaller DNN)
###

#false && \
{ # Pre-train the DBN
dir=exp/tri3b_pretrain-dbn
(tail --pid=$$ -F $dir/_pretrain_dbn.log)&
$cuda_cmd $dir/_pretrain_dbn.log \
  steps/pretrain_dbn.sh --hid-dim 1024 --rbm-iter 20 data-fmllr-tri3b/train $dir || exit 1;
}



###
### Now we train the DNN optimizing cross-entropy.
###

#false && \
{ # Train the MLP
dir=exp/tri3b_pretrain-dbn_dnn
ali=exp/tri3b_ali
feature_transform=exp/tri3b_pretrain-dbn/final.feature_transform
dbn=exp/tri3b_pretrain-dbn/6.dbn
(tail --pid=$$ -F $dir/_train_nnet.log)& 
$cuda_cmd $dir/_train_nnet.log \
  steps/train_nnet.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
  data-fmllr-tri3b/train_tr90 data-fmllr-tri3b/train_cv10 data/lang $ali $ali $dir || exit 1;
# decode (reuse HCLG graph)
steps/decode_nnet.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
  exp/tri3b/graph data-fmllr-tri3b/test $dir/decode || exit 1;
}



###
### Finally we train using sMBR criterion.
### We do Stochastic-GD with per-utterance updates. 
### Use acwt 0.1, although it is not the best-WER value.
###

dir=exp/tri3b_pretrain-dbn_dnn_smbr
srcdir=exp/tri3b_pretrain-dbn_dnn
acwt=0.1

# First we need to generate lattices and alignments:
#false && \
{
steps/align_nnet.sh --nj 20 --cmd "$train_cmd" \
  data-fmllr-tri3b/train data/lang $srcdir ${srcdir}_ali || exit 1;
steps/make_denlats_nnet.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
  data-fmllr-tri3b/train data/lang $srcdir ${srcdir}_denlats  || exit 1;
}
# Now we re-train the DNN by 6 iterations of sMBR 
#false && \
{
steps/train_nnet_mpe.sh --cmd "$cuda_cmd" --num-iters 6 --acwt $acwt --do-smbr true \
  data-fmllr-tri3b/train data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
}
# Decode
#false && \
{
for ITER in 1 2 3 4 5 6; do
  # decode
  steps/decode_nnet.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config \
    --nnet $dir/${ITER}.nnet --acwt $acwt \
    exp/tri3b/graph data-fmllr-tri3b/test $dir/decode_it${ITER} || exit 1
done 
}


# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
