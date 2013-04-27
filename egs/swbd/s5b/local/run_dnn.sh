#!/bin/bash


. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)


#false && \
{

###
### First we need to dump the fMLLR features, so we can train on them easily
###

gmmdir=exp/tri4b

# eval2000
dir=data-fmllr-tri4b/eval2000
steps/make_fmllr_feats.sh --nj 30 --cmd "$train_cmd" \
   --transform-dir exp/tri4b/decode_eval2000_sw1_fsh_tgpr \
   $dir data/eval2000 $gmmdir $dir/_log $dir/_data || exit 1

# train_dev
dir=data-fmllr-tri4b/train_dev
# we need alignment
steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" \
  data/train_dev data/lang exp/tri4b exp/tri4b_ali_dev || exit 1
# we need fMLLR transforms, so we run decoding...
steps/decode_fmllr.sh --nj 20 --cmd "$decode_cmd" \
   --config conf/decode.config  exp/tri4b/graph_sw1_fsh_tgpr \
   data/train_dev exp/tri4b/decode_train_dev_sw1_fsh_tgpr || exit 1
#generate the feats
steps/make_fmllr_feats.sh --nj 20 --cmd "$train_cmd" \
   --transform-dir exp/tri4b/decode_train_dev_sw1_fsh_tgpr \
   $dir data/train_dev $gmmdir $dir/_log $dir/_data || exit 1

# train_nodup
# we need fMLLR transforms, so we run alignment...
#
# ALREADY IN run_edit.sh:
#steps/align_fmllr.sh --nj 60 --cmd "$train_cmd" \
#  data/train_nodup data/lang exp/tri4b exp/tri4b_ali_all || exit 1
#
# generate the features
dir=data-fmllr-tri4b/train_nodup
steps/make_fmllr_feats.sh --nj 60 --cmd "$train_cmd" \
   --transform-dir exp/tri4b_ali_all \
   $dir data/train_nodup $gmmdir $dir/_log $dir/_data || exit 1
}



###
### Now we can pre-train stack of RBMs
###
#false && \
{ # Pre-train the DBN
dir=exp/tri4b_pretrain-dbn
$cuda_cmd $dir/_pretrain_dbn.log \
  steps/pretrain_dbn.sh data-fmllr-tri4b/train_nodup $dir
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
$cuda_cmd $dir/_train_nnet.log \
  steps/train_nnet.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
  data-fmllr-tri4b/train_nodup data-fmllr-tri4b/train_dev data/lang ${ali}_all ${ali}_dev $dir || exit 1;
# decode (reuse HCLG graph)
steps/decode_nnet.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.0833 \
  exp/tri4b/graph_sw1_fsh_tgpr data-fmllr-tri4b/train_dev $dir/decode_train_dev_sw1_fsh_tgpr || exit 1;
steps/decode_nnet.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.0833 \
  exp/tri4b/graph_sw1_fsh_tgpr data-fmllr-tri4b/eval2000 $dir/decode_eval2000_sw1_fsh_tgpr || exit 1;
# rescore eval2000 with trigram sw1_fsh
steps/lmrescore.sh --mode 3 --cmd "$decodebig_cmd" data/lang_sw1_fsh_tgpr data/lang_sw1_fsh_tg data/eval2000 \
  $dir/decode_eval2000_sw1_fsh_tgpr $dir/decode_eval2000_sw1_fsh_tg.3 || exit 1 
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
acwt=0.08333

# First we need to generate lattices and alignments:
#false && \
{
steps/align_nnet.sh --nj 250 --cmd "$train_cmd" \
  data-fmllr-tri4b/train_nodup data/lang $srcdir ${srcdir}_ali_all || exit 1;
steps/make_denlats_nnet.sh --nj 250 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
  data-fmllr-tri4b/train_nodup data/lang $srcdir ${srcdir}_denlats_all  || exit 1;
}
# Now we re-train the hybrid by single iteration of sMBR 
#false && \
{
steps/train_nnet_mpe.sh --cmd "$cuda_cmd" --num-iters 1 --acwt $acwt --do-smbr=true \
  data-fmllr-tri4b/train_nodup data/lang $srcdir \
  ${srcdir}_ali_all \
  ${srcdir}_denlats_all \
  $dir 
}
# Decode
#false && \
{
for ITER in 1; do
  # decode eval2000 with pruned trigram sw1_fsh_tgpr
  steps/decode_nnet.sh --nj 30 --cmd "$decode_cmd" --config conf/decode_dnn.config \
    --nnet $dir/${ITER}.nnet --acwt $acwt \
    exp/tri4b/graph_sw1_fsh_tgpr data-fmllr-tri4b/eval2000 $dir/decode_eval2000_sw1_fsh_tgpr_it${ITER}
  # rescore eval2000 with trigram sw1_fsh
  steps/lmrescore.sh --mode 3 --cmd "$decodebig_cmd" data/lang_sw1_fsh_tgpr data/lang_sw1_fsh_tg data/eval2000 \
    $dir/decode_eval2000_sw1_fsh_tgpr_it${ITER} $dir/decode_eval2000_sw1_fsh_tg.3_it${ITER} || exit 1 
done 
}


###
### Re-generate lattices and run several more iterations of sMBR
###

dir=exp/tri4b_pretrain-dbn_dnn_smbr_iter1-lats
srcdir=exp/tri4b_pretrain-dbn_dnn_smbr
acwt=0.08333

# First we need to generate lattices and alignments:
#false && \
{
steps/align_nnet.sh --nj 250 --cmd "$train_cmd" \
  data-fmllr-tri4b/train_nodup data/lang $srcdir ${srcdir}_ali_all || exit 1;
steps/make_denlats_nnet.sh --nj 250 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
  data-fmllr-tri4b/train_nodup data/lang $srcdir ${srcdir}_denlats_all  || exit 1;
}
# Now we re-train the hybrid by several iterations of sMBR 
#false && \
{
steps/train_nnet_mpe.sh --cmd "$cuda_cmd" --num-iters 2 --acwt $acwt --do-smbr=true \
  data-fmllr-tri4b/train_nodup data/lang $srcdir \
  ${srcdir}_ali_all \
  ${srcdir}_denlats_all \
  $dir 
}
# Decode
#false && \
{
for ITER in 1 2; do
  # decode eval2000 with pruned trigram sw1_fsh_tgpr
  steps/decode_nnet.sh --nj 30 --cmd "$decode_cmd" --config conf/decode_dnn.config \
    --nnet $dir/${ITER}.nnet --acwt $acwt \
    exp/tri4b/graph_sw1_fsh_tgpr data-fmllr-tri4b/eval2000 $dir/decode_eval2000_sw1_fsh_tgpr_it${ITER}
  # rescore eval2000 with trigram sw1_fsh
  steps/lmrescore.sh --mode 3 --cmd "$decodebig_cmd" data/lang_sw1_fsh_tgpr data/lang_sw1_fsh_tg data/eval2000 \
    $dir/decode_eval2000_sw1_fsh_tgpr_it${ITER} $dir/decode_eval2000_sw1_fsh_tg.3_it${ITER} || exit 1 
done 
}



# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
