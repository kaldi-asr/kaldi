#!/bin/bash


. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)


#false && \
{

###
### First we need to dump the fMLLR features, so we can train on them easily
###

gmmdir=exp/tri5a

# eval2000
dir=data-fmllr/eval2000
steps/make_fmllr_feats.sh --nj 30 --cmd "$train_cmd" \
   --transform-dir exp/tri5a/decode_eval2000 \
   $dir data/eval2000 $gmmdir $dir/_log $dir/_data || exit 1

# train_dev
dir=data-fmllr/train_dev
# we need the alignment
steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train_dev data/lang exp/tri5a exp/tri5a_ali_dev || exit 1
# but, we generate the feats using fMLLRs from decode
steps/make_fmllr_feats.sh --nj 30 --cmd "$train_cmd" \
   --transform-dir exp/tri5a/decode_train_dev \
   $dir data/train_dev $gmmdir $dir/_log $dir/_data || exit 1

# train_100k_nodup
# we need fMLLR transforms, so we run alignment...
steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
  data/train_100k_nodup data/lang exp/tri5a exp/tri5a_ali_100k_nodup || exit 1
# generate the features
dir=data-fmllr/train_100k_nodup
steps/make_fmllr_feats.sh --nj 40 --cmd "$train_cmd" \
   --transform-dir exp/tri5a_ali_100k_nodup \
   $dir data/train_100k_nodup $gmmdir $dir/_log $dir/_data || exit 1
}



###
### Now we can pre-train stack of RBMs
###
#false && \
{ # Pre-train the DBN
dir=exp/tri5a_pretrain-dbn
$cuda_cmd $dir/_pretrain_dbn.log \
  steps/pretrain_dbn.sh data-fmllr/train_100k_nodup $dir || exit 1
}



###
### Now we train the DNN optimizing cross-entropy.
### This will take quite some time.
###

#false && \
{ # Train the MLP
dir=exp/tri5a_pretrain-dbn_dnn
ali=exp/tri5a_ali
feature_transform=exp/tri5a_pretrain-dbn/final.feature_transform
dbn=exp/tri5a_pretrain-dbn/6.dbn
$cuda_cmd $dir/_train_nnet.log \
  steps/train_nnet.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
  data-fmllr/train_100k_nodup data-fmllr/train_dev data/lang ${ali}_100k_nodup ${ali}_dev $dir || exit 1;
# decode (reuse HCLG graph)
steps/decode_nnet.sh --nj 20 --cmd "$decode_cmd" --conf conf/decode_dnn.config --acwt 0.1 \
  exp/tri5a/graph data-fmllr/train_dev $dir/decode_train_dev || exit 1;
steps/decode_nnet.sh --nj 20 --cmd "$decode_cmd" --conf conf/decode_dnn.config --acwt 0.1 \
  exp/tri5a/graph data-fmllr/eval2000 $dir/decode_eval2000 || exit 1;
}



###
### Finally we train using sMBR criterion.
### We do Stochastic-GD with per-utterance updates. 
###
### To get faster convergence, we will re-generate 
### the lattices after 1st epoch of sMBR.
###

dir=exp/tri5a_pretrain-dbn_dnn_smbr
srcdir=exp/tri5a_pretrain-dbn_dnn
acwt=0.1

# First we need to generate lattices and alignments:
#false && \
{
steps/align_nnet.sh --nj 100 --cmd "$train_cmd" \
  data-fmllr/train_100k_nodup data/lang $srcdir ${srcdir}_ali_100k_nodup || exit 1;
steps/make_denlats_nnet.sh --nj 100 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
  data-fmllr/train_100k_nodup data/lang $srcdir ${srcdir}_denlats_100k_nodup  || exit 1;
}
# Now we re-train the hybrid by single iteration of sMBR 
#false && \
{
steps/train_nnet_mpe.sh --cmd "$cuda_cmd" --num-iters 1 --acwt $acwt --do-smbr true \
  data-fmllr/train_100k_nodup data/lang $srcdir \
  ${srcdir}_ali_100k_nodup \
  ${srcdir}_denlats_100k_nodup \
  $dir || exit 1;
}
# Decode
#false && \
{
for ITER in 1; do
  steps/decode_nnet.sh --nj 30 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
    --nnet $dir/${ITER}.nnet \
    exp/tri5a/graph data-fmllr/eval2000 $dir/decode_eval2000_it${ITER} || exit 1
done 
}


###
### Re-generate lattices and run several more iterations of sMBR
###

dir=exp/tri5a_pretrain-dbn_dnn_smbr_iter1-lats
srcdir=exp/tri5a_pretrain-dbn_dnn_smbr
acwt=0.1

# First we need to generate lattices and alignments:
#false && \
{
steps/align_nnet.sh --nj 100 --cmd "$train_cmd" \
  data-fmllr/train_100k_nodup data/lang $srcdir ${srcdir}_ali_100k_nodup || exit 1;
steps/make_denlats_nnet.sh --nj 100 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
  data-fmllr/train_100k_nodup data/lang $srcdir ${srcdir}_denlats_100k_nodup  || exit 1;
}
# Now we re-train the hybrid by several iterations of sMBR 
#false && \
{
steps/train_nnet_mpe.sh --cmd "$cuda_cmd" --num-iters 4 --acwt $acwt --do-smbr true \
  data-fmllr/train_100k_nodup data/lang $srcdir \
  ${srcdir}_ali_100k_nodup \
  ${srcdir}_denlats_100k_nodup \
  $dir || exit 1;
}
# Decode
#false && \
{
for ITER in 1 2 3 4; do
  steps/decode_nnet.sh --nj 30 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
    --nnet $dir/${ITER}.nnet \
    exp/tri5a/graph data-fmllr/eval2000 $dir/decode_eval2000_it${ITER} || exit 1
done 
}



# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
