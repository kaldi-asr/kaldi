#!/bin/bash


. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)


#false && \
(

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
# we need fMLLR transforms, so we run alignment...
steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" \
  data/train_dev data/lang exp/tri5a exp/tri5a_ali_dev || exit 1
#generate the feats
steps/make_fmllr_feats.sh --nj 20 --cmd "$train_cmd" \
   --transform-dir exp/tri5a_ali_dev \
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
)


###
### Now we can train the Deep Neural Network in a hybrid setup
###
### The fMLLR features are 
###   -spliced, 
###   -decorrelated by LDA 
###   -rescaled by CMVN over dataset
###

( # Train the MLP
dir=exp/tri5a_dnn
ali=exp/tri5a_ali
$cuda_cmd $dir/_train_nnet.log \
  steps/train_nnet.sh --hid-layers 4 --hid-dim 1200 \
  --apply-cmvn false --splice-lr 4 --feat-type lda --lda-dim 300 \
  --learn-rate 0.008 --bunch-size 256 \
  data-fmllr/train_100k_nodup data-fmllr/train_dev data/lang ${ali}_100k_nodup ${ali}_dev $dir || exit 1;
# build graph
$mkgraph_cmd $dir/_mkgraph.log utils/mkgraph.sh data/lang_test $dir $dir/graph || exit 1;
# decode 
steps/decode_nnet.sh --nj 20 --cmd "$decode_cmd" --acwt 0.10 \
  $dir/graph data-fmllr/train_dev $dir/decode_train_dev &&
steps/decode_nnet.sh --nj 20 --cmd "$decode_cmd" --acwt 0.10 \
  $dir/graph data-fmllr/eval2000 $dir/decode_eval2000
)



# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
