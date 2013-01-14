#!/bin/bash


. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)



#false && \
(

###
### First we need to generate the alignments, 
###
### these are used as DNN training targets,
### also the fMLLR transforms are needed 
###

# We don't really need the alignment directory, as tri4a was trained
# on si284 and already contains alignments.
#steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
#  data/train_si284 data/lang exp/tri4a exp/tri4a_ali_si284 || exit 1

steps/align_fmllr.sh --nj 10 --cmd "$train_cmd" \
  data/test_dev93 data/lang exp/tri4a exp/tri4a_ali_dev93 || exit 1

###
### As next step we store the fMLLR features, so we can train on them easily
###

gmmdir=exp/tri4a

# dev93 (using alignments)
dir=data-fmllr/test_dev93_sup
# generate the features
steps/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
   --transform-dir exp/tri4a_ali_dev93 \
   $dir data/test_dev93 $gmmdir $dir/_log $dir/_data || exit 1

# train si284
# generate the features
dir=data-fmllr/train_si284
steps/make_fmllr_feats.sh --nj 20 --cmd "$train_cmd" \
   --transform-dir exp/tri4a \
   $dir data/train_si284 $gmmdir $dir/_log $dir/_data || exit 1

# eval92
dir=data-fmllr/test_eval92
steps/make_fmllr_feats.sh --nj 8 --cmd "$train_cmd" \
   --transform-dir exp/tri4a/decode_tgpr_eval92 \
   $dir data/test_eval92 $gmmdir $dir/_log $dir/_data || exit 1

dir=data-fmllr/test_dev93
steps/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
   --transform-dir exp/tri4a/decode_tgpr_dev93 \
   $dir data/test_dev93 $gmmdir $dir/_log $dir/_data || exit 1


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
dir=exp/tri4a_dnn
$cuda_cmd $dir/_train_nnet.log \
  steps/train_nnet.sh --hid-layers 4 --hid-dim 1200 \
  --apply-cmvn false --splice-lr 4 --feat-type lda --lda-dim 300 \
  --learn-rate 0.008 --bunch-size 256 \
  data-fmllr/train_si284 data-fmllr/test_dev93_sup data/lang exp/tri4a exp/tri4a_ali_dev93 $dir || exit 1;
# we can use the graph from the baseline system, tri4a.
# decode 
steps/decode_nnet.sh --nj 10 --cmd "$decode_cmd" --acwt 0.10 \
  exp/tri4a/graph_tgpr data-fmllr/test_dev93 exp/tri4a_dnn/decode_tgpr_dev93 &&
steps/decode_nnet.sh --nj 8 --cmd "$decode_cmd" --acwt 0.10 \
  exp/tri4a/graph_tgpr data-fmllr/test_eval92 exp/tri4a_dnn/decode_tgpr_eval92
)



# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
