#!/bin/bash


. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)


###
### Now we can train the Deep Neural Network in a hybrid setup
###
### The fMLLR features are 
###   -spliced, 
###   -decorrelated by LDA 
###   -rescaled by CMVN over dataset
###

#( # Train the MLP
dir=exp/tri4a_dnn
$cuda_cmd $dir/_train_nnet.log \
  steps/train_nnet.sh --hid-layers 4 --hid-dim 1200 \
  --apply-cmvn false --splice-lr 4 --feat-type lda --lda-dim 300 \
  --learn-rate 0.008 --bunch-size 256 \
  data-fmllr/train data-fmllr/test_test_sup data/lang exp/tri3b exp/tri3b_ali_test $dir || exit 1;

# we can use the graph from the baseline system, tri4a.
# decode .  Note: the dev93 results are not valid as testing results because
# the fMLLR was from the training transcripts.
steps/decode_nnet.sh --nj 10 --cmd "$decode_cmd" --acwt 0.10 \
  exp/tri3b/graph_bg data-fmllr/test exp/tri3b_dnn/decode_bg_test 

# decode with big dictionary.
 utils/mkgraph.sh data/lang_test_bg exp/tri3b_dnn exp/tri3b_dnn/graph_bg || exit 1;

steps/decode_nnet.sh --nj 10 --cmd "$decode_cmd" --acwt 0.10 \
  exp/tri3b_dnn/graph_bg data-fmllr/test exp/tri3b_dnn/decode_bg_test
#)

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done



#from here


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
  data/test data/lang exp/tri3b exp/tri3b_ali_test || exit 1  #dev

###
### As next step we store the fMLLR features, so we can train on them easily
###

gmmdir=exp/tri3b

# dev93 (using alignments)
dir=data-fmllr/test_test_sup
# generate the features
steps/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
   --transform-dir exp/tri3b_ali_test \
   $dir data/test $gmmdir $dir/_log $dir/_data || exit 1

# train si284
# generate the features
dir=data-fmllr/train
steps/make_fmllr_feats.sh --nj 20 --cmd "$train_cmd" \
   --transform-dir exp/tri3b \
   $dir data/train $gmmdir $dir/_log $dir/_data || exit 1

# eval92
dir=data-fmllr/test
steps/make_fmllr_feats.sh --nj 8 --cmd "$train_cmd" \
   --transform-dir exp/tri3b/decode_bg_test \
   $dir data/test $gmmdir $dir/_log $dir/_data || exit 1

dir=data-fmllr/test
steps/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
   --transform-dir exp/tri3b/decode_bg_test \
   $dir data/test $gmmdir $dir/_log $dir/_data || exit 1
)
