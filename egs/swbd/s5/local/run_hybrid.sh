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
# generate the feats using fMLLRs from decode
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
### Now we can train the Deep Neural Network in a hybrid setup
###
### The fMLLR features are 
###   -spliced, 
###   -decorrelated by LDA 
###   -rescaled by CMVN over dataset
###

#false && \
{ # Train the MLP
dir=exp/tri5a_dnn
ali=exp/tri5a_ali
$cuda_cmd $dir/_train_nnet.log \
  steps/train_nnet.sh --hid-layers 4 --hid-dim 1200 \
  --apply-cmvn false --splice-lr 4 --feat-type lda --lda-dim 350 \
  --learn-rate 0.008 --bunch-size 256 \
  data-fmllr/train_100k_nodup data-fmllr/train_dev data/lang ${ali}_100k_nodup ${ali}_dev $dir || exit 1;
# build graph
$mkgraph_cmd $dir/_mkgraph.log utils/mkgraph.sh data/lang_test $dir $dir/graph || exit 1;
# decode 
steps/decode_nnet.sh --nj 20 --cmd "$decode_cmd" --acwt 0.10 \
  $dir/graph data-fmllr/train_dev $dir/decode_train_dev &&
steps/decode_nnet.sh --nj 20 --cmd "$decode_cmd" --acwt 0.10 \
  $dir/graph data-fmllr/eval2000 $dir/decode_eval2000
}



###
### As the next step we apply MMI training to the hybrid system,
### using Stochastic-GD with per-utterance updates. 
###
### - During the training, part of the training frames is dropped.
###   These are the frames, where the alignment has zero support from the den-lattice
###   (ie. for given state and time from alignment, 
###    there is hard zero in FW-BW posterior matrix)
### - Originally I thought, these frames are there due to search errors when generating 
###   den-lattices, but after implementing tracking of the reference path, the number
###   of dropped frames decreased only by ~10%
### - By inspecting the sph files and annotations, I found out that at least part of
###   such frames are wrongly annotated, so it is reasonable thing to exclude those.
###   Anyway by disabling the frame dropping, the WER improvement was smaller.
###
### - I also experimented with lattice boosting, but no further improvement was observed
###


dir=exp/tri5a_dnn_mmi
srcdir=exp/tri5a_dnn

# MMI starting from system in tri5a_dnn.  Use the same data (100k_nodup)

# First we need to generate lattices and alignments:
#false && \
{
steps/align_nnet.sh --nj 40 --cmd "$train_cmd" \
  data-fmllr/train_100k_nodup data/lang $srcdir exp/tri5a_dnn_ali_100k_nodup || exit 1;
steps/make_denlats_nnet.sh --nj 40 --cmd "$decode_cmd" --config conf/decode.config \
  data-fmllr/train_100k_nodup data/lang $srcdir exp/tri5a_dnn_denlats_100k_nodup  || exit 1;
}

# Now we can re-train the hybrid by the sequence criterion (MMI)
#false && \
{
steps/train_nnet_mmi.sh --cmd "$cuda_cmd" \
  data-fmllr/train_100k_nodup data/lang $srcdir \
  exp/tri5a_dnn_ali_100k_nodup \
  exp/tri5a_dnn_denlats_100k_nodup \
  $dir 
}

# Finally let's decode
#false && \
{
for ITER in 1 2 3 4; do
  steps/decode_nnet.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
    --nnet $dir/${ITER}.nnet \
    exp/tri5a/graph data-fmllr/eval2000 $dir/decode_eval2000_it${ITER} 
done 
}




# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
