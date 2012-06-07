#!/bin/bash

# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.
# Caution: some of the graph creation steps use quite a bit of memory, so you
# might want to run this script on a machine that has plenty of memory.


# The next command line is an example; you have to give the script
# command-line arguments corresponding to the WSJ disks from LDC.  
# Another example set of command line arguments is
# /ais/gobi2/speech/WSJ/*/??-{?,??}.?
#  These must be absolute,  not relative, pathnames.

local/wsj_data_prep.sh /mnt/matylda2/data/WSJ?/??-{?,??}.? || exit 1;
#local/wsj_data_prep.sh  /export/corpora5/LDC/LDC{93S6,94S13}B/??-{?,??}.? || exit 1;

local/wsj_prepare_dict.sh || exit 1;

local/wsj_format_data.sh || exit 1;

# We suggest to run the next three commands in the background,
# as they are not a precondition for the system building and
# most of the tests: these commands build a dictionary
# containing many of the OOVs in the WSJ LM training data,
# and an LM trained directly on that data (i.e. not just
# copying the arpa files from the disks from LDC).
(
# on CSLP: local/wsj_extend_dict.sh /export/corpora5/LDC/LDC94S13B/13-32.1/ && \
 local/wsj_extend_dict.sh /mnt/matylda2/data/WSJ1/13-32.1  && \
 local/wsj_prepare_local_dict.sh && \
 local/wsj_train_lms.sh && \
 local/wsj_format_data_local.sh
) &


# Now make MFCC features.
# featdir should be some place with a largish disk where you
# want to store MFCC features.
featdir=$PWD/exp/kaldi_wsj_feats
for x in test_eval92 test_eval93 test_dev93 train_si284; do 
  steps/make_mfcc.sh data/$x exp/make_mfcc/$x $featdir/mfcc 2
  steps/make_fbank.sh data/$x exp/make_fbank/$x $featdir/fbank 2
  ln -s $PWD/data/$x/feats.scp.mfcc data/$x/feats.scp #select mfcc as default
done


mkdir data/train_si84
for x in feats.scp feats.scp.fbank text utt2spk wav.scp; do
  head -7138 data/train_si284/$x > data/train_si84/$x
done
scripts/utt2spk_to_spk2utt.pl data/train_si84/utt2spk > data/train_si84/spk2utt || exit 1;
scripts/filter_scp.pl data/train_si84/spk2utt data/train_si284/spk2gender > data/train_si84/spk2gender || exit 1;

# Now make subset with the shortest 2k utterances from si-84.
scripts/subset_data_dir.sh --shortest data/train_si84 2000 data/train_si84_2kshort || exit 1;

# Now make subset with half of the data from si-84.
scripts/subset_data_dir.sh data/train_si84 3500 data/train_si84_half || exit 1;

# you can change these commands to just run.pl to make them run
# locally, but in that case you should change the num-jobs to
# the #cpus on your machine or fewer.
decode_cmd="queue.pl -q all.q@@blade -l ram_free=1200M,mem_free=1200M"
train_cmd="queue.pl -q all.q@@blade -l ram_free=700M,mem_free=700M"
cuda_cmd="queue.pl -q long.q@@pco203 -l gpu=1"

# put the scripts to path
source path.sh



######################################################
### WE FIRST NEED THE GMM MODELS TO GET ALIGNMENTS ###

# Train mono0a with 2k shortest segments of si84 data.
steps/train_mono.sh --num-jobs 10 --cmd "$train_cmd" \
  data/train_si84_2kshort data/lang exp/mono0a || exit 1;
# Decode mono0a
(
scripts/mkgraph.sh --mono data/lang_test_tgpr exp/mono0a exp/mono0a/graph_tgpr || exit 1;
scripts/decode.sh --cmd "$decode_cmd" steps/decode_deltas.sh exp/mono0a/graph_tgpr data/test_dev93 exp/mono0a/decode_tgpr_dev93 || exit 1;
scripts/decode.sh --cmd "$decode_cmd" steps/decode_deltas.sh exp/mono0a/graph_tgpr data/test_eval92 exp/mono0a/decode_tgpr_eval92 || exit 1;
)&

# This queue option will be supplied to all alignment
# and training scripts.  Note: you have to supply the same num-jobs
# to the alignment and training scripts, as the archives are split
# up in this way.


# Train mono1a - a bigger monophone system trained with si84 data, 
# (more accurate monophone alignments will be useful for tandem MLP training)
# Align mono0a with si84 data.
steps/align_deltas.sh --num-jobs 10 --cmd "$train_cmd" \
   data/train_si84 data/lang exp/mono0a exp/mono0a_ali_si84
# Train mono1a with si84 data.
steps/train_mono_ali.sh --num-jobs 10 --cmd "$train_cmd" \
  data/train_si84 data/lang exp/mono0a_ali_si84 exp/mono1a

# 
# Decode mono1a
(
scripts/mkgraph.sh --mono data/lang_test_tgpr exp/mono1a exp/mono1a/graph_tgpr
scripts/decode.sh --cmd "$decode_cmd" steps/decode_deltas.sh exp/mono1a/graph_tgpr data/test_dev93 exp/mono1a/decode_tgpr_dev93
scripts/decode.sh --cmd "$decode_cmd" steps/decode_deltas.sh exp/mono1a/graph_tgpr data/test_eval92 exp/mono1a/decode_tgpr_eval92
)&
# Align mono1a with si84 data
steps/align_deltas.sh --num-jobs 10 --cmd "$train_cmd" \
   data/train_si84 data/lang exp/mono1a exp/mono1a_ali_si84
# Align mono1a with si284 data
steps/align_deltas.sh --num-jobs 10 --cmd "$train_cmd" \
   data/train_si284 data/lang exp/mono1a exp/mono1a_ali_si284
# Align mono1a with dev93 data
steps/align_deltas.sh --num-jobs 10 --cmd "$train_cmd" \
   data/test_dev93 data/lang exp/mono1a exp/mono1a_ali_dev93


# Proceed in training the triphone baselines:
# Align mono0a with half of si84 data.
steps/align_deltas.sh --num-jobs 10 --cmd "$train_cmd" \
   data/train_si84_half data/lang exp/mono0a exp/mono0a_ali_si84_half || exit 1;

# Train tri1a, which is deltas + delta-deltas, on half of si84 data.
steps/train_deltas.sh --num-jobs 10 --cmd "$train_cmd" \
    2000 10000 data/train_si84_half data/lang exp/mono0a_ali_si84_half exp/tri1 || exit 1;

# Build graph
wait; # or the mono mkgraph.sh might be writing 
# data/lang_test_tgpr/tmp/LG.fst which will cause this to fail.
scripts/mkgraph.sh data/lang_test_tgpr exp/tri1 exp/tri1/graph_tgpr || exit 1;
# Decode dev93, eval92
scripts/decode.sh --cmd "$decode_cmd" steps/decode_deltas.sh exp/tri1/graph_tgpr data/test_dev93 exp/tri1/decode_tgpr_dev93 || exit 1;
scripts/decode.sh --cmd "$decode_cmd" steps/decode_deltas.sh exp/tri1/graph_tgpr data/test_eval92 exp/tri1/decode_tgpr_eval92 || exit 1;

# Align tri1 system with si84 data.
steps/align_deltas.sh --num-jobs 10 --cmd "$train_cmd" \
  data/train_si84 data/lang exp/tri1 exp/tri1_ali_si84 || exit 1;


# Train tri2a, which is deltas + delta-deltas, on si84 data.
# this is hot candidate for NN TARGET ALIGNMENTS
numleavesL=(2500)
for numleaves in ${numleavesL[@]}; do
  # Train
  steps/train_deltas.sh  --num-jobs 10 --cmd "$train_cmd" \
    $numleaves 15000 data/train_si84 data/lang exp/tri1_ali_si84 exp/tri2a-$numleaves || exit 1;
  # Decode
  scripts/mkgraph.sh data/lang_test_tgpr exp/tri2a-$numleaves exp/tri2a-$numleaves/graph_tgpr || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" steps/decode_deltas.sh exp/tri2a-$numleaves/graph_tgpr data/test_dev93 exp/tri2a-$numleaves/decode_tgpr_dev93 || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" steps/decode_deltas.sh exp/tri2a-$numleaves/graph_tgpr data/test_eval92 exp/tri2a-$numleaves/decode_tgpr_eval92 || exit 1;

  # Align tri2a-???? system with si84 data.
  steps/align_deltas.sh  --num-jobs 10 --cmd "$train_cmd" \
    --use-graphs data/train_si84 data/lang exp/tri2a-$numleaves exp/tri2a-${numleaves}_ali_si84
  # Align tri2a-???? system with si284 data.
  steps/align_deltas.sh  --num-jobs 10 --cmd "$train_cmd" \
    data/train_si284 data/lang exp/tri2a-$numleaves exp/tri2a-${numleaves}_ali_si284
done




#####################################
### HERE START THE MLP EXPERIMENTS
###
#train the MLPs..., first on the si84 set



#Train so far best working pure hybrid system
numleaves=2500
modelsize=3000000
lrate=0.004
( $cuda_cmd exp/tri2a-${numleaves}_nnet4L_modelsize${modelsize}_lrate${lrate}/_train_nnet.log \
    steps/train_nnet_MLP4.sh --model-size $modelsize --lrate $lrate data/train_si84 data/lang exp/tri2a-${numleaves}_ali_si84 exp/tri2a-${numleaves}_nnet4L_modelsize${modelsize}_lrate${lrate} || exit 1;
  # build graph
  scripts/mkgraph.sh data/lang_test_tgpr exp/tri2a-${numleaves} exp/tri2a-${numleaves}_nnet4L_modelsize${modelsize}_lrate${lrate}/graph_tgpr || exit 1;
  # decode 
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.1428" steps/decode_nnet.sh exp/tri2a-${numleaves}_nnet4L_modelsize${modelsize}_lrate${lrate}/graph_tgpr data/test_dev93 exp/tri2a-${numleaves}_nnet4L_modelsize${modelsize}_lrate${lrate}/decode_tgpr_dev93 || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.1428" steps/decode_nnet.sh exp/tri2a-${numleaves}_nnet4L_modelsize${modelsize}_lrate${lrate}/graph_tgpr data/test_eval92 exp/tri2a-${numleaves}_nnet4L_modelsize${modelsize}_lrate${lrate}/decode_tgpr_eval92 || exit 1;
) &



#Train so far best working pure hybrid system on big dataset si284
numleaves=2500
modelsize=3000000
lrate=0.004
( $cuda_cmd exp/tri2a-${numleaves}_nnet4L_modelsize${modelsize}_lrate${lrate}_si284/_train_nnet.log \
    steps/train_nnet_MLP4.sh --model-size $modelsize --lrate $lrate data/train_si284 data/lang exp/tri2a-${numleaves}_ali_si284 exp/tri2a-${numleaves}_nnet4L_modelsize${modelsize}_lrate${lrate}_si284 || exit 1;
  # build graph
  scripts/mkgraph.sh data/lang_test_tgpr exp/tri2a-${numleaves} exp/tri2a-${numleaves}_nnet4L_modelsize${modelsize}_lrate${lrate}_si284/graph_tgpr || exit 1;
  # decode 
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.1428" steps/decode_nnet.sh exp/tri2a-${numleaves}_nnet4L_modelsize${modelsize}_lrate${lrate}_si284/graph_tgpr data/test_dev93 exp/tri2a-${numleaves}_nnet4L_modelsize${modelsize}_lrate${lrate}_si284/decode_tgpr_dev93 || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.1428" steps/decode_nnet.sh exp/tri2a-${numleaves}_nnet4L_modelsize${modelsize}_lrate${lrate}_si284/graph_tgpr data/test_eval92 exp/tri2a-${numleaves}_nnet4L_modelsize${modelsize}_lrate${lrate}_si284/decode_tgpr_eval92 || exit 1;
) &



#Train so far best working tandem system:
#TANDEM SYSTEM "B"
#trained on linear BN-features, on monophone targets, use dev93 for NN training stopping
#
# Train the nnet
( $cuda_cmd exp/mono1a_dev93_nnet-linBN-5L/_train_nnet.log \
  steps/train_nnet_dev_MLP5-linBN.sh --lrate 0.00025 data/train_si84 data/test_dev93 data/lang exp/mono1a_ali_si84 exp/mono1a_ali_dev93 exp/mono1a_dev93_nnet-linBN-5L || exit 1;
#Dump the BN-features
nndir=exp/mono1a_dev93_nnet-linBN-5L
bnroot=$PWD/exp/make_bnfeats_$(basename $nndir)
for x in test_eval92 test_eval93 test_dev93 train_si84; do
  steps/make_bnfeats.sh --bn-dim 30 data/$x $nndir $bnroot/$x $featdir/bnfeats_$(basename $nndir) 4
done
#re-train the GMMs to new feature space
numleaves=2500
# Train
# the best alignment options seem to be: _tr0.5_aw0.05_sl0.1
steps/train_lda_mllt_bnfeats_singlepass_tri2a.sh --num-jobs 10 --cmd "$train_cmd" \
  --scale-opts "--transition-scale=0.5 --acoustic-scale=0.05 --self-loop-scale=0.1" \
  $numleaves 15000 $bnroot/train_si84 data/train_si84/ data/lang exp/tri2a-${numleaves}_ali_si84 exp/mono1a_dev93_nnet-linBN-5L.gmm-${numleaves}_singlepass_tri2a_tuneali || exit 1;
# Decode
scripts/mkgraph.sh data/lang_test_tgpr exp/mono1a_dev93_nnet-linBN-5L.gmm-${numleaves}_singlepass_tri2a_tuneali exp/mono1a_dev93_nnet-linBN-5L.gmm-${numleaves}_singlepass_tri2a_tuneali/graph_tgpr || exit 1;
scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.05 --scale-beams 1.25" steps/decode_lda_mllt_bnfeats.sh exp/mono1a_dev93_nnet-linBN-5L.gmm-${numleaves}_singlepass_tri2a_tuneali/graph_tgpr $bnroot/test_eval92 exp/mono1a_dev93_nnet-linBN-5L.gmm-${numleaves}_singlepass_tri2a_tuneali/decode_tgpr_eval92 || exit 1;
scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.05 --scale-beams 1.25" steps/decode_lda_mllt_bnfeats.sh exp/mono1a_dev93_nnet-linBN-5L.gmm-${numleaves}_singlepass_tri2a_tuneali/graph_tgpr $bnroot/test_eval93 exp/mono1a_dev93_nnet-linBN-5L.gmm-${numleaves}_singlepass_tri2a_tuneali/decode_tgpr_eval93 || exit 1;
scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.05 --scale-beams 1.25" steps/decode_lda_mllt_bnfeats.sh exp/mono1a_dev93_nnet-linBN-5L.gmm-${numleaves}_singlepass_tri2a_tuneali/graph_tgpr $bnroot/test_dev93 exp/mono1a_dev93_nnet-linBN-5L.gmm-${numleaves}_singlepass_tri2a_tuneali/decode_tgpr_dev93 || exit 1;
) &

wait


#Train so far best working pure hybrid system on big dataset si284, realign
numleaves=2500
modelsize=3000000
lrate=0.004
( # align
  $cuda_cmd exp/tri2a-${numleaves}_nnet4L_modelsize${modelsize}_lrate${lrate}_ali_si284/_align_nnet.log \
    steps/align_nnet.sh data/train_si284 \
    data/lang exp/tri2a-${numleaves}_nnet4L_modelsize${modelsize}_lrate${lrate}_si284 \
    exp/tri2a-${numleaves}_nnet4L_modelsize${modelsize}_lrate${lrate}_ali_si284 || exit 1;

  ####
  #hack-in the GMM models:
  cp exp/tri2a-2500/final.mdl exp/tri2a-2500_nnet4L_modelsize3000000_lrate0.004_ali_si284/final.mdl
  ####

  # train
  $cuda_cmd exp/tri2a-${numleaves}_nnet4L_modelsize${modelsize}_lrate${lrate}_si284_iter2/_train_nnet.log \
    steps/train_nnet_MLP4.sh --model-size $modelsize --lrate $lrate data/train_si284 data/lang \
    exp/tri2a-${numleaves}_nnet4L_modelsize${modelsize}_lrate${lrate}_ali_si284 \
    exp/tri2a-${numleaves}_nnet4L_modelsize${modelsize}_lrate${lrate}_si284_iter2 || exit 1;
  # build graph
  scripts/mkgraph.sh data/lang_test_tgpr exp/tri2a-${numleaves} exp/tri2a-${numleaves}_nnet4L_modelsize${modelsize}_lrate${lrate}_si284_iter2/graph_tgpr || exit 1;
  # decode 
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.1428" steps/decode_nnet.sh exp/tri2a-${numleaves}_nnet4L_modelsize${modelsize}_lrate${lrate}_si284_iter2/graph_tgpr data/test_dev93 exp/tri2a-${numleaves}_nnet4L_modelsize${modelsize}_lrate${lrate}_si284_iter2/decode_tgpr_dev93 || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.1428" steps/decode_nnet.sh exp/tri2a-${numleaves}_nnet4L_modelsize${modelsize}_lrate${lrate}_si284_iter2/graph_tgpr data/test_eval92 exp/tri2a-${numleaves}_nnet4L_modelsize${modelsize}_lrate${lrate}_si284_iter2/decode_tgpr_eval92 || exit 1;
) &




