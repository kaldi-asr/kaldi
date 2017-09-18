#!/bin/bash -e

# Copyright 2014 QCRI (author: Ahmed Ali)
# Apache 2.0

. path.sh
. cmd.sh   ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
num_jobs=120
num_decode_jobs=40

#NB: You can add whatever number of copora you like. The supported extensions 
#NB: (formats) are wav and flac. Flac will be converted using sox and in contrast
#NB: with the old approach, the conversion will be on-the-fly and one-time-only
#NB: during the parametrization.

#NB: Text corpora scpecification. We support either tgz files, which are unpacked
#NB: or just plain (already unpacked) directories. The list of transcript is then
#NB: obtained using find command

#Make sure you edit this section to reflect whers you keep the LDC data on your cluster

#This is CLSP configuration. We add the 2014 GALE data. We got around 2 % 
#improvement just by including it. The gain might be large if someone would tweak
# the number of leaves and states and so on.

#audio=(
#  /export/corpora/LDC/LDC2013S02/
#  /export/corpora/LDC/LDC2013S07/
#  /export/corpora/LDC/LDC2014S07/
#)
#text=(
#  /export/corpora/LDC/LDC2013T17
#  /export/corpora/LDC/LDC2013T04
#  /export/corpora/LDC/LDC2014T17
#)

audio=(
  /data/sls/scratch/amali/data/GALE/LDC2013S02
  /data/sls/scratch/amali/data/GALE/LDC2013S07
  /data/sls/scratch/amali/data/GALE/LDC2014S07
)
text=(
  /data/sls/scratch/amali/data/GALE/LDC2013T17.tgz
  /data/sls/scratch/amali/data/GALE/LDC2013T04.tgz
  /data/sls/scratch/amali/data/GALE/LDC2014T17.tgz
)

galeData=GALE
#prepare the data
#split train dev test 
#prepare lexicon and LM 

# You can run the script from here automatically, but it is recommended to run the data preparation,
# and features extraction manually and and only once.
# By copying and pasting into your shell.

#copy the audio files to local folder wav and convet flac files to wav
local/gale_data_prep_audio.sh  "${audio[@]}" $galeData || exit 1;

#get the transcription and remove empty prompts and all noise markers  
local/gale_data_prep_txt.sh  "${text[@]}" $galeData || exit 1;

# split the data to reports and conversational and for each class will have rain/dev and test
local/gale_data_prep_split.sh $galeData  || exit 1;

# get all Arabic grapheme dictionaries and add silence and UNK
local/gale_prep_grapheme_dict.sh  || exit 1;


#prepare the langauge resources
utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang   || exit 1;

# LM training
local/gale_train_lms.sh || exit 1;

local/gale_format_data.sh  || exit 1;
# G compilation, check LG composition

# Now make MFCC features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
mfccdir=mfcc

for x in train test ; do
  steps/make_mfcc.sh --cmd "$train_cmd" --nj $num_jobs \
    data/$x exp/make_mfcc/$x $mfccdir
  utils/fix_data_dir.sh data/$x # some files fail to get mfcc for many reasons
  steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir
done


# Here we start the AM

# Let's create a subset with 10k segments to make quick flat-start training:
utils/subset_data_dir.sh data/train 10000 data/train.10K || exit 1;

# Train monophone models on a subset of the data, 10K segment
# Note: the --boost-silence option should probably be omitted by default
steps/train_mono.sh --nj 40 --cmd "$train_cmd" \
  data/train.10K data/lang exp/mono || exit 1;


# Get alignments from monophone system.
steps/align_si.sh --nj $num_jobs --cmd "$train_cmd" \
  data/train data/lang exp/mono exp/mono_ali || exit 1;

# train tri1 [first triphone pass]
steps/train_deltas.sh --cmd "$train_cmd" \
  2500 30000 data/train data/lang exp/mono_ali exp/tri1 || exit 1;

# First triphone decoding
utils/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph
steps/decode.sh  --nj $num_decode_jobs --cmd "$decode_cmd" \
  exp/tri1/graph data/test exp/tri1/decode
  
steps/align_si.sh --nj $num_jobs --cmd "$train_cmd" \
  data/train data/lang exp/tri1 exp/tri1_ali || exit 1;

# Train tri2a, which is deltas+delta+deltas
steps/train_deltas.sh --cmd "$train_cmd" \
  3000 40000 data/train data/lang exp/tri1_ali exp/tri2a || exit 1;

# tri2a decoding
utils/mkgraph.sh data/lang_test exp/tri2a exp/tri2a/graph
steps/decode.sh --nj $num_decode_jobs --cmd "$decode_cmd" \
  exp/tri2a/graph data/test exp/tri2a/decode

# train and decode tri2b [LDA+MLLT]
steps/train_lda_mllt.sh --cmd "$train_cmd" 4000 50000 \
  data/train data/lang exp/tri1_ali exp/tri2b || exit 1;

utils/mkgraph.sh data/lang_test exp/tri2b exp/tri2b/graph
steps/decode.sh --nj $num_decode_jobs --cmd "$decode_cmd" \
  exp/tri2b/graph data/test exp/tri2b/decode

# Align all data with LDA+MLLT system (tri2b)
steps/align_si.sh --nj $num_jobs --cmd "$train_cmd" \
  --use-graphs true data/train data/lang exp/tri2b exp/tri2b_ali  || exit 1;


# From 2b system, train 3b which is LDA + MLLT + SAT.
steps/train_sat.sh --cmd "$train_cmd" \
  5000 100000 data/train data/lang exp/tri2b_ali exp/tri3b || exit 1;

utils/mkgraph.sh data/lang_test exp/tri3b exp/tri3b/graph
steps/decode_fmllr.sh --nj $num_decode_jobs --cmd \
  "$decode_cmd" exp/tri3b/graph data/test exp/tri3b/decode

# From 3b system, align all data.
steps/align_fmllr.sh --nj $num_jobs --cmd "$train_cmd" \
  data/train data/lang exp/tri3b exp/tri3b_ali || exit 1;
  

# nnet3 cross-entropy 
local/nnet3/run_tdnn.sh #tdnn recipe:
local/nnet3/run_lstm.sh --stage 12  #lstm recipe (we skip ivector training)

# chain lattice-free 
local/chain/run_tdnn.sh      #tdnn recipe:
local/chain/run_tdnn_lstm.sh #tdnn-lstm recipe:

time=$(date +"%Y-%m-%d-%H-%M-%S")

#get detailed WER; reports, conversational and combined
local/split_wer.sh $galeData > RESULTS.details.$USER.$time # to make sure you keep the results timed and owned

echo training succedded
exit 0

#TODO:
#LM (4-gram and RNN) rescoring
#combine lattices
#dialect detection





